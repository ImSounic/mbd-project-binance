"""
features_1m.py

Purpose:
- Transform raw 1-minute candlestick Parquet files into a single analysis-ready table.
- Add identifiers (symbol/base/quote) that are missing in the raw rows (encoded in filenames).
- Compute scale-invariant returns (log returns) and multiple liquidity/activity proxies.
- Produce a derived dataset suitable as a base input for RQ1/RQ2/RQ3.

Design choice :
- Local development should run on a SMALL subset of files (laptop memory constraints).
- Full feature generation runs on the Spark cluster over HDFS.
- We avoid creating thousands of small output files by NOT partitioning output by symbol.
"""

from __future__ import annotations

import os
import glob
import math
from typing import List

from pyspark.sql import SparkSession, DataFrame, functions as F, Window

# Your repo uses config.py at spark/config.py, and you're importing it as "from config import ..."
# Keep that consistent with your ingest_check fix.
from config import raw_path, DATA_DERIVED, IS_LOCAL


# -----------------------------
# Spark session construction
# -----------------------------
def build_spark(app_name: str) -> SparkSession:
    """
    Why we do this:
    - Local mode accelerates development/debugging on small samples.
    - Cluster mode uses spark-submit/yarn configs, so we do not set master there.
    - We optionally increase driver memory locally to reduce out-of-memory errors.
    """
    builder = SparkSession.builder.appName(app_name)

    if IS_LOCAL:
        builder = (
            builder.master("local[*]")
            .config("spark.driver.memory", "6g")          # adjust if needed
            .config("spark.sql.shuffle.partitions", "32") # keep local shuffles manageable
        )

    return builder.getOrCreate()


# -----------------------------
# Reading input in a safe way
# -----------------------------
def read_raw_with_symbol(spark: SparkSession, in_dir: str) -> DataFrame:
    """
    Read raw dataset and attach symbol.

    Local mode strategy:
    - Do NOT load all 1000 pairs on a laptop.
    - Read a limited number of files and union them.
    - Attach 'symbol' as a constant per file (fast, no input_file_name overhead).

    Cluster mode strategy:
    - Read the whole HDFS folder (scalable).
    - Use input_file_name() to recover symbol from filename.
      (Acceptable on cluster resources; can be optimized later.)
    """
    print("Reading from:", in_dir)

    if IS_LOCAL:
        LOCAL_MAX_FILES = int(os.environ.get("LOCAL_MAX_FILES", "10"))  # override via env if you want
        files: List[str] = sorted(glob.glob(os.path.join(in_dir, "*.parquet")))[:LOCAL_MAX_FILES]

        if not files:
            raise FileNotFoundError(f"No parquet files found in local folder: {in_dir}")

        print(f"Local mode: reading {len(files)} parquet files (LOCAL_MAX_FILES={LOCAL_MAX_FILES})")

        dfs: List[DataFrame] = []
        for p in files:
            sym = os.path.basename(p).replace(".parquet", "")
            # Attach symbol per file without input_file_name() to keep memory low.
            dfs.append(spark.read.parquet(p).withColumn("symbol", F.lit(sym)))

        df = dfs[0]
        for d in dfs[1:]:
            df = df.unionByName(d)

        # Derive base/quote identifiers from symbol
        df = (
            df.withColumn("base_asset", F.split(F.col("symbol"), "-").getItem(0))
              .withColumn("quote_asset", F.split(F.col("symbol"), "-").getItem(1))
        )
        return df

    # Cluster mode (HDFS): read full folder and derive symbol from filename
    df = spark.read.parquet(in_dir).withColumn("_file", F.input_file_name())

    # Example file path ends with ".../BTC-USDT.parquet"
    sym = F.regexp_extract(F.col("_file"), r"([^/]+)\.parquet$", 1)

    df = (
        df.withColumn("symbol", sym)
          .withColumn("base_asset", F.split(F.col("symbol"), "-").getItem(0))
          .withColumn("quote_asset", F.split(F.col("symbol"), "-").getItem(1))
          .drop("_file")
    )
    return df


# -----------------------------
# Feature engineering
# -----------------------------
def add_features(df: DataFrame) -> DataFrame:
    """
    Create 1-minute features used across research questions.

    Why these features:
    - log_return: comparable across assets and standard for volatility analysis.
    - number_of_trades: trading intensity / activity measure.
    - quote_asset_volume: traded value proxy for liquidity.
    - taker_ratio: order-flow imbalance proxy (aggressor buy fraction).
    - amihud_illiq: illiquidity proxy (price impact per traded value).
    - parkinson_var_1m: intrabar volatility proxy using high/low.
    - zero_volume_flag: indicates illiquid minutes (common in small caps).
    """
    # Window per symbol for lagging close prices
    w = Window.partitionBy("symbol").orderBy(F.col("open_time"))

    # Log return: ln(C_t / C_{t-1})
    df = df.withColumn("close_lag", F.lag("close").over(w))

    df = df.withColumn(
        "log_return",
        F.when(
            (F.col("close_lag").isNull()) | (F.col("close_lag") <= 0) | (F.col("close") <= 0),
            F.lit(None),
        ).otherwise(F.log(F.col("close") / F.col("close_lag")))
    ).drop("close_lag")

    # Liquidity/activity proxies
    df = df.withColumn(
        "taker_ratio",
        F.when(
            F.col("quote_asset_volume") > 0,
            F.col("taker_buy_quote_asset_volume") / F.col("quote_asset_volume")
        ).otherwise(F.lit(None))
    )

    df = df.withColumn(
        "amihud_illiq",
        F.when(
            (F.col("quote_asset_volume") > 0) & F.col("log_return").isNotNull(),
            F.abs(F.col("log_return")) / F.col("quote_asset_volume")
        ).otherwise(F.lit(None))
    )

    df = df.withColumn(
        "zero_volume_flag",
        (F.col("quote_asset_volume") == 0) | (F.col("number_of_trades") == 0)
    )

    # Parkinson variance per bar: (ln(H/L))^2 / (4 ln(2))
    ln2 = math.log(2.0)
    df = df.withColumn(
        "parkinson_var_1m",
        F.when(
            (F.col("high") > 0) & (F.col("low") > 0) & (F.col("high") >= F.col("low")),
            (F.log(F.col("high") / F.col("low")) ** 2) / F.lit(4.0 * ln2)
        ).otherwise(F.lit(None))
    )

    return df


def select_output_columns(df: DataFrame) -> DataFrame:
    """
    Keep a compact schema to reduce storage and speed up downstream queries.
    We keep OHLC for later regime detection, correlation tests, and debugging.
    """
    return df.select(
        "open_time",
        "symbol", "base_asset", "quote_asset",
        "open", "high", "low", "close",
        "volume", "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
        "log_return",
        "taker_ratio",
        "amihud_illiq",
        "zero_volume_flag",
        "parkinson_var_1m",
    )


# -----------------------------
# Sanity checks + writing output
# -----------------------------
def sanity_checks(df: DataFrame) -> None:
    """
    Lightweight validation (report-friendly).
    We avoid expensive global actions on the full dataset.
    """
    required = {"open_time", "symbol", "quote_asset_volume", "number_of_trades", "log_return"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # small action to confirm non-empty
    if df.limit(1).count() == 0:
        raise RuntimeError("No rows loaded. Check input path and file selection.")


def write_output(df: DataFrame) -> str:
    """
    Output paths:
    - Local: data/derived/features_1m
    - Cluster: hdfs:///user/<owner>/binance/derived/features_1m

    File count control:
    - Local: coalesce to a small number of files for convenience
    - Cluster: repartition to a moderate number to balance parallelism and small-file risk
    """
    out_path = "data/derived/features_1m" if IS_LOCAL else f"{DATA_DERIVED}/features_1m"
    print("Writing:", out_path)

    if IS_LOCAL:
        final = df.coalesce(8)
    else:
        # adjust later depending on cluster performance; this is a safe starting point
        final = df.repartition(200)

    final.write.mode("overwrite").parquet(out_path)
    return out_path


def main():
    spark = build_spark("binance_features_1m")

    # 1) Read raw + attach identifiers
    df_raw = read_raw_with_symbol(spark, raw_path())

    # 2) Features
    df_feat = add_features(df_raw)

    # 3) Select output schema
    out_df = select_output_columns(df_feat)

    # 4) Validate quickly
    sanity_checks(out_df)

    # 5) Write derived dataset
    out_path = write_output(out_df)

    # Optional quick peek (local only)
    if IS_LOCAL:
        print("Preview of derived features_1m:")
        spark.read.parquet(out_path).show(5, truncate=False)

    print("Done.")
    spark.stop()


if __name__ == "__main__":
    main()
