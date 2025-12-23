"""
features_1m.py

Goal:
- Build a single "features_1m" table from raw 1-minute Binance candles (Parquet per pair).
- Works in BOTH modes:
  (1) Local dev: reads only a small subset of files from ./binance-dataset
  (2) Cluster: reads full dataset from HDFS and writes to HDFS

Why these steps (report-friendly):
1) Attach identifiers (symbol/base/quote) -> raw rows don't include the trading pair; it's in filename.
2) Compute returns + liquidity/activity proxies -> inputs for RQ1/RQ2/RQ3.
3) Write derived parquet in a controlled file count -> avoid HDFS small-file problems and avoid shuffle OOMs.
"""

from __future__ import annotations

import os
import glob
import math
from typing import List

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.window import Window

# Keep consistent with your earlier fix: "from config import ..."
from config import raw_path, DATA_DERIVED, IS_LOCAL


# -----------------------------
# Spark session
# -----------------------------
def build_spark(app_name: str) -> SparkSession:
    """
    Local:
    - Run in local[*] mode (fast iteration).
    - Slightly higher driver memory if your laptop can handle it.
    Cluster:
    - Do not set .master() (spark-submit/YARN handles it).
    - Keep shuffle partitions moderate.
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "400")
    )

    if IS_LOCAL:
        builder = (
            builder.master("local[*]")
            .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "6g"))
            .config("spark.sql.shuffle.partitions", "32")
        )

    spark = builder.getOrCreate()

    # Make output readable: still shows real failures
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "ERROR"))

    return spark


# -----------------------------
# Read raw data + attach symbol
# -----------------------------
def read_raw_with_symbol(spark: SparkSession, in_path: str) -> DataFrame:
    """
    Local mode:
    - Avoid reading 1000 pairs (too big for a laptop).
    - Read N files and union; attach 'symbol' as a constant per file (no input_file_name overhead).

    Cluster mode:
    - Robust & scalable approach: read per file from HDFS list, attach symbol without input_file_name().
      This avoids huge per-row file-path metadata overhead and is usually more stable.
    """
    print("Reading from:", in_path)

    if IS_LOCAL:
        # Local folder (e.g., "binance-dataset")
        max_files = int(os.environ.get("LOCAL_MAX_FILES", "10"))
        files = sorted(glob.glob(os.path.join(in_path, "*.parquet")))[:max_files]
        if not files:
            raise FileNotFoundError(f"No parquet files found in local folder: {in_path}")

        print(f"Local mode: reading {len(files)} parquet files (LOCAL_MAX_FILES={max_files})")

        dfs: List[DataFrame] = []
        for p in files:
            sym = os.path.basename(p).replace(".parquet", "")
            dfs.append(spark.read.parquet(p).withColumn("symbol", F.lit(sym)))

        df = dfs[0]
        for d in dfs[1:]:
            df = df.unionByName(d)

        return (
            df.withColumn("base_asset", F.split(F.col("symbol"), "-").getItem(0))
              .withColumn("quote_asset", F.split(F.col("symbol"), "-").getItem(1))
        )

    # -------- Cluster mode (HDFS) --------
    # Read file paths from HDFS and union with symbol literal.
    # This avoids expensive input_file_name() and helps stability during wide shuffles/writes.
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    Path = sc._jvm.org.apache.hadoop.fs.Path

    p = Path(in_path)  # should be like hdfs:///user/.../raw
    if not fs.exists(p):
        raise FileNotFoundError(f"HDFS path does not exist: {in_path}")

    # List only parquet files
    it = fs.listStatus(p)
    files: List[str] = []
    for st in it:
        name = st.getPath().getName()
        if name.endswith(".parquet"):
            files.append(st.getPath().toString())

    if not files:
        raise FileNotFoundError(f"No parquet files found under HDFS path: {in_path}")

    print(f"Cluster mode: found {len(files)} parquet files in HDFS raw folder")

    # Union all files, adding symbol literal based on filename
    # (This is a loop, but Spark reads lazily; it builds a logical plan.)
    dfs: List[DataFrame] = []
    for fp in files:
        sym = fp.split("/")[-1].replace(".parquet", "")
        dfs.append(spark.read.parquet(fp).withColumn("symbol", F.lit(sym)))

    df = dfs[0]
    for d in dfs[1:]:
        df = df.unionByName(d)

    return (
        df.withColumn("base_asset", F.split(F.col("symbol"), "-").getItem(0))
          .withColumn("quote_asset", F.split(F.col("symbol"), "-").getItem(1))
    )


# -----------------------------
# Feature engineering
# -----------------------------
def add_features(df: DataFrame) -> DataFrame:
    """
    Features (for report):
    - log_return: standard for volatility comparisons across assets.
    - taker_ratio: aggressor buy fraction -> order-flow imbalance.
    - amihud_illiq: price impact per traded value -> illiquidity proxy.
    - zero_volume_flag: identifies illiquid minutes (esp. small caps).
    - parkinson_var_1m: intrabar volatility proxy using high/low.
    """
    w = Window.partitionBy("symbol").orderBy(F.col("open_time"))

    # log return ln(C_t / C_{t-1})
    df = df.withColumn("close_lag", F.lag("close").over(w))
    df = df.withColumn(
        "log_return",
        F.when(
            (F.col("close_lag").isNull()) | (F.col("close_lag") <= 0) | (F.col("close") <= 0),
            F.lit(None),
        ).otherwise(F.log(F.col("close") / F.col("close_lag")))
    ).drop("close_lag")

    # taker buy ratio (quote-based)
    df = df.withColumn(
        "taker_ratio",
        F.when(
            F.col("quote_asset_volume") > 0,
            F.col("taker_buy_quote_asset_volume") / F.col("quote_asset_volume")
        ).otherwise(F.lit(None))
    )

    # Amihud illiquidity proxy
    df = df.withColumn(
        "amihud_illiq",
        F.when(
            (F.col("quote_asset_volume") > 0) & F.col("log_return").isNotNull(),
            F.abs(F.col("log_return")) / F.col("quote_asset_volume")
        ).otherwise(F.lit(None))
    )

    # flag dead minutes
    df = df.withColumn(
        "zero_volume_flag",
        (F.col("quote_asset_volume") == 0) | (F.col("number_of_trades") == 0)
    )

    # Parkinson variance for 1-minute bar
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
    Keep the dataset compact:
    - We retain OHLC and key raw volumes for later analyses/regime detection.
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
# Output writing (OOM-safe)
# -----------------------------
def write_output(df: DataFrame) -> str:
    """
    Critical fix:
    - DO NOT use repartition() on the full dataset before writing -> it forces a full shuffle.
      Shuffles are where executor churn, FetchFailed, and OOM often happen.

    We use:
    - coalesce(N): reduces file count WITHOUT shuffle (much safer)
    - maxRecordsPerFile: avoids too many small files
    """
    out_path = "data/derived/features_1m" if IS_LOCAL else f"{DATA_DERIVED}/features_1m"
    print("Writing:", out_path)

    if IS_LOCAL:
        final = df.coalesce(8)
        (final.write
              .mode("overwrite")
              .parquet(out_path))
        return out_path

    # Cluster: choose a moderate output file count; tune if needed.
    # Coalesce avoids shuffle. 400 is a reasonable start for 35GB.
    n_out = int(os.environ.get("COALESCE_OUT", "400"))
    final = df.coalesce(n_out)

    (final.write
          .mode("overwrite")
          .option("maxRecordsPerFile", os.environ.get("MAX_RECORDS_PER_FILE", "2000000"))
          .parquet(out_path))

    return out_path


def main():
    spark = build_spark("binance_features_1m")

    # 1) Load raw
    df_raw = read_raw_with_symbol(spark, raw_path())

    # 2) Features
    df_feat = add_features(df_raw)

    # 3) Select schema
    out_df = select_output_columns(df_feat)

    # 4) Write derived features_1m (OOM-safe)
    out_path = write_output(out_df)

    # Local preview only
    if IS_LOCAL:
        print("Preview:")
        spark.read.parquet(out_path).show(5, truncate=False)

    print("Done.")
    spark.stop()


if __name__ == "__main__":
    main()
