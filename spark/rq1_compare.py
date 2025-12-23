#!/usr/bin/env python3
"""
RQ1 Compare: Stress vs Non-Stress
--------------------------------
Computes liquidity/volatility metrics by tier for:
- Stress periods (BTC volatility in top X%)
- Non-stress periods (everything else)

Writes:
- Parquet summary to results (or local folder when IS_LOCAL=True)
- CSV summary (single file) for quick viewing

Why each step (report-friendly):
- We define stress using BTC volatility as BTC is market-wide benchmark.
- We assign tiers (Large vs Small cap) to compare market quality under stress.
- We aggregate multiple complementary liquidity/volatility proxies:
  volume (activity), zero-volume (market breakdown), Amihud (price impact),
  Parkinson (range-based vol), abs return (directionless volatility).
"""

import os
from pyspark.sql import SparkSession, functions as F, Window

# ---- config import (works whether config.py is at repo root or under spark/) ----
try:
    from config import IS_LOCAL, DATA_DERIVED, DATA_RESULTS, DATA_RAW_LOCAL, DATA_RAW
    from config import raw_path  # if you already have it
except Exception:
    # fallback if your config is spark/config.py and you run from repo root
    from spark.config import IS_LOCAL, DATA_DERIVED, DATA_RESULTS, DATA_RAW_LOCAL, DATA_RAW  # type: ignore
    from spark.config import raw_path  # type: ignore


# -------------------- SETTINGS --------------------
STRESS_Q = float(os.environ.get("STRESS_Q", "0.95"))  # BTC stress percentile threshold
LOCAL_MAX_FILES = int(os.environ.get("LOCAL_MAX_FILES", "0"))  # 0 = all local files

# Large-cap definition (simple, explainable). You can edit this list anytime.
# For the report: "large-cap proxies are the most traded benchmark assets on Binance"
LARGE_CAP_BASE = set(os.environ.get(
    "LARGE_CAP_BASE",
    "BTC,ETH,BNB,XRP,SOL,ADA,DOGE,TRX,DOT,MATIC,LTC,LINK,AVAX,ATOM"
).split(","))


def derived_features_path() -> str:
    # features_1m output location
    return "data/derived/features_1m" if IS_LOCAL else f"{DATA_DERIVED}/features_1m"


def results_rq1_path() -> str:
    # where to store RQ1 outputs
    return "data/results/rq1_compare" if IS_LOCAL else f"{DATA_RESULTS}/rq1_compare"


def ensure_spark(app: str) -> SparkSession:
    builder = SparkSession.builder.appName(app)

    # Local convenience; on cluster spark-submit will override master anyway
    if IS_LOCAL:
        builder = builder.master("local[*]")

    # Practical defaults (safe on both)
    builder = (
        builder
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
    )
    return builder.getOrCreate()


def select_input_files_for_local(path: str) -> str:
    """
    If LOCAL_MAX_FILES > 0, read only that many parquet files locally.
    This prevents your laptop from trying to read 1000 files at once.
    """
    if not IS_LOCAL or LOCAL_MAX_FILES <= 0:
        return path

    # Path is a folder with many parquet files
    # We'll list them with Hadoop filesystem API via Spark
    spark = SparkSession.getActiveSession()
    if spark is None:
        return path

    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hconf)
    p = jvm.org.apache.hadoop.fs.Path(path)

    if not fs.exists(p):
        return path

    statuses = fs.listStatus(p)
    # keep only parquet files
    files = []
    for st in statuses:
        name = st.getPath().getName()
        if name.endswith(".parquet"):
            files.append(st.getPath().toString())

    files = sorted(files)[:LOCAL_MAX_FILES]
    if not files:
        return path

    print(f"Local mode: reading {len(files)} parquet files (LOCAL_MAX_FILES={LOCAL_MAX_FILES})")
    # spark.read.parquet can accept multiple paths
    return ",".join(files)


def main():
    spark = ensure_spark("rq1_compare_stress_vs_nonstress")

    in_path = derived_features_path()
    in_path = select_input_files_for_local(in_path)

    print(f"Reading features_1m from: {in_path}")

    df = spark.read.parquet(in_path)

    # -------------------- STEP 1: Define tiers --------------------
    # Why: we need a consistent grouping variable (Large vs Small).
    df = df.withColumn(
        "tier",
        F.when(F.col("base_asset").isin([x.strip() for x in LARGE_CAP_BASE]), F.lit("LARGE_CAP"))
         .otherwise(F.lit("SMALL_CAP"))
    )

    # -------------------- STEP 2: Define market stress using BTC volatility --------------------
    # Why: BTC is the primary market benchmark; its volatility spikes represent market-wide stress.
    # We prefer BTC-USDT if present; otherwise any BTC-*.
    btc = (
        df.filter(F.col("symbol").startswith("BTC-"))
          .withColumn(
              "btc_pair_rank",
              F.when(F.col("symbol") == F.lit("BTC-USDT"), F.lit(0)).otherwise(F.lit(1))
          )
    )

    # pick one BTC row per minute using rank preference
    w = Window.partitionBy("open_time").orderBy("btc_pair_rank")
    btc_1 = (
        btc.withColumn("rn", F.row_number().over(w))
           .filter(F.col("rn") == 1)
           .select("open_time", F.col("parkinson_var_1m").alias("btc_parkinson_var_1m"))
    )

    # compute stress threshold (percentile)
    threshold_row = btc_1.select(F.expr(f"percentile_approx(btc_parkinson_var_1m, {STRESS_Q}, 10000) as thr")).collect()
    thr = threshold_row[0]["thr"] if threshold_row else None

    print(f"BTC stress threshold ({int(STRESS_Q*100)}th percentile): {thr}")

    # mark stress minutes
    btc_1 = btc_1.withColumn("is_stress", F.col("btc_parkinson_var_1m") >= F.lit(thr))

    # join stress flag to all assets by minute
    df = df.join(btc_1.select("open_time", "is_stress"), on="open_time", how="left")

    # -------------------- STEP 3: Aggregate metrics by (tier, is_stress) --------------------
    # Why each metric:
    # - quote_asset_volume: activity/liquidity proxy in quote currency (USDT/BUSD/etc)
    # - zero_volume_ratio: market "breakdown" frequency
    # - amihud_illiq: price impact proxy (higher = worse liquidity)
    # - parkinson_var_1m: range-based volatility proxy
    # - abs(log_return): directionless volatility magnitude proxy
    agg = (
        df.groupBy("tier", "is_stress")
          .agg(
              F.avg("quote_asset_volume").alias("avg_quote_volume"),
              F.expr("percentile_approx(quote_asset_volume, 0.5, 10000)").alias("median_quote_volume"),
              F.avg("amihud_illiq").alias("avg_amihud"),
              F.expr("percentile_approx(amihud_illiq, 0.95, 10000)").alias("p95_amihud"),
              F.avg(F.col("zero_volume_flag").cast("double")).alias("zero_volume_ratio"),
              F.avg("parkinson_var_1m").alias("avg_parkinson_vol"),
              F.expr("percentile_approx(parkinson_var_1m, 0.95, 10000)").alias("p95_parkinson_vol"),
              F.avg(F.abs("log_return")).alias("avg_abs_return"),
          )
          .orderBy("tier", "is_stress")
    )

    print("RQ1 Compare Results (Stress vs Non-Stress):")
    agg.show(50, truncate=False)

    out_path = results_rq1_path()
    print(f"Writing RQ1 compare parquet to: {out_path}")
    agg.write.mode("overwrite").parquet(out_path)

    # Also write a single CSV for convenience (easy to open)
    csv_out = out_path + "_csv"
    print(f"Writing RQ1 compare CSV to: {csv_out}")
    (
        agg.coalesce(1)
           .write.mode("overwrite")
           .option("header", "true")
           .csv(csv_out)
    )

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
