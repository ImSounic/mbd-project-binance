#!/usr/bin/env python3
"""
RQ2: Shock propagation from BTC/ETH trading intensity to altcoin liquidity/volatility.

We study how extreme trading-intensity shocks in BTC and ETH propagate
to volatility and liquidity of smaller cryptocurrencies at different time lags.
"""

import os
from pyspark.sql import SparkSession, functions as F

# -------------------- Tunables (safe defaults for cluster) --------------------
MAX_LAG = int(os.environ.get("MAX_LAG", "120"))
SHOCK_Q = float(os.environ.get("SHOCK_Q", "0.99"))
SHUFFLE_PARTS = int(os.environ.get("SHUFFLE_PARTS", "120"))
OUT_PARTS = int(os.environ.get("OUT_PARTS", "16"))
TOP_N_PRINT = int(os.environ.get("TOP_N_PRINT", "20"))

# Optional debugging filter
RECEIVER_FILTER = os.environ.get("RECEIVER_FILTER", "").strip()

from config import derived_path


def ensure_spark(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )
    spark.conf.set("spark.sql.shuffle.partitions", str(SHUFFLE_PARTS))
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    return spark


def main():
    spark = ensure_spark("rq2_shock_propagation")

    # ----------- FIXED PATH HANDLING (IMPORTANT) -----------
    base_derived = derived_path()
    in_path = f"{base_derived}/features_1m"
    out_base = f"{base_derived}/rq2_results"

    print("Reading features_1m from:", in_path)

    df = (
        spark.read.parquet(in_path)
        .select(
            "open_time",
            "symbol",
            "number_of_trades",
            "amihud_illiq",
            "parkinson_var_1m",
        )
        .withColumn("open_time", F.col("open_time").cast("timestamp"))
    )

    if RECEIVER_FILTER:
        keep = [s.strip() for s in RECEIVER_FILTER.split(",") if s.strip()]
        df = df.filter(F.col("symbol").isin(keep + ["BTC-USDT", "ETH-USDT"]))
        print("Receiver filter enabled:", keep)

    # -------------------- BTC & ETH shock definition --------------------
    btc = df.filter(F.col("symbol") == "BTC-USDT") \
            .select("open_time", F.col("number_of_trades").alias("btc_trades"))
    eth = df.filter(F.col("symbol") == "ETH-USDT") \
            .select("open_time", F.col("number_of_trades").alias("eth_trades"))

    btc_thr = btc.approxQuantile("btc_trades", [SHOCK_Q], 0.001)[0]
    eth_thr = eth.approxQuantile("eth_trades", [SHOCK_Q], 0.001)[0]

    print(f"Using driver BTC symbol: BTC-USDT")
    print(f"Using driver ETH symbol: ETH-USDT")
    print(f"BTC trades shock threshold (q={SHOCK_Q}): {btc_thr}")
    print(f"ETH trades shock threshold (q={SHOCK_Q}): {eth_thr}")

    btc_shocks = btc.withColumn(
        "btc_shock", (F.col("btc_trades") >= F.lit(btc_thr)).cast("int")
    ).select("open_time", "btc_shock")

    eth_shocks = eth.withColumn(
        "eth_shock", (F.col("eth_trades") >= F.lit(eth_thr)).cast("int")
    ).select("open_time", "eth_shock")

    shocks = (
        btc_shocks.join(eth_shocks, on="open_time", how="outer")
        .na.fill(0, ["btc_shock", "eth_shock"])
        .withColumn("any_shock", F.greatest("btc_shock", "eth_shock"))
        .select("open_time", "any_shock")
    )

    receivers = df.filter(~F.col("symbol").isin("BTC-USDT", "ETH-USDT"))

    print(f"Scanning lags 0..{MAX_LAG} minutes")

    results = []
    for lag in [0, 1, 5, 15, 30, 60, 120]:
        if lag > MAX_LAG:
            continue

        shifted = receivers.withColumn(
            "t_key", F.expr(f"open_time - INTERVAL {lag} MINUTES")
        )

        joined = (
            shifted.join(shocks, shifted.t_key == shocks.open_time, "inner")
            .drop(shocks.open_time)
        )

        agg = (
            joined.groupBy("symbol")
            .agg(
                F.avg(F.when(F.col("any_shock") == 1, F.col("parkinson_var_1m"))).alias("vol_on_shock"),
                F.avg(F.when(F.col("any_shock") == 0, F.col("parkinson_var_1m"))).alias("vol_on_noshock"),
                F.avg(F.when(F.col("any_shock") == 1, F.col("amihud_illiq"))).alias("illiq_on_shock"),
                F.avg(F.when(F.col("any_shock") == 0, F.col("amihud_illiq"))).alias("illiq_on_noshock"),
                F.count(F.when(F.col("any_shock") == 1, 1)).alias("n_shock_obs"),
                F.count(F.when(F.col("any_shock") == 0, 1)).alias("n_noshock_obs"),
            )
            .withColumn("lag_min", F.lit(lag))
            .withColumn("delta_vol", F.col("vol_on_shock") - F.col("vol_on_noshock"))
            .withColumn("delta_illiq", F.col("illiq_on_shock") - F.col("illiq_on_noshock"))
        )

        results.append(agg)
        print(f"  lag={lag} done")

    final = results[0]
    for r in results[1:]:
        final = final.unionByName(r, allowMissingColumns=True)

    final = final.filter(
        (F.col("n_shock_obs") > 50) & (F.col("n_noshock_obs") > 200)
    )

    out_path = f"{out_base}/lag_effects"
    print("Writing lag effects to:", out_path)

    (
        final
        .coalesce(OUT_PARTS)
        .write.mode("overwrite")
        .parquet(out_path)
    )

    print("RQ2 sample (lag=0, strongest volatility response):")
    (
        final.filter(F.col("lag_min") == 0)
        .orderBy(F.desc("delta_vol"))
        .select("symbol", "delta_vol", "delta_illiq", "n_shock_obs")
        .limit(TOP_N_PRINT)
        .show(truncate=False)
    )

    print("Done.")
    spark.stop()


if __name__ == "__main__":
    main()
