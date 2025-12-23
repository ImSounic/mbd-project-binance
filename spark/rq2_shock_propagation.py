"""
RQ2: Shock propagation from BTC/ETH trading intensity to altcoin liquidity/volatility
"""

import os
from pyspark.sql import SparkSession, functions as F, Window

# ---- config import (cluster-safe) ----
try:
    from config import DATA_DERIVED
except Exception:
    from spark.config import DATA_DERIVED  # type: ignore


# ---------------------- Parameters ----------------------
MAX_LAG = int(os.environ.get("MAX_LAG", "120"))
SHOCK_Q = float(os.environ.get("SHOCK_Q", "0.99"))
TOP_N_SHOW = int(os.environ.get("TOP_N_SHOW", "30"))

DRIVER_BTC_DEFAULT = os.environ.get("DRIVER_BTC", "BTC-USDT")
DRIVER_ETH_DEFAULT = os.environ.get("DRIVER_ETH", "ETH-USDT")


def ensure_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def pick_driver_symbol(df, preferred: str, contains: str) -> str:
    symbols = [r["symbol"] for r in df.select("symbol").distinct().collect()]
    if preferred in symbols:
        return preferred
    for s in symbols:
        if contains in s:
            return s
    return symbols[0]


def main():
    spark = ensure_spark("rq2_shock_propagation")

    # -------------------- INPUT --------------------
    in_path = f"{DATA_DERIVED}/features_1m"
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
        .filter(F.col("open_time").isNotNull())
        .filter(F.col("symbol").isNotNull())
    )

    # -------------------- DRIVER SYMBOLS --------------------
    driver_btc = pick_driver_symbol(df, DRIVER_BTC_DEFAULT, "BTC")
    driver_eth = pick_driver_symbol(df, DRIVER_ETH_DEFAULT, "ETH")

    print("Using driver BTC symbol:", driver_btc)
    print("Using driver ETH symbol:", driver_eth)

    # -------------------- SHOCK DEFINITION --------------------
    drivers = df.filter(F.col("symbol").isin([driver_btc, driver_eth]))

    btc_thr = drivers.filter(F.col("symbol") == driver_btc) \
        .approxQuantile("number_of_trades", [SHOCK_Q], 0.01)[0]

    eth_thr = drivers.filter(F.col("symbol") == driver_eth) \
        .approxQuantile("number_of_trades", [SHOCK_Q], 0.01)[0]

    print(f"BTC trades shock threshold (q={SHOCK_Q}): {btc_thr}")
    print(f"ETH trades shock threshold (q={SHOCK_Q}): {eth_thr}")

    shocks = (
        drivers
        .withColumn(
            "is_shock",
            F.when(
                (F.col("symbol") == driver_btc) & (F.col("number_of_trades") >= btc_thr), 1
            ).when(
                (F.col("symbol") == driver_eth) & (F.col("number_of_trades") >= eth_thr), 1
            ).otherwise(0)
        )
        .filter(F.col("is_shock") == 1)
        .select(
            F.col("open_time").alias("shock_time"),
            F.col("symbol").alias("driver"),
        )
        .persist()
    )

    # -------------------- RECEIVERS --------------------
    receivers = df.filter(~F.col("symbol").isin([driver_btc, driver_eth]))

    print(f"Scanning lags 0..{MAX_LAG} minutes")

    results = None

    for lag in range(MAX_LAG + 1):
        resp = (
            receivers
            .withColumn("shock_time", F.expr(f"open_time - INTERVAL {lag} MINUTES"))
            .select(
                "symbol",
                "shock_time",
                F.col("amihud_illiq").alias("amihud_resp"),
                F.col("parkinson_var_1m").alias("vol_resp"),
            )
        )

        joined = (
            shocks.join(resp, on="shock_time", how="inner")
            .select(
                "driver",
                F.col("symbol").alias("receiver"),
                F.lit(lag).alias("lag"),
                "amihud_resp",
                "vol_resp",
            )
        )

        agg = (
            joined.groupBy("driver", "receiver", "lag")
            .agg(
                F.avg("vol_resp").alias("shock_mean_vol"),
                F.avg("amihud_resp").alias("shock_mean_amihud"),
                F.count("*").alias("n_shock_points"),
            )
        )

        results = agg if results is None else results.unionByName(agg)

        if lag in (0, 1, 5, 15, 30, 60, MAX_LAG):
            print(f"  lag={lag} done")

    # -------------------- BASELINES --------------------
    baseline = (
        receivers.groupBy("symbol")
        .agg(
            F.avg("parkinson_var_1m").alias("baseline_vol"),
            F.avg("amihud_illiq").alias("baseline_amihud"),
        )
        .withColumnRenamed("symbol", "receiver")
    )

    final = (
        results.join(baseline, on="receiver", how="left")
        .withColumn("delta_vol", F.col("shock_mean_vol") - F.col("baseline_vol"))
        .withColumn("delta_amihud", F.col("shock_mean_amihud") - F.col("baseline_amihud"))
    )

    # -------------------- BEST LAG --------------------
    w = Window.partitionBy("driver", "receiver").orderBy(F.desc("delta_vol"))

    best = (
        final
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .orderBy(F.desc("delta_vol"))
    )

    # -------------------- OUTPUT --------------------
    out_base = f"{DATA_DERIVED}/rq2_results"

    print("Writing lag effects to:", f"{out_base}/lag_effects")
    final.write.mode("overwrite").parquet(f"{out_base}/lag_effects")

    print("Writing best lag to:", f"{out_base}/best_lag")
    best.write.mode("overwrite").parquet(f"{out_base}/best_lag")

    print("RQ2 best-lag summary (top rows):")
    best.select(
        "driver", "receiver", "lag", "delta_vol", "delta_amihud", "n_shock_points"
    ).limit(TOP_N_SHOW).show(truncate=False)

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
