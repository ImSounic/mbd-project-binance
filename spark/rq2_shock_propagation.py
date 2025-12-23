"""
RQ2: Shock propagation from BTC/ETH trading intensity to altcoin liquidity/volatility

Key changes (to avoid executor disk-full):
- Limit receivers to top-K by activity (quote volume)
- Write ONLY best_lag output (not the huge lag_effects table)
- Allow tuning via env vars
"""

import os
from pyspark.sql import SparkSession, functions as F, Window

# ---- config import (cluster-safe) ----
try:
    from config import DATA_DERIVED
except Exception:
    from spark.config import DATA_DERIVED  # type: ignore


# ---------------------- Parameters ----------------------
MAX_LAG = int(os.environ.get("MAX_LAG", "120"))              # scan lags 0..MAX_LAG minutes
SHOCK_Q = float(os.environ.get("SHOCK_Q", "0.99"))           # shock threshold quantile
TOP_K_RECEIVERS = int(os.environ.get("TOP_K", "200"))        # limit number of altcoins
TOP_N_SHOW = int(os.environ.get("TOP_N_SHOW", "30"))

DRIVER_BTC_DEFAULT = os.environ.get("DRIVER_BTC", "BTC-USDT")
DRIVER_ETH_DEFAULT = os.environ.get("DRIVER_ETH", "ETH-USDT")


def ensure_spark(app_name: str) -> SparkSession:
    # NOTE: spark.local.dir should be set from spark-submit via --conf
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

    in_path = f"{DATA_DERIVED}/features_1m"
    print("Reading features_1m from:", in_path)

    base = (
        spark.read.parquet(in_path)
        .select(
            "open_time",
            "symbol",
            "quote_asset_volume",
            "number_of_trades",
            "amihud_illiq",
            "parkinson_var_1m",
        )
        .filter(F.col("open_time").isNotNull())
        .filter(F.col("symbol").isNotNull())
    )

    # -------------------- DRIVER SYMBOLS --------------------
    driver_btc = pick_driver_symbol(base, DRIVER_BTC_DEFAULT, "BTC")
    driver_eth = pick_driver_symbol(base, DRIVER_ETH_DEFAULT, "ETH")

    print("Using driver BTC symbol:", driver_btc)
    print("Using driver ETH symbol:", driver_eth)

    # -------------------- LIMIT RECEIVERS (Top-K by activity) --------------------
    # Why: The join size is roughly (#shock_times) * (#receivers). Limiting receivers massively reduces shuffles.
    activity = (
        base.filter(~F.col("symbol").isin([driver_btc, driver_eth]))
        .groupBy("symbol")
        .agg(F.sum("quote_asset_volume").alias("sum_qav"))
        .orderBy(F.desc("sum_qav"))
        .limit(TOP_K_RECEIVERS)
        .select("symbol")
    )

    receivers = (
        base.join(activity, on="symbol", how="inner")
        .select(
            "open_time",
            "symbol",
            "amihud_illiq",
            "parkinson_var_1m",
        )
        .persist()
    )

    print(f"Receivers limited to TOP_K={TOP_K_RECEIVERS} symbols")

    # -------------------- BASELINES (for delta) --------------------
    baseline = (
        receivers.groupBy("symbol")
        .agg(
            F.avg("parkinson_var_1m").alias("baseline_vol"),
            F.avg("amihud_illiq").alias("baseline_amihud"),
        )
        .withColumnRenamed("symbol", "receiver")
        .persist()
    )

    # -------------------- SHOCKS (driver intensity spikes) --------------------
    drivers = (
        base.filter(F.col("symbol").isin([driver_btc, driver_eth]))
        .select("open_time", "symbol", "number_of_trades")
        .persist()
    )

    btc_thr = (
        drivers.filter(F.col("symbol") == driver_btc)
        .approxQuantile("number_of_trades", [SHOCK_Q], 0.01)[0]
    )
    eth_thr = (
        drivers.filter(F.col("symbol") == driver_eth)
        .approxQuantile("number_of_trades", [SHOCK_Q], 0.01)[0]
    )

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

    # -------------------- MAIN LOOP: compute best lag WITHOUT writing huge lag table --------------------
    print(f"Scanning lags 0..{MAX_LAG} minutes")

    best_all = None

    for lag in range(MAX_LAG + 1):
        # Receiver rows that correspond to (shock_time - lag)
        resp = (
            receivers
            .withColumn("shock_time", F.expr(f"open_time - INTERVAL {lag} MINUTES"))
            .select(
                F.col("symbol").alias("receiver"),
                "shock_time",
                F.col("amihud_illiq").alias("amihud_resp"),
                F.col("parkinson_var_1m").alias("vol_resp"),
            )
        )

        joined = (
            shocks.join(resp, on="shock_time", how="inner")
            .select("driver", "receiver", "amihud_resp", "vol_resp")
        )

        agg = (
            joined.groupBy("driver", "receiver")
            .agg(
                F.avg("vol_resp").alias("shock_mean_vol"),
                F.avg("amihud_resp").alias("shock_mean_amihud"),
                F.count("*").alias("n_shock_points"),
            )
            .withColumn("lag", F.lit(lag))
        )

        scored = (
            agg.join(baseline, on="receiver", how="left")
            .withColumn("delta_vol", F.col("shock_mean_vol") - F.col("baseline_vol"))
            .withColumn("delta_amihud", F.col("shock_mean_amihud") - F.col("baseline_amihud"))
            .select("driver", "receiver", "lag", "delta_vol", "delta_amihud", "n_shock_points")
        )

        best_all = scored if best_all is None else best_all.unionByName(scored)

        if lag in (0, 1, 5, 15, 30, 60, MAX_LAG):
            print(f"  lag={lag} done")

    # pick best lag per (driver, receiver) by strongest delta_vol
    w = Window.partitionBy("driver", "receiver").orderBy(F.desc("delta_vol"))
    best = (
        best_all.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .orderBy(F.desc("delta_vol"))
    )

    # -------------------- OUTPUT --------------------
    out_base = f"{DATA_DERIVED}/rq2_results"
    out_best = f"{out_base}/best_lag"

    print("Writing best lag to:", out_best)
    best.write.mode("overwrite").parquet(out_best)

    print("RQ2 best-lag summary (top rows):")
    best.limit(TOP_N_SHOW).show(truncate=False)

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
