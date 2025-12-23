"""
RQ2: Shock propagation from BTC/ETH trading intensity to altcoin liquidity/volatility.

Research Question:
How do shocks in the trading intensity (number_of_trades) of BTC and ETH propagate
to the liquidity (Amihud illiquidity) and volatility (Parkinson variance) of smaller altcoins,
and what is the typical time lag?

Key idea (why these steps):
1) Use number_of_trades spikes as a proxy for sudden market attention / intensity shocks.
2) Use BTC and ETH as "drivers" because they are the dominant benchmarks.
3) Measure altcoin response using:
   - parkinson_var_1m  (intraminute volatility proxy)
   - amihud_illiq      (illiquidity proxy: |return| / volume proxy)
4) We measure response with a time lag: lag = 0..MAX_LAG minutes.
5) For each lag, compare mean response on "shock minutes" vs "non-shock minutes".
   The difference = propagation effect size.

Performance choices (important on cluster):
- Avoid python loops over symbols with collect() (driver OOM risk)
- Avoid huge cartesian joins.
- Use Spark window functions (lead) to compute future response at different lags.
- Only show/collect small top-N summaries.
"""

import os
from pyspark.sql import SparkSession, functions as F, Window

# ---- config import that works with spark-submit from repo root ----
try:
    # when running from spark/ directory sometimes
    from config import DATA_DERIVED, raw_path, derived_path
except Exception:
    # when spark-submit resolves modules differently
    from spark.config import DATA_DERIVED, raw_path, derived_path  # type: ignore


# ---------------------- Parameters (easy to tweak) ----------------------
MAX_LAG = int(os.environ.get("MAX_LAG", "120"))          # minutes
SHOCK_Q = float(os.environ.get("SHOCK_Q", "0.99"))       # top 1% trades as shocks
TOP_N_SHOW = int(os.environ.get("TOP_N_SHOW", "30"))     # print only top N rows

# Drivers: try these first; if missing, we auto-pick by contains("BTC") etc.
DRIVER_BTC_DEFAULT = os.environ.get("DRIVER_BTC", "BTC-USDT")
DRIVER_ETH_DEFAULT = os.environ.get("DRIVER_ETH", "ETH-USDT")


def ensure_spark(app_name: str) -> SparkSession:
    """
    Create SparkSession with reasonable defaults.
    Why: Make local+cluster runs consistent and reduce accidental huge shuffles.
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.files.maxPartitionBytes", "134217728")  # 128MB
    )

    # If running locally, force local master. On cluster, spark-submit manages it.
    if os.environ.get("IS_LOCAL", "").lower() in ("1", "true", "yes"):
        builder = builder.master("local[*]")

    return builder.getOrCreate()


def pick_driver_symbol(df, preferred: str, fallback_contains: str) -> str:
    """
    Choose driver symbol robustly.
    Why: Some datasets may use different quote assets, so we fall back gracefully.
    """
    symbols = [r["symbol"] for r in df.select("symbol").distinct().collect()]
    if preferred in symbols:
        return preferred
    # fallback: first symbol containing BTC or ETH
    for s in symbols:
        if fallback_contains in s:
            return s
    # last resort: just pick the first (should not happen)
    return symbols[0]


def main():
    spark = ensure_spark("rq2_shock_propagation")

    # ---- Input: features_1m produced earlier ----
    in_path = derived_path("features_1m")
    print("Reading features_1m from:", in_path)

    # keep only what we need (why: smaller memory, faster shuffles)
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

    # Basic sanitation (why: avoid NaN/Inf issues and meaningless rows)
    df = df.filter(F.col("open_time").isNotNull()).filter(F.col("symbol").isNotNull())

    # Identify actual driver symbols in your dataset
    driver_btc = pick_driver_symbol(df, DRIVER_BTC_DEFAULT, "BTC")
    driver_eth = pick_driver_symbol(df, DRIVER_ETH_DEFAULT, "ETH")
    print("Using driver BTC symbol:", driver_btc)
    print("Using driver ETH symbol:", driver_eth)

    # -------------------- STEP 1: Build shock flags for drivers --------------------
    # Why: define "shock minute" as extreme spikes in number_of_trades (top q percentile).
    drivers = df.filter(F.col("symbol").isin([driver_btc, driver_eth]))

    # compute thresholds
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
        drivers.select("open_time", "symbol", "number_of_trades")
        .withColumn(
            "is_shock",
            F.when(
                (F.col("symbol") == driver_btc) & (F.col("number_of_trades") >= F.lit(btc_thr)),
                F.lit(1),
            ).when(
                (F.col("symbol") == driver_eth) & (F.col("number_of_trades") >= F.lit(eth_thr)),
                F.lit(1),
            ).otherwise(F.lit(0))
        )
        .filter(F.col("is_shock") == 1)
        .select(
            F.col("open_time").alias("shock_time"),
            F.col("symbol").alias("driver"),
            "is_shock",
        )
    )

    # Cache shocks because we reuse it many times
    shocks = shocks.persist()

    # -------------------- STEP 2: Prepare receiver panel + future values (lead) --------------------
    # Why: propagation means receiver reacts AFTER the shock.
    # Instead of joining receiver to shifted times repeatedly via python loops,
    # we compute "future response at lag L" using lead(...) within each symbol.
    w = Window.partitionBy("symbol").orderBy("open_time")

    base = (
        df.filter(~F.col("symbol").isin([driver_btc, driver_eth]))
        .select("open_time", "symbol", "amihud_illiq", "parkinson_var_1m")
    )

    # For RQ2 we only need response variables at t+lag
    # We'll build results by iterating lags, but never collecting symbol lists to the driver.
    print(f"Scanning lags 0..{MAX_LAG} minutes over receivers...")

    results = None

    # We join shocks -> receivers by aligning:
    # shock at time t  influences receiver at time t+lag
    # so we build receiver_future where response_time = open_time, and shock_time = open_time - lag
    for lag in range(0, MAX_LAG + 1):
        # receiver response at t (current row), but we want response at t+lag relative to shock_time
        # easiest: for each receiver row at time t, define shock_time = t - lag
        receiver_at_response = (
            base
            .withColumn("lag", F.lit(lag))
            .withColumn("shock_time", F.expr(f"open_time - INTERVAL {lag} MINUTES"))
            .select(
                "symbol",
                "lag",
                "shock_time",
                F.col("amihud_illiq").alias("amihud_resp"),
                F.col("parkinson_var_1m").alias("vol_resp"),
            )
        )

        joined = (
            shocks.join(receiver_at_response, on="shock_time", how="inner")
            .select("driver", F.col("symbol").alias("receiver"), "lag", "amihud_resp", "vol_resp")
        )

        # Aggregate effect size for this lag (why: compare mean response on shock minutes)
        # Note: we're not computing non-shock baseline here; to get baseline we compute overall mean later.
        agg = (
            joined.groupBy("driver", "receiver", "lag")
            .agg(
                F.avg("vol_resp").alias("shock_mean_vol"),
                F.avg("amihud_resp").alias("shock_mean_amihud"),
                F.count("*").alias("n_shock_points"),
            )
        )

        results = agg if results is None else results.unionByName(agg)

        # small progress print (doesn't spam)
        if lag in (0, 1, 5, 15, 30, 60, 120) or lag == MAX_LAG:
            print(f"  computed lag={lag}")

    # -------------------- STEP 3: Add non-shock baselines per receiver --------------------
    # Why: we want a difference: (shock mean) - (normal mean)
    baseline = (
        base.groupBy("symbol")
        .agg(
            F.avg("parkinson_var_1m").alias("baseline_mean_vol"),
            F.avg("amihud_illiq").alias("baseline_mean_amihud"),
        )
        .withColumnRenamed("symbol", "receiver")
    )

    final = (
        results.join(baseline, on="receiver", how="left")
        .withColumn("delta_vol", F.col("shock_mean_vol") - F.col("baseline_mean_vol"))
        .withColumn("delta_amihud", F.col("shock_mean_amihud") - F.col("baseline_mean_amihud"))
    )

    # -------------------- STEP 4: pick best lag per (driver, receiver) --------------------
    # Why: we want "typical time lag" -> lag with strongest volatility response.
    w_best = Window.partitionBy("driver", "receiver").orderBy(F.desc("delta_vol"))

    best = (
        final.withColumn("rn", F.row_number().over(w_best))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .orderBy(F.desc("delta_vol"))
    )

    # ---- Output paths ----
    out_dir = derived_path("rq2_results")
    out_effects = f"{out_dir}/lag_effects"
    out_best = f"{out_dir}/best_lag"

    print("Writing lag effects to:", out_effects)
    final.write.mode("overwrite").parquet(out_effects)

    print("Writing best lags to:", out_best)
    best.write.mode("overwrite").parquet(out_best)

    # Keep driver safe: only show top-N
    print("RQ2 best-lag summary (top by strongest volatility response):")
    best.select(
        "driver", "receiver", "lag", "delta_vol", "delta_amihud", "n_shock_points"
    ).limit(TOP_N_SHOW).show(truncate=False)

    print("Done.")

    spark.stop()


if __name__ == "__main__":
    main()
