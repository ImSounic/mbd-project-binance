"""
RQ2: Shock propagation from BTC/ETH trading intensity to altcoin liquidity/volatility.

Goal:
- Detect "shocks" (spikes) in trading intensity (number_of_trades) for BTC and ETH.
- Quantify how those shocks propagate to smaller altcoins' liquidity and volatility.
- Estimate the typical time lag (in minutes) where the influence is strongest.

Inputs:
- features_1m (derived) containing per-minute features per symbol.

Outputs:
- A compact result table per driver (BTC, ETH) with best lag + strength metrics,
  written to DATA_DERIVED/rq2_results (or local data/derived/rq2_results if local).

How it works (high level):
1) Load features_1m.
2) Choose driver assets: BTC and ETH (we use symbols with quote_asset in {USDT,BUSD,BTC,ETH} etc).
   In practice, you will likely use BTC-USDT and ETH-USDT as the "market benchmark" pairs.
3) Build driver shock series: minute-level trades z-score (or percentile) and flag top q as shocks.
4) For the "receiver" set: SMALL_CAP tier only (or all non-driver symbols).
   Measure response variables:
   - liquidity proxy: amihud_illiq (higher = less liquid)
   - volatility proxy: parkinson_var_1m (higher = more volatile)
5) For each lag in [0..MAX_LAG], align driver shocks at time t with receiver metrics at time t+lag.
   Aggregate response (mean) during shock minutes vs non-shock minutes.
6) Pick lag with max difference (shock - nonshock) as "typical lag".

Notes:
- This is designed to run on the cluster (YARN) and locally (Spark local[*]).
- Keeps joins narrow by selecting only required columns.
"""

import os
from typing import Tuple, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

# IMPORTANT: import config correctly when running as `spark-submit spark/...py`
# (repo root is on sys.path, but "spark" is a folder, not a package)
from config import (
    IS_LOCAL,
    derived_path,
    features_1m_path,
)

# -----------------------
# Tunables (env override)
# -----------------------
DRIVER_QUOTE = os.environ.get("RQ2_DRIVER_QUOTE", "USDT")  # pick benchmark quote (USDT is typical)
SHOCK_Q = float(os.environ.get("RQ2_SHOCK_Q", "0.99"))     # top 1% trades = shock
MAX_LAG = int(os.environ.get("RQ2_MAX_LAG", "120"))        # search up to 120 minutes
RECEIVER_TIER = os.environ.get("RQ2_RECEIVER_TIER", "SMALL_CAP")  # focus on small caps
OUTPUT_SUBDIR = os.environ.get("RQ2_OUT", "rq2_results")


def ensure_spark(app_name: str) -> SparkSession:
    b = SparkSession.builder.appName(app_name)
    if IS_LOCAL:
        b = b.master("local[*]")
    # Keep shuffle sane; can still be overridden by spark-submit --conf
    b = b.config("spark.sql.adaptive.enabled", "true")
    return b.getOrCreate()


def pick_driver_symbol(df: DataFrame, base_asset: str, quote_asset: str) -> str:
    """
    Pick a driver symbol like BTC-USDT from available rows.
    If missing, fall back to any symbol where base_asset matches.
    """
    cand = (
        df.filter((F.col("base_asset") == base_asset) & (F.col("quote_asset") == quote_asset))
          .select("symbol")
          .distinct()
          .limit(1)
          .collect()
    )
    if cand:
        return cand[0]["symbol"]

    cand2 = (
        df.filter(F.col("base_asset") == base_asset)
          .select("symbol")
          .distinct()
          .limit(1)
          .collect()
    )
    if not cand2:
        raise RuntimeError(f"No symbol found for base_asset={base_asset}")
    return cand2[0]["symbol"]


def compute_driver_shocks(df: DataFrame, driver_symbol: str, shock_q: float) -> Tuple[DataFrame, float]:
    """
    Build a driver shock dataframe with columns:
    - open_time
    - is_shock (1/0)
    - trades (number_of_trades)
    Threshold is computed as percentile(shock_q) of number_of_trades.
    """
    d = (
        df.filter(F.col("symbol") == driver_symbol)
          .select("open_time", F.col("number_of_trades").cast("double").alias("trades"))
          .filter(F.col("trades").isNotNull())
    )

    # percentile_approx is efficient on big data
    thr = d.select(F.expr(f"percentile_approx(trades, {shock_q})").alias("thr")).collect()[0]["thr"]

    d = d.withColumn("is_shock", (F.col("trades") >= F.lit(thr)).cast("int"))
    return d, float(thr)


def response_by_lag(
    df: DataFrame,
    driver_shocks: DataFrame,
    lags: List[int],
    receiver_tier: str,
) -> DataFrame:
    """
    For each lag, compute mean response during shock vs non-shock for:
    - amihud_illiq
    - parkinson_var_1m
    Returns one row per lag with deltas.
    """

    # Receivers: exclude driver rows; optionally focus on SMALL_CAP
    base = (
        df.filter(F.col("tier") == receiver_tier) if receiver_tier else df
    ).select(
        "open_time",
        "symbol",
        F.col("amihud_illiq").cast("double").alias("amihud_illiq"),
        F.col("parkinson_var_1m").cast("double").alias("parkinson_var_1m"),
    )

    results = []
    for lag in lags:
        # Align receiver at t+lag with driver shock at t
        shifted = base.withColumn("driver_time", F.expr(f"open_time - INTERVAL {lag} MINUTES"))
        joined = (
            shifted.join(driver_shocks.select(F.col("open_time").alias("driver_time"), "is_shock"),
                         on="driver_time",
                         how="inner")
                   .drop("driver_time")
        )

        agg = (
            joined.groupBy("is_shock")
                  .agg(
                      F.avg("amihud_illiq").alias("avg_amihud"),
                      F.avg("parkinson_var_1m").alias("avg_parkinson"),
                      F.count(F.lit(1)).alias("n_obs"),
                  )
        )

        # Pivot shock vs non-shock into a single row
        pivot = (
            agg.groupBy()
               .pivot("is_shock", [0, 1])
               .agg(
                   F.first("avg_amihud").alias("avg_amihud"),
                   F.first("avg_parkinson").alias("avg_parkinson"),
                   F.first("n_obs").alias("n_obs"),
               )
        )

        # Columns come out like: `0_avg_amihud`, `1_avg_amihud`, etc.
        row = (
            pivot.select(
                F.lit(lag).alias("lag_min"),
                F.col("1_avg_amihud").alias("shock_avg_amihud"),
                F.col("0_avg_amihud").alias("nonshock_avg_amihud"),
                (F.col("1_avg_amihud") - F.col("0_avg_amihud")).alias("delta_amihud"),
                F.col("1_avg_parkinson").alias("shock_avg_parkinson"),
                F.col("0_avg_parkinson").alias("nonshock_avg_parkinson"),
                (F.col("1_avg_parkinson") - F.col("0_avg_parkinson")).alias("delta_parkinson"),
                F.col("1_n_obs").alias("n_shock"),
                F.col("0_n_obs").alias("n_nonshock"),
            )
        )
        results.append(row)

    # Union all lag rows
    out = results[0]
    for r in results[1:]:
        out = out.unionByName(r)
    return out


def main():
    spark = ensure_spark("rq2_shock_propagation")

    in_path = features_1m_path()
    out_path = f"{derived_path()}/{OUTPUT_SUBDIR}"

    print("Reading features_1m from:", in_path)
    df = spark.read.parquet(in_path).select(
        "open_time",
        "symbol",
        "base_asset",
        "quote_asset",
        "tier",
        "number_of_trades",
        "amihud_illiq",
        "parkinson_var_1m",
    )

    # -------------------- STEP 1: Choose driver pairs (BTC/ETH) --------------------
    # Why: We need a single representative market-traded pair for BTC and ETH to define shocks.
    btc_symbol = pick_driver_symbol(df, "BTC", DRIVER_QUOTE)
    eth_symbol = pick_driver_symbol(df, "ETH", DRIVER_QUOTE)
    print("Using driver BTC symbol:", btc_symbol)
    print("Using driver ETH symbol:", eth_symbol)

    # -------------------- STEP 2: Compute shock flags for drivers --------------------
    # Why: A "shock" is a rare spike in trades (top 1% by default).
    btc_shocks, btc_thr = compute_driver_shocks(df, btc_symbol, SHOCK_Q)
    eth_shocks, eth_thr = compute_driver_shocks(df, eth_symbol, SHOCK_Q)
    print(f"BTC trades shock threshold (q={SHOCK_Q}): {btc_thr}")
    print(f"ETH trades shock threshold (q={SHOCK_Q}): {eth_thr}")

    # -------------------- STEP 3: Compute responses across lags --------------------
    # Why: We test many candidate lags to estimate typical delay of propagation.
    lags = list(range(0, MAX_LAG + 1))
    print(f"Computing lag response for receiver tier={RECEIVER_TIER}, lags=0..{MAX_LAG} minutes")

    btc_lag_tbl = response_by_lag(df, btc_shocks, lags, RECEIVER_TIER).withColumn("driver", F.lit("BTC"))
    eth_lag_tbl = response_by_lag(df, eth_shocks, lags, RECEIVER_TIER).withColumn("driver", F.lit("ETH"))
    all_lags = btc_lag_tbl.unionByName(eth_lag_tbl)

    # -------------------- STEP 4: Pick best lag per driver --------------------
    # Why: "Typical lag" = where the response difference is strongest.
    w = Window.partitionBy("driver").orderBy(F.desc(F.abs(F.col("delta_parkinson"))))
    best = (
        all_lags.withColumn("rk", F.row_number().over(w))
                .filter(F.col("rk") == 1)
                .drop("rk")
                .select(
                    "driver",
                    "lag_min",
                    "delta_amihud",
                    "delta_parkinson",
                    "shock_avg_amihud",
                    "nonshock_avg_amihud",
                    "shock_avg_parkinson",
                    "nonshock_avg_parkinson",
                    "n_shock",
                    "n_nonshock",
                )
    )

    print("RQ2 best-lag summary (by strongest volatility response):")
    best.show(truncate=False)

    # Save both detailed lag table and best summary
    print("Writing detailed lag table to:", f"{out_path}/lag_scan")
    all_lags.repartition(1).write.mode("overwrite").parquet(f"{out_path}/lag_scan")

    print("Writing best-lag summary to:", f"{out_path}/best_lag")
    best.repartition(1).write.mode("overwrite").parquet(f"{out_path}/best_lag")

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
