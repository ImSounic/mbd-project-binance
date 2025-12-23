"""
RQ2: Shock propagation from BTC/ETH trading intensity (number_of_trades) to altcoin liquidity/volatility.

This version DOES NOT require a `tier` column in features_1m.

Idea:
- Use BTC and ETH as drivers.
- Define "shock" minutes as top q percentile of number_of_trades for BTC/ETH.
- For all receiver assets (everything except BTC/ETH), measure how liquidity/volatility changes
  at time t+lag when BTC/ETH had a shock at time t.
- Scan lags 0..MAX_LAG, pick typical lag where the impact is strongest.

Inputs:
- features_1m parquet folder

Outputs (parquet):
- {derived}/rq2_results/lag_scan
- {derived}/rq2_results/best_lag
"""

import os
from typing import Tuple, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

from config import (
    IS_LOCAL,
    derived_path,
    features_1m_path,
)

# -----------------------
# Tunables (env override)
# -----------------------
DRIVER_QUOTE = os.environ.get("RQ2_DRIVER_QUOTE", "USDT")  # prefer BTC-USDT, ETH-USDT
SHOCK_Q = float(os.environ.get("RQ2_SHOCK_Q", "0.99"))     # top 1% trades = shock
MAX_LAG = int(os.environ.get("RQ2_MAX_LAG", "120"))        # search up to 120 minutes
OUTPUT_SUBDIR = os.environ.get("RQ2_OUT", "rq2_results")

# Optional: restrict receivers to smaller universe for speed (default = all non-BTC/ETH)
# Example: "USDT" keeps only *-USDT receiver pairs
RECEIVER_QUOTE = os.environ.get("RQ2_RECEIVER_QUOTE", "")


def ensure_spark(app_name: str) -> SparkSession:
    b = SparkSession.builder.appName(app_name)
    if IS_LOCAL:
        b = b.master("local[*]")
    b = b.config("spark.sql.adaptive.enabled", "true")
    return b.getOrCreate()


def pick_driver_symbol(df: DataFrame, base_asset: str, quote_asset: str) -> str:
    """
    Pick driver like BTC-USDT / ETH-USDT.
    Falls back to any symbol where base_asset matches if exact quote not found.
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
    Return (driver_shocks_df, threshold).
    driver_shocks_df columns: open_time, is_shock (0/1), trades
    """
    d = (
        df.filter(F.col("symbol") == driver_symbol)
          .select("open_time", F.col("number_of_trades").cast("double").alias("trades"))
          .filter(F.col("trades").isNotNull())
    )

    thr = d.select(F.expr(f"percentile_approx(trades, {shock_q})").alias("thr")).collect()[0]["thr"]
    d = d.withColumn("is_shock", (F.col("trades") >= F.lit(thr)).cast("int"))
    return d, float(thr)


def response_by_lag(
    df: DataFrame,
    driver_shocks: DataFrame,
    lags: List[int],
    receiver_filter: DataFrame,
) -> DataFrame:
    """
    For each lag, join driver shocks at time t to receivers at time t+lag, then compare:
      E[metric | shock] - E[metric | nonshock]

    receiver_filter is already the receiver dataframe with needed cols.
    """

    results = []
    for lag in lags:
        # receiver at (open_time) is affected by driver shock at (open_time - lag)
        shifted = receiver_filter.withColumn("driver_time", F.expr(f"open_time - INTERVAL {lag} MINUTES"))

        joined = (
            shifted.join(
                driver_shocks.select(F.col("open_time").alias("driver_time"), "is_shock"),
                on="driver_time",
                how="inner",
            ).drop("driver_time")
        )

        agg = (
            joined.groupBy("is_shock")
                  .agg(
                      F.avg("amihud_illiq").alias("avg_amihud"),
                      F.avg("parkinson_var_1m").alias("avg_parkinson"),
                      F.count(F.lit(1)).alias("n_obs"),
                  )
        )

        pivot = (
            agg.groupBy()
               .pivot("is_shock", [0, 1])
               .agg(
                   F.first("avg_amihud").alias("avg_amihud"),
                   F.first("avg_parkinson").alias("avg_parkinson"),
                   F.first("n_obs").alias("n_obs"),
               )
        )

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
        "number_of_trades",
        "amihud_illiq",
        "parkinson_var_1m",
    )

    # -------------------- STEP 1: choose driver pairs BTC/ETH --------------------
    btc_symbol = pick_driver_symbol(df, "BTC", DRIVER_QUOTE)
    eth_symbol = pick_driver_symbol(df, "ETH", DRIVER_QUOTE)
    print("Using driver BTC symbol:", btc_symbol)
    print("Using driver ETH symbol:", eth_symbol)

    # -------------------- STEP 2: define shocks from trade spikes --------------------
    btc_shocks, btc_thr = compute_driver_shocks(df, btc_symbol, SHOCK_Q)
    eth_shocks, eth_thr = compute_driver_shocks(df, eth_symbol, SHOCK_Q)
    print(f"BTC trades shock threshold (q={SHOCK_Q}): {btc_thr}")
    print(f"ETH trades shock threshold (q={SHOCK_Q}): {eth_thr}")

    # -------------------- STEP 3: define receiver universe (all non-BTC/ETH) --------------------
    # Why: we want propagation into "smaller altcoins", but we don't have market cap tiers in features_1m.
    # So we include all non-BTC/ETH base assets; we can later narrow using a cap-list if desired.
    receivers = (
        df.filter(~F.col("base_asset").isin(["BTC", "ETH"]))
          .select(
              "open_time",
              "symbol",
              "base_asset",
              "quote_asset",
              F.col("amihud_illiq").cast("double").alias("amihud_illiq"),
              F.col("parkinson_var_1m").cast("double").alias("parkinson_var_1m"),
          )
          .filter(F.col("amihud_illiq").isNotNull() & F.col("parkinson_var_1m").isNotNull())
    )

    if RECEIVER_QUOTE:
        receivers = receivers.filter(F.col("quote_asset") == RECEIVER_QUOTE)
        print("Receiver filter: quote_asset =", RECEIVER_QUOTE)

    # -------------------- STEP 4: scan lags --------------------
    lags = list(range(0, MAX_LAG + 1))
    print(f"Scanning lags 0..{MAX_LAG} minutes over receivers...")

    btc_lag_tbl = response_by_lag(df, btc_shocks, lags, receivers).withColumn("driver", F.lit("BTC"))
    eth_lag_tbl = response_by_lag(df, eth_shocks, lags, receivers).withColumn("driver", F.lit("ETH"))
    all_lags = btc_lag_tbl.unionByName(eth_lag_tbl)

    # -------------------- STEP 5: pick "typical lag" per driver --------------------
    # We define typical lag as the lag with the largest absolute delta in volatility proxy.
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

    print("Writing detailed lag table to:", f"{out_path}/lag_scan")
    all_lags.repartition(1).write.mode("overwrite").parquet(f"{out_path}/lag_scan")

    print("Writing best-lag summary to:", f"{out_path}/best_lag")
    best.repartition(1).write.mode("overwrite").parquet(f"{out_path}/best_lag")

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
