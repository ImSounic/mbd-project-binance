#!/usr/bin/env python3
"""
RQ3: Hyper-sensitivity to Bitcoin price movements + regime comparison (Bull vs Bear)

We estimate, for each receiver symbol i:

    r_i,t = alpha_i + beta_i * r_btc,t + eps_i,t

Compute betas:
- overall (all minutes)
- bull regime only
- bear regime only

Regime definition (Option 1):
- Compute BTC 30-day return from DAILY BTC close
- bull if 30d_return >= +BULL_TH
- bear if 30d_return <= -BEAR_TH
- neutral otherwise (optional; we keep for completeness)

Outputs written to:
  derived/rq3_results/
    - betas_overall
    - betas_by_regime
    - hypersensitive_lists
    - beta_change_summary

Notes:
- Excludes leveraged tokens: UP/DOWN/BULL/BEAR (anywhere in symbol)
- Excludes BTC-quoted pairs (e.g., XXX-BTC) by default
- Keeps quote assets: USDT, USDC, USD, EUR by default
- Uses simple OLS closed-form beta = cov(r_i, r_btc) / var(r_btc)
  (computed via aggregated sums to stay Spark-safe)
"""

import os
from typing import List

from pyspark.sql import functions as F
from pyspark.sql import Window

# Make imports work both locally and on cluster when you do PYTHONPATH=$PWD
from spark.config import ensure_spark, derived_path


# ---------------------- Config via env vars ----------------------
# Regime thresholds on 30-day BTC return
BULL_TH = float(os.environ.get("RQ3_BULL_TH", "0.10"))   # +10%
BEAR_TH = float(os.environ.get("RQ3_BEAR_TH", "0.10"))   # -10% (absolute); bear if <= -BEAR_TH

# Beta hyper-sensitivity cutoff
HYPER_Q = float(os.environ.get("RQ3_HYPER_Q", "0.95"))   # top 5%

# Data filters
KEEP_QUOTES = os.environ.get("RQ3_KEEP_QUOTES", "USDT,USDC,USD,EUR").split(",")
EXCLUDE_BTC_QUOTED = os.environ.get("RQ3_EXCLUDE_BTC_QUOTED", "1") == "1"
EXCLUDE_LEVERAGED = os.environ.get("RQ3_EXCLUDE_LEVERAGED", "1") == "1"

# Minimum rows for stable beta estimation
MIN_ROWS = int(os.environ.get("RQ3_MIN_ROWS", "50000"))  # per symbol per regime (tune if needed)

# BTC driver symbol
BTC_SYMBOL = os.environ.get("RQ3_BTC_SYMBOL", "BTC-USDT")

OUT_BASE = os.environ.get("RQ3_OUT", derived_path("rq3_results"))


# ---------------------- Helpers ----------------------
def is_leveraged_symbol(col_sym):
    # Matches typical Binance leveraged tokens and similar
    # E.g., SUSHIUP-USDT, ETHDOWN-USDT, etc.
    return (
        F.upper(col_sym).contains("UP")
        | F.upper(col_sym).contains("DOWN")
        | F.upper(col_sym).contains("BULL")
        | F.upper(col_sym).contains("BEAR")
    )


def keep_quote_asset(df, allowed: List[str]):
    return df.filter(F.col("quote_asset").isin([x.strip() for x in allowed if x.strip()]))


def compute_beta(df, group_cols: List[str], y_col: str, x_col: str, min_rows: int):
    """
    Compute OLS beta = cov(x,y)/var(x) using aggregates:
      beta = (E[xy] - E[x]E[y]) / (E[x^2] - (E[x])^2)
    Also returns:
      n, mean_x, mean_y, var_x, cov_xy, beta
    """
    agg = (
        df.groupBy(*group_cols)
        .agg(
            F.count(F.lit(1)).alias("n"),
            F.avg(F.col(x_col)).alias("mean_x"),
            F.avg(F.col(y_col)).alias("mean_y"),
            F.avg(F.col(x_col) * F.col(x_col)).alias("mean_x2"),
            F.avg(F.col(x_col) * F.col(y_col)).alias("mean_xy"),
        )
        .filter(F.col("n") >= F.lit(min_rows))
        .withColumn("var_x", F.col("mean_x2") - F.col("mean_x") * F.col("mean_x"))
        .withColumn("cov_xy", F.col("mean_xy") - F.col("mean_x") * F.col("mean_y"))
        .withColumn(
            "beta",
            F.when(F.abs(F.col("var_x")) > F.lit(1e-18), F.col("cov_xy") / F.col("var_x")).otherwise(F.lit(None)),
        )
    )
    return agg


# ---------------------- Main ----------------------
def main():
    spark = ensure_spark("rq3_btc_hypersensitivity")

    in_path = derived_path("features_1m")
    print("Reading features_1m from:", in_path)
    df = spark.read.parquet(in_path).select(
        "open_time", "symbol", "base_asset", "quote_asset", "close", "log_return"
    )

    # Basic hygiene
    df = df.filter(F.col("open_time").isNotNull()).filter(F.col("symbol").isNotNull())
    df = df.filter(F.col("log_return").isNotNull())

    # Optional filtering: keep only fiat/stable quoted markets
    df = keep_quote_asset(df, KEEP_QUOTES)

    # Exclude BTC-quoted pairs (XXX-BTC)
    if EXCLUDE_BTC_QUOTED:
        df = df.filter(~(F.col("quote_asset") == F.lit("BTC")))

    # Exclude leveraged tokens
    if EXCLUDE_LEVERAGED:
        df = df.filter(~is_leveraged_symbol(F.col("symbol")))

    # Split BTC and receivers
    btc = df.filter(F.col("symbol") == F.lit(BTC_SYMBOL)).select(
        F.col("open_time"),
        F.col("close").alias("btc_close"),
        F.col("log_return").alias("btc_ret"),
    )

    # If BTC is missing for some minutes, the join will drop those minutes (fine)
    receivers = df.filter(F.col("symbol") != F.lit(BTC_SYMBOL)).select(
        "open_time", "symbol", "base_asset", "quote_asset", F.col("log_return").alias("ret")
    )

    # -------------------- Regime definition (Option 1): BTC 30-day return from DAILY close --------------------
    # 1) Create daily BTC close (last close of each day)
    btc_daily = (
        btc.withColumn("date", F.to_date("open_time"))
        .withColumn("ts", F.col("open_time").cast("timestamp"))
    )

    w_last = Window.partitionBy("date").orderBy(F.col("ts").desc())
    btc_daily_close = (
        btc_daily.withColumn("rn", F.row_number().over(w_last))
        .filter(F.col("rn") == 1)
        .select("date", F.col("btc_close").alias("btc_daily_close"))
        .orderBy("date")
    )

    # 2) 30-day lag close (by 30 calendar days using date_sub)
    # We do a self-join on date_sub(date, 30)
    btc_daily_lag = btc_daily_close.select(
        F.col("date").alias("date_lag"),
        F.col("btc_daily_close").alias("btc_close_lag30"),
    )

    btc_regime = (
        btc_daily_close.join(
            btc_daily_lag,
            btc_daily_close["date"] == F.date_add(btc_daily_lag["date_lag"], 30),
            how="left",
        )
        .withColumn(
            "btc_ret_30d",
            F.when(
                F.col("btc_close_lag30").isNotNull() & (F.col("btc_close_lag30") > 0),
                F.col("btc_daily_close") / F.col("btc_close_lag30") - F.lit(1.0),
            ).otherwise(F.lit(None)),
        )
        .withColumn(
            "regime",
            F.when(F.col("btc_ret_30d") >= F.lit(BULL_TH), F.lit("BULL"))
             .when(F.col("btc_ret_30d") <= F.lit(-BEAR_TH), F.lit("BEAR"))
             .otherwise(F.lit("NEUTRAL")),
        )
        .select("date", "btc_ret_30d", "regime")
    )

    # Attach regime to minute-level by date
    receivers = receivers.withColumn("date", F.to_date("open_time"))
    btc = btc.withColumn("date", F.to_date("open_time"))

    receivers = receivers.join(btc_regime, on="date", how="left")
    receivers = receivers.filter(F.col("regime").isNotNull())  # drop early days lacking 30d history

    # Join BTC return onto each receiver minute
    joined = receivers.join(btc.select("open_time", "btc_ret"), on="open_time", how="inner")

    # -------------------- Overall beta per symbol --------------------
    overall_beta = compute_beta(
        joined,
        group_cols=["symbol", "base_asset", "quote_asset"],
        y_col="ret",
        x_col="btc_ret",
        min_rows=MIN_ROWS,
    ).select(
        "symbol", "base_asset", "quote_asset", "n", "beta", "var_x", "cov_xy", "mean_x", "mean_y"
    )

    # -------------------- Regime betas per symbol --------------------
    regime_beta = compute_beta(
        joined,
        group_cols=["symbol", "base_asset", "quote_asset", "regime"],
        y_col="ret",
        x_col="btc_ret",
        min_rows=MIN_ROWS // 2,  # allow fewer rows per regime; adjust if too strict
    ).select(
        "symbol", "base_asset", "quote_asset", "regime", "n", "beta", "var_x", "cov_xy", "mean_x", "mean_y"
    )

    # Pivot regime betas for easy comparison
    pivot = (
        regime_beta.groupBy("symbol", "base_asset", "quote_asset")
        .pivot("regime", ["BULL", "BEAR", "NEUTRAL"])
        .agg(F.first("beta"))
        .withColumnRenamed("BULL", "beta_bull")
        .withColumnRenamed("BEAR", "beta_bear")
        .withColumnRenamed("NEUTRAL", "beta_neutral")
        .withColumn("delta_beta_bear_minus_bull", F.col("beta_bear") - F.col("beta_bull"))
    )

    # -------------------- Hyper-sensitivity thresholds --------------------
    # Use overall beta distribution for hyper-sensitive identification
    beta_vals = overall_beta.filter(F.col("beta").isNotNull())
    q = beta_vals.approxQuantile("beta", [HYPER_Q], 0.01)[0]
    print(f"Hyper-sensitivity cutoff: top {int(HYPER_Q*100)}% beta >= {q}")

    hyper_overall = (
        overall_beta.filter(F.col("beta").isNotNull())
        .withColumn("is_hyper", F.col("beta") >= F.lit(q))
        .filter(F.col("is_hyper") == 1)
        .orderBy(F.col("beta").desc())
    )

    # Also hyper within bear and bull separately (optional, useful)
    bull_q = regime_beta.filter((F.col("regime") == "BULL") & F.col("beta").isNotNull()) \
                        .approxQuantile("beta", [HYPER_Q], 0.01)[0]
    bear_q = regime_beta.filter((F.col("regime") == "BEAR") & F.col("beta").isNotNull()) \
                        .approxQuantile("beta", [HYPER_Q], 0.01)[0]
    print(f"BULL hyper cutoff (top {int(HYPER_Q*100)}%): {bull_q}")
    print(f"BEAR hyper cutoff (top {int(HYPER_Q*100)}%): {bear_q}")

    hyper_bull = (
        regime_beta.filter((F.col("regime") == "BULL") & F.col("beta").isNotNull() & (F.col("beta") >= F.lit(bull_q)))
        .orderBy(F.col("beta").desc())
    )
    hyper_bear = (
        regime_beta.filter((F.col("regime") == "BEAR") & F.col("beta").isNotNull() & (F.col("beta") >= F.lit(bear_q)))
        .orderBy(F.col("beta").desc())
    )

    # -------------------- Summary stats --------------------
    # Compare distributions of betas by regime
    beta_regime_summary = (
        regime_beta.filter(F.col("beta").isNotNull())
        .groupBy("regime")
        .agg(
            F.count("*").alias("n_symbols"),
            F.avg("beta").alias("avg_beta"),
            F.expr("percentile_approx(beta, 0.5)").alias("median_beta"),
            F.expr("percentile_approx(beta, 0.95)").alias("p95_beta"),
            F.expr("percentile_approx(beta, 0.05)").alias("p05_beta"),
        )
        .orderBy("regime")
    )

    # Largest regime change
    top_delta = (
        pivot.filter(F.col("beta_bull").isNotNull() & F.col("beta_bear").isNotNull())
        .orderBy(F.col("delta_beta_bear_minus_bull").desc())
    )

    # -------------------- Write outputs --------------------
    print("Writing:", f"{OUT_BASE}/betas_overall")
    overall_beta.write.mode("overwrite").parquet(f"{OUT_BASE}/betas_overall")

    print("Writing:", f"{OUT_BASE}/betas_by_regime")
    regime_beta.write.mode("overwrite").parquet(f"{OUT_BASE}/betas_by_regime")

    print("Writing:", f"{OUT_BASE}/beta_pivot")
    pivot.write.mode("overwrite").parquet(f"{OUT_BASE}/beta_pivot")

    print("Writing:", f"{OUT_BASE}/hyper_overall")
    hyper_overall.write.mode("overwrite").parquet(f"{OUT_BASE}/hyper_overall")

    print("Writing:", f"{OUT_BASE}/hyper_bull")
    hyper_bull.write.mode("overwrite").parquet(f"{OUT_BASE}/hyper_bull")

    print("Writing:", f"{OUT_BASE}/hyper_bear")
    hyper_bear.write.mode("overwrite").parquet(f"{OUT_BASE}/hyper_bear")

    print("Writing:", f"{OUT_BASE}/beta_regime_summary")
    beta_regime_summary.write.mode("overwrite").parquet(f"{OUT_BASE}/beta_regime_summary")

    print("Writing:", f"{OUT_BASE}/top_delta_beta")
    top_delta.write.mode("overwrite").parquet(f"{OUT_BASE}/top_delta_beta")

    # -------------------- Terminal previews (safe) --------------------
    print("\nRQ3: Regime beta summary:")
    beta_regime_summary.show(truncate=False)

    print("\nRQ3: Top 15 hyper-sensitive overall (highest beta):")
    hyper_overall.select("symbol", "quote_asset", "n", "beta").show(15, truncate=False)

    print("\nRQ3: Top 15 biggest BEAR - BULL beta increases:")
    top_delta.select("symbol", "quote_asset", "beta_bull", "beta_bear", "delta_beta_bear_minus_bull").show(15, truncate=False)

    print("Done.")


if __name__ == "__main__":
    main()
