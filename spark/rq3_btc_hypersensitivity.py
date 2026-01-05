#!/usr/bin/env python3
"""
RQ3: Hyper-sensitivity to Bitcoin price movements + regime comparison (Bull vs Bear)

Option 1 (implemented now):
- Regimes defined by BTC 30-day return using DAILY BTC close:
    bull if 30d_return >= +BULL_TH
    bear if 30d_return <= -BEAR_TH
    neutral otherwise

Model:
    r_i,t = alpha_i + beta_i * r_btc,t + eps_i,t
beta computed via closed-form OLS:
    beta = cov(x,y)/var(x)

Outputs written to derived/rq3_results/* (HDFS on cluster, local path locally).

Filters:
- Excludes leveraged tokens containing UP/DOWN/BULL/BEAR in symbol (default on)
- Keeps quote assets USDT/USDC/USD/EUR (default)
- Optionally excludes XXX-BTC by excluding quote_asset == BTC (default on)
"""

import os
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

# ---------------------- Paths + Spark builder (self-contained) ----------------------
def is_hdfs_path(p: str) -> bool:
    return p.startswith("hdfs://") or p.startswith("/user/") or p.startswith("viewfs://")

def default_derived_base() -> str:
    # Cluster convention (yours): /user/<user>/binance/derived
    user = os.environ.get("USER", "user")
    return f"hdfs:///user/{user}/binance/derived"

def derived_path(name: str = "") -> str:
    """
    If DATA_DERIVED_BASE is set, use it. Otherwise:
    - If on cluster (heuristic: HADOOP_CONF_DIR or YARN_CONF_DIR set), use HDFS derived base.
    - Else use local repo/data/derived
    """
    override = os.environ.get("DATA_DERIVED_BASE", "").strip()
    if override:
        base = override
    else:
        on_cluster = bool(os.environ.get("HADOOP_CONF_DIR") or os.environ.get("YARN_CONF_DIR"))
        base = default_derived_base() if on_cluster else os.path.join(os.getcwd(), "data", "derived")

    if not name:
        return base
    if base.endswith("/"):
        return base + name
    return base + "/" + name

def ensure_spark(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )
    # keep logs manageable by default
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "WARN"))
    return spark

#Config via env vars
BULL_TH = float(os.environ.get("RQ3_BULL_TH", "0.10"))  # +10%
BEAR_TH = float(os.environ.get("RQ3_BEAR_TH", "0.10"))  # -10% (abs)

HYPER_Q = float(os.environ.get("RQ3_HYPER_Q", "0.95"))  # top 5%
KEEP_QUOTES = os.environ.get("RQ3_KEEP_QUOTES", "USDT,USDC,USD,EUR").split(",")

EXCLUDE_BTC_QUOTED = os.environ.get("RQ3_EXCLUDE_BTC_QUOTED", "1") == "1"
EXCLUDE_LEVERAGED = os.environ.get("RQ3_EXCLUDE_LEVERAGED", "1") == "1"

MIN_ROWS = int(os.environ.get("RQ3_MIN_ROWS", "50000"))
BTC_SYMBOL = os.environ.get("RQ3_BTC_SYMBOL", "BTC-USDT")

OUT_BASE = os.environ.get("RQ3_OUT", derived_path("rq3_results"))

#Helpers
def is_leveraged_symbol(col_sym):
    u = F.upper(col_sym)
    return u.contains("UP") | u.contains("DOWN") | u.contains("BULL") | u.contains("BEAR")

def keep_quote_asset(df, allowed: List[str]):
    allowed = [x.strip() for x in allowed if x.strip()]
    return df.filter(F.col("quote_asset").isin(allowed))

def compute_beta(df, group_cols: List[str], y_col: str, x_col: str, min_rows: int):
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

#Main
def main():
    spark = ensure_spark("rq3_btc_hypersensitivity")

    in_path = derived_path("features_1m")
    print("Reading features_1m from:", in_path)

    df = spark.read.parquet(in_path).select(
        "open_time", "symbol", "base_asset", "quote_asset", "close", "log_return"
    )

    df = df.filter(F.col("open_time").isNotNull()) \
           .filter(F.col("symbol").isNotNull()) \
           .filter(F.col("log_return").isNotNull())

    # Keep only fiat/stable quoted markets
    df = keep_quote_asset(df, KEEP_QUOTES)

    # Exclude BTC quoted pairs
    if EXCLUDE_BTC_QUOTED:
        df = df.filter(~(F.col("quote_asset") == F.lit("BTC")))

    # Exclude leveraged tokens
    if EXCLUDE_LEVERAGED:
        df = df.filter(~is_leveraged_symbol(F.col("symbol")))

    btc = df.filter(F.col("symbol") == F.lit(BTC_SYMBOL)).select(
        "open_time",
        F.col("close").alias("btc_close"),
        F.col("log_return").alias("btc_ret"),
    )

    receivers = df.filter(F.col("symbol") != F.lit(BTC_SYMBOL)).select(
        "open_time", "symbol", "base_asset", "quote_asset", F.col("log_return").alias("ret")
    )

    #Regime definition: BTC 30d return from DAILY close
    btc_daily = btc.withColumn("date", F.to_date("open_time")) \
                   .withColumn("ts", F.col("open_time").cast("timestamp"))

    w_last = Window.partitionBy("date").orderBy(F.col("ts").desc())
    btc_daily_close = (
        btc_daily.withColumn("rn", F.row_number().over(w_last))
        .filter(F.col("rn") == 1)
        .select("date", F.col("btc_close").alias("btc_daily_close"))
        .orderBy("date")
    )

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

    receivers = receivers.withColumn("date", F.to_date("open_time"))
    btc = btc.withColumn("date", F.to_date("open_time"))

    receivers = receivers.join(btc_regime, on="date", how="left") \
                         .filter(F.col("regime").isNotNull())

    joined = receivers.join(btc.select("open_time", "btc_ret"), on="open_time", how="inner")

    #Betas
    overall_beta = compute_beta(
        joined,
        group_cols=["symbol", "base_asset", "quote_asset"],
        y_col="ret",
        x_col="btc_ret",
        min_rows=MIN_ROWS,
    ).select("symbol", "base_asset", "quote_asset", "n", "beta", "var_x", "cov_xy", "mean_x", "mean_y")

    regime_beta = compute_beta(
        joined,
        group_cols=["symbol", "base_asset", "quote_asset", "regime"],
        y_col="ret",
        x_col="btc_ret",
        min_rows=max(1000, MIN_ROWS // 2),
    ).select("symbol", "base_asset", "quote_asset", "regime", "n", "beta", "var_x", "cov_xy", "mean_x", "mean_y")

    pivot = (
        regime_beta.groupBy("symbol", "base_asset", "quote_asset")
        .pivot("regime", ["BULL", "BEAR", "NEUTRAL"])
        .agg(F.first("beta"))
        .withColumnRenamed("BULL", "beta_bull")
        .withColumnRenamed("BEAR", "beta_bear")
        .withColumnRenamed("NEUTRAL", "beta_neutral")
        .withColumn("delta_beta_bear_minus_bull", F.col("beta_bear") - F.col("beta_bull"))
    )

    # Hyper threshold from overall betas
    beta_vals = overall_beta.filter(F.col("beta").isNotNull())
    q = beta_vals.approxQuantile("beta", [HYPER_Q], 0.01)[0]
    print(f"Hyper-sensitivity cutoff: top {int(HYPER_Q*100)}% beta >= {q}")

    hyper_overall = (
        overall_beta.filter(F.col("beta").isNotNull() & (F.col("beta") >= F.lit(q)))
        .orderBy(F.col("beta").desc())
    )

    # Regime summaries
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

    top_delta = (
        pivot.filter(F.col("beta_bull").isNotNull() & F.col("beta_bear").isNotNull())
        .orderBy(F.col("delta_beta_bear_minus_bull").desc())
    )

    #outputs
    print("Writing:", f"{OUT_BASE}/betas_overall")
    overall_beta.write.mode("overwrite").parquet(f"{OUT_BASE}/betas_overall")

    print("Writing:", f"{OUT_BASE}/betas_by_regime")
    regime_beta.write.mode("overwrite").parquet(f"{OUT_BASE}/betas_by_regime")

    print("Writing:", f"{OUT_BASE}/beta_pivot")
    pivot.write.mode("overwrite").parquet(f"{OUT_BASE}/beta_pivot")

    print("Writing:", f"{OUT_BASE}/hyper_overall")
    hyper_overall.write.mode("overwrite").parquet(f"{OUT_BASE}/hyper_overall")

    print("Writing:", f"{OUT_BASE}/beta_regime_summary")
    beta_regime_summary.write.mode("overwrite").parquet(f"{OUT_BASE}/beta_regime_summary")

    print("Writing:", f"{OUT_BASE}/top_delta_beta")
    top_delta.write.mode("overwrite").parquet(f"{OUT_BASE}/top_delta_beta")

    #previews
    print("\nRQ3: Regime beta summary:")
    beta_regime_summary.show(truncate=False)

    print("\nRQ3: Top 15 hyper-sensitive overall (highest beta):")
    hyper_overall.select("symbol", "quote_asset", "n", "beta").show(15, truncate=False)

    print("\nRQ3: Top 15 biggest BEAR - BULL beta increases:")
    top_delta.select("symbol", "quote_asset", "beta_bull", "beta_bear", "delta_beta_bear_minus_bull").show(15, truncate=False)

    print("Done.")


if __name__ == "__main__":
    main()
