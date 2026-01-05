"""
RQ2 - Best lag summary (post-processing)

Reads lag effects produced by rq2_shock_propagation.py and produces:
- best lag per symbol for volatility response (delta_vol)
- best lag per symbol for illiquidity response (delta_illiq)
- typical lag distribution stats (mean/median/p25/p75)
- histograms (counts) of best lags

Also supports filtering out leveraged tokens (UP/DOWN/BULL/BEAR).
"""

from __future__ import annotations

import os
import sys
from typing import List

from pyspark.sql import functions as F
from pyspark.sql.window import Window


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # Preferred import when project root is on PYTHONPATH
    from spark.config import ensure_spark, derived_path  # type: ignore
except Exception:
    # Fallback: if your config file is used differently
    # (This keeps the script runnable even if imports are messy)
    from pyspark.sql import SparkSession

    def ensure_spark(app_name: str):
        return SparkSession.builder.appName(app_name).getOrCreate()

    def derived_path() -> str:
        # Default HDFS path guess (works on your cluster user structure)
        user = os.environ.get("USER", "s3702111")
        return f"hdfs:///user/{user}/binance/derived"


#Config
IN_SUBDIR = "rq2_results/lag_effects"
OUT_BASE_SUBDIR = "rq2_results"

# minimum shock observations to keep a symbol reliable
MIN_SHOCK_OBS = int(os.environ.get("MIN_SHOCK_OBS", "200"))

# filter leveraged tokens
EXCLUDE_LEVERAGED = os.environ.get("EXCLUDE_LEVERAGED", "1") == "1"
# Binance leveraged tokens commonly contain these suffixes
LEVERAGED_PATTERNS = ["UP", "DOWN", "BULL", "BEAR"]


#Helpers
def is_leveraged_symbol_col(col: F.Column) -> F.Column:
    """
    Returns a Spark boolean expression for leveraged token symbols.
    Match token suffixes like:
      BTCUP-USDT, ETHDOWN-USDT, ADA-BULL/BEAR etc.
    """
    # match "-(UP|DOWN|BULL|BEAR)" anywhere after a hyphen
    # e.g. "SUSHIUP-USDT" doesn't contain "-UP", but "SUSHIUP-USDT" is still leveraged style.
    # So also match tokens ending with UP/DOWN/BULL/BEAR before the quote asset separator.
    # Practical robust approach: regex for (UP|DOWN|BULL|BEAR) before '-' in quote part.
    regex = r"(UP|DOWN|BULL|BEAR)(-|$)"
    return col.rlike(regex)


def best_by_metric(df, metric_col: str):
    """
    Pick the best lag per symbol for a given metric.

    For delta_vol:
      higher is "stronger volatility increase after shock" -> maximize metric

    For delta_illiq:
      interpretation depends on your delta definition, but your pipeline uses delta_illiq
      as (illiq_on_shock - illiq_on_noshock). Higher => more illiquid after shock.
      So also maximize delta_illiq.

    Tie-breakers:
      1) metric desc
      2) lag asc (earlier propagation preferred)
    """
    w = Window.partitionBy("symbol").orderBy(F.col(metric_col).desc(), F.col("lag").asc())
    best = (
        df.withColumn("rn", F.row_number().over(w))
          .filter(F.col("rn") == 1)
          .drop("rn")
    )

    # Rename the chosen metric column to avoid ambiguity downstream
    best_metric_name = f"best_{metric_col}"
    best = best.withColumnRenamed(metric_col, best_metric_name)

    # keep the other metric too, but do NOT rename it
    return best.orderBy(F.col(best_metric_name).desc(), F.col("lag").asc())


def lag_histogram(best_df, best_metric_name: str, metric_label: str):
    """
    Count how often each lag occurs as the best lag.
    """
    return (
        best_df.groupBy("lag")
               .agg(F.count("*").alias("count_symbols"))
               .withColumn("metric", F.lit(metric_label))
               .orderBy(F.col("count_symbols").desc(), F.col("lag").asc())
    )


def lag_typical_stats(best_df, metric_label: str):
    """
    Typical lag stats across symbols.
    """
    return (
        best_df.agg(
            F.count("*").alias("n_symbols"),
            F.avg("lag").alias("mean_lag"),
            F.expr("percentile_approx(lag, 0.50)").alias("median_lag"),
            F.expr("percentile_approx(lag, 0.25)").alias("p25_lag"),
            F.expr("percentile_approx(lag, 0.75)").alias("p75_lag"),
        )
        .withColumn("metric", F.lit(metric_label))
        .select("metric", "n_symbols", "mean_lag", "median_lag", "p25_lag", "p75_lag")
    )


#Main
def main():
    spark = ensure_spark("rq2_best_lag_summary")

    base = derived_path()  # base derived folder
    in_path = f"{base}/{IN_SUBDIR}"
    out_base = f"{base}/{OUT_BASE_SUBDIR}"

    print("Reading RQ2 lag effects from:", in_path)
    df = spark.read.parquet(in_path)

    # Normalize lag column name (your writer uses lag_min)
    if "lag" not in df.columns and "lag_min" in df.columns:
        df = df.withColumnRenamed("lag_min", "lag")

    # Validate required columns
    required = [
        "symbol", "lag",
        "delta_vol", "delta_illiq",
        "vol_on_shock", "vol_on_noshock",
        "illiq_on_shock", "illiq_on_noshock",
        "n_shock_obs", "n_noshock_obs",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in lag_effects: {missing}. Found: {df.columns}")

    # Filter reliability
    print(f"Keeping rows with n_shock_obs >= {MIN_SHOCK_OBS}")
    df = df.filter(F.col("n_shock_obs") >= F.lit(MIN_SHOCK_OBS))

    # Exclude leveraged tokens
    if EXCLUDE_LEVERAGED:
        print("Excluding leveraged token symbols containing UP/DOWN/BULL/BEAR patterns")
        df = df.filter(~is_leveraged_symbol_col(F.col("symbol")))

    # Compute best lag per symbol for each metric
    best_vol = best_by_metric(df, "delta_vol")          # has column best_delta_vol
    best_illiq = best_by_metric(df, "delta_illiq")      # has column best_delta_illiq

    # Typical stats
    stats_vol = lag_typical_stats(best_vol, "delta_vol")
    stats_illiq = lag_typical_stats(best_illiq, "delta_illiq")
    stats_all = stats_vol.unionByName(stats_illiq)

    # Histograms of best lags
    hist_vol = lag_histogram(best_vol, "best_delta_vol", "delta_vol")
    hist_illiq = lag_histogram(best_illiq, "best_delta_illiq", "delta_illiq")
    hist_all = hist_vol.unionByName(hist_illiq)

    # Add a "best_metric" unified view: pick whichever metric is bigger in absolute sense
    # (This is just for a single top-10 table — optional but nice for report.)
    best_join = (
        best_vol.select(
            "symbol", "lag",
            F.col("best_delta_vol").alias("best_delta_vol"),
            "delta_illiq",
            "n_shock_obs",
            "vol_on_shock", "vol_on_noshock",
            "illiq_on_shock", "illiq_on_noshock",
            "n_noshock_obs",
        )
    )

    best_join = (
        best_join.withColumn("best_metric", F.col("best_delta_vol"))
                 .withColumn("best_metric_name", F.lit("delta_vol"))
                 .orderBy(F.col("best_metric").desc(), F.col("lag").asc())
    )

    #outputs
    out1 = f"{out_base}/best_lag_by_vol"
    out2 = f"{out_base}/best_lag_by_illiq"
    out3 = f"{out_base}/lag_typical_stats"
    out4 = f"{out_base}/best_lag_hist_vol"
    out5 = f"{out_base}/best_lag_hist_illiq"

    print("Writing:", out1)
    best_vol.write.mode("overwrite").parquet(out1)

    print("Writing:", out2)
    best_illiq.write.mode("overwrite").parquet(out2)

    print("Writing:", out3)
    stats_all.coalesce(1).write.mode("overwrite").parquet(out3)

    print("Writing:", out4)
    hist_vol.coalesce(1).write.mode("overwrite").parquet(out4)

    print("Writing:", out5)
    hist_illiq.coalesce(1).write.mode("overwrite").parquet(out5)

    #print top 10s and stats
    print("\nTop 10 symbols by strongest propagation (volatility metric):")
    best_join.select(
        "symbol", "lag",
        "best_delta_vol",
        "delta_illiq",
        "n_shock_obs",
        "vol_on_shock", "vol_on_noshock",
        "illiq_on_shock", "illiq_on_noshock",
        "n_noshock_obs",
        "best_metric", "best_metric_name",
    ).show(10, truncate=False)

    print("\nTypical lag stats across symbols:")
    stats_all.show(truncate=False)

    print("Done.")
    spark.stop()


if __name__ == "__main__":
    main()
