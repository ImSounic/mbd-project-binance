# spark/rq2_best_lag_summary.py
"""
RQ2 Summary: Best lag + typical lag distribution

Input (produced by rq2_shock_propagation.py):
  - {derived_base}/rq2_results/lag_effects   (parquet)

Output:
  - {derived_base}/rq2_results/best_lag_by_vol     (parquet)
  - {derived_base}/rq2_results/best_lag_by_illiq   (parquet)
  - {derived_base}/rq2_results/lag_typical_stats   (parquet)
  - {derived_base}/rq2_results/best_lag_hist_vol   (parquet)
  - {derived_base}/rq2_results/best_lag_hist_illiq (parquet)

Also writes CSV "single file" versions (coalesced to 1 partition) for easy download:
  - .../csv_best_lag_by_vol
  - .../csv_best_lag_by_illiq
  - .../csv_lag_typical_stats
  - .../csv_best_lag_hist_vol
  - .../csv_best_lag_hist_illiq

Why this file exists:
  - We avoid big collect/show on the driver (can OOM).
  - We standardize the outputs so your report can cite tables directly.
"""

import os
from pyspark.sql import SparkSession, functions as F, Window


# -------------------- Config helpers --------------------
def _get_derived_base() -> str:
    """
    Uses your existing config.py if available.
    - Cluster: hdfs:///user/<id>/binance/derived
    - Local:   data/derived
    """
    try:
        # Prefer local import style (spark-submit from repo root)
        import config  # type: ignore

        # If your config has DATA_DERIVED and IS_LOCAL (as in your earlier snippet)
        if getattr(config, "IS_LOCAL", False):
            return "data/derived"
        return getattr(config, "DATA_DERIVED", "data/derived")
    except Exception:
        # Fallback: assume local layout
        return "data/derived"


def ensure_spark(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )
    # Keep Spark logs quieter in console (still visible in YARN logs if needed)
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "WARN"))
    return spark


# -------------------- Core logic --------------------
def best_by_metric(df, metric_col: str, out_name: str):
    """
    For each symbol, pick the lag that maximizes `metric_col`.
    Tie-breaker: prefer smaller lag (faster propagation).
    """
    w = Window.partitionBy("symbol").orderBy(F.col(metric_col).desc(), F.col("lag").asc())

    best = (
        df
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .select(
            "symbol",
            "lag",
            F.col(metric_col).alias(f"best_{metric_col}"),
            # Keep the paired metric too for context in the report
            "delta_vol",
            "delta_illiq",
            "n_shock_obs",
        )
        .orderBy(F.col(f"best_{metric_col}").desc())
    )
    return best


def typical_lag_stats(best_vol, best_illiq):
    """
    Provide typical lag summary stats across symbols:
      - count symbols
      - median lag
      - p25/p75 lag
      - mean lag
    """
    def stats_for(best_df, label: str):
        # percentile_approx is scalable + safe
        return (
            best_df
            .agg(
                F.lit(label).alias("metric"),
                F.count("*").alias("n_symbols"),
                F.avg("lag").alias("mean_lag"),
                F.expr("percentile_approx(lag, 0.50)").alias("median_lag"),
                F.expr("percentile_approx(lag, 0.25)").alias("p25_lag"),
                F.expr("percentile_approx(lag, 0.75)").alias("p75_lag"),
            )
        )

    return stats_for(best_vol, "delta_vol").unionByName(stats_for(best_illiq, "delta_illiq"))


def best_lag_hist(best_df, label: str):
    """
    Count how many symbols have each lag as their best lag.
    """
    return (
        best_df
        .groupBy("lag")
        .agg(F.count("*").alias("n_symbols"))
        .withColumn("metric", F.lit(label))
        .orderBy("lag")
        .select("metric", "lag", "n_symbols")
    )


def write_parquet_and_csv(df, out_base: str):
    """
    Writes:
      - parquet to out_base
      - coalesced CSV to out_base + "_csv"
    """
    df.write.mode("overwrite").parquet(out_base)
    # CSV for easy retrieval; single file for convenience
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_base + "_csv")


def main():
    derived_base = _get_derived_base()
    in_path = f"{derived_base}/rq2_results/lag_effects"
    out_base = f"{derived_base}/rq2_results"

    spark = ensure_spark("rq2_best_lag_summary")

    print("Reading RQ2 lag effects from:", in_path)
    df = spark.read.parquet(in_path)

    # Minimal schema expectation:
    # lag (int), symbol (str), delta_vol (double), delta_illiq (double), n_shock_obs (long/int)
    needed = {"lag", "symbol", "delta_vol", "delta_illiq", "n_shock_obs"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in lag_effects: {missing}. Found: {df.columns}")

    # Optional hygiene: drop extremely tiny samples (unstable estimates)
    # You can tune this threshold; keep it mild so we don’t throw away too much.
    MIN_SHOCK_OBS = int(os.environ.get("MIN_SHOCK_OBS", "200"))
    df = df.filter(F.col("n_shock_obs") >= F.lit(MIN_SHOCK_OBS))

    print(f"Keeping rows with n_shock_obs >= {MIN_SHOCK_OBS}")

    # Best lag per symbol for each metric
    best_vol = best_by_metric(df, "delta_vol", "best_lag_by_vol")
    best_illiq = best_by_metric(df, "delta_illiq", "best_lag_by_illiq")

    # Typical lag statistics + histograms
    stats = typical_lag_stats(best_vol, best_illiq)
    hist_vol = best_lag_hist(best_vol, "delta_vol")
    hist_illiq = best_lag_hist(best_illiq, "delta_illiq")

    # Write outputs
    print("Writing:", f"{out_base}/best_lag_by_vol")
    write_parquet_and_csv(best_vol, f"{out_base}/best_lag_by_vol")

    print("Writing:", f"{out_base}/best_lag_by_illiq")
    write_parquet_and_csv(best_illiq, f"{out_base}/best_lag_by_illiq")

    print("Writing:", f"{out_base}/lag_typical_stats")
    write_parquet_and_csv(stats, f"{out_base}/lag_typical_stats")

    print("Writing:", f"{out_base}/best_lag_hist_vol")
    write_parquet_and_csv(hist_vol, f"{out_base}/best_lag_hist_vol")

    print("Writing:", f"{out_base}/best_lag_hist_illiq")
    write_parquet_and_csv(hist_illiq, f"{out_base}/best_lag_hist_illiq")

    # Safe previews (tiny, no huge show)
    print("\nTop 10 symbols by strongest volatility propagation (best delta_vol):")
    best_vol.limit(10).show(truncate=False)

    print("\nTop 10 symbols by strongest liquidity change (best delta_illiq):")
    best_illiq.limit(10).show(truncate=False)

    print("\nTypical lag stats across symbols:")
    stats.show(truncate=False)

    print("\nBest-lag histogram (volatility):")
    hist_vol.show(200, truncate=False)

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
