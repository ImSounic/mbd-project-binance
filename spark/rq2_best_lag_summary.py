# spark/rq2_best_lag_summary.py
"""
RQ2 Summary: Best lag + typical lag distribution

Input (produced by rq2_shock_propagation.py):
  - {derived_base}/rq2_results/lag_effects   (parquet)

Your current schema includes:
  symbol, lag_min, delta_vol, delta_illiq, n_shock_obs, ...

This script:
  1) normalizes lag_min -> lag
  2) picks best lag per symbol (max delta_vol / max delta_illiq)
  3) outputs parquet + CSV summaries (report-friendly)
"""

import os
from pyspark.sql import SparkSession, functions as F, Window


def _get_derived_base() -> str:
    """Use config.py if present; fallback to local derived folder."""
    try:
        import config  # type: ignore
        if getattr(config, "IS_LOCAL", False):
            return "data/derived"
        return getattr(config, "DATA_DERIVED", "data/derived")
    except Exception:
        return "data/derived"


def ensure_spark(app_name: str) -> SparkSession:
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "WARN"))
    return spark


def best_by_metric(df, metric_col: str):
    """
    For each symbol:
      - choose lag that maximizes metric_col (e.g., delta_vol)
      - tie-breaker: smaller lag (faster propagation)

    Returns one row per symbol.
    """
    w = Window.partitionBy("symbol").orderBy(F.col(metric_col).desc(), F.col("lag").asc())

    best = (
        df.withColumn("rn", F.row_number().over(w))
          .filter(F.col("rn") == 1)
          .drop("rn")
    )

    # Keep only what we need (IMPORTANT: don't include metric_col twice!)
    base_cols = ["symbol", "lag", metric_col, "delta_vol", "delta_illiq", "n_shock_obs"]
    optional = ["vol_on_shock", "vol_on_noshock", "illiq_on_shock", "illiq_on_noshock", "n_noshock_obs"]
    cols = base_cols + [c for c in optional if c in best.columns]

    best = best.select(*cols)

    # Rename the chosen metric for clarity
    best_metric_name = f"best_{metric_col}"
    best = best.withColumnRenamed(metric_col, best_metric_name)

    return best.orderBy(F.col(best_metric_name).desc(), F.col("lag").asc())


def typical_lag_stats(best_vol, best_illiq):
    """
    Typical lag stats across symbols:
      mean/median/p25/p75 of the selected best lag.
    """
    def stats_for(best_df, label: str):
        return (
            best_df.agg(
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
    """Histogram: how many symbols have each lag as their 'best'."""
    return (
        best_df.groupBy("lag")
               .agg(F.count("*").alias("n_symbols"))
               .withColumn("metric", F.lit(label))
               .select("metric", "lag", "n_symbols")
               .orderBy("lag")
    )


def write_parquet_and_csv(df, out_base: str):
    """
    Write:
      - parquet to out_base
      - single CSV file to out_base_csv (coalesce(1))
    """
    df.write.mode("overwrite").parquet(out_base)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_base + "_csv")


def main():
    derived_base = _get_derived_base()
    in_path = f"{derived_base}/rq2_results/lag_effects"
    out_base = f"{derived_base}/rq2_results"

    spark = ensure_spark("rq2_best_lag_summary")

    print("Reading RQ2 lag effects from:", in_path)
    df = spark.read.parquet(in_path)

    # ---- Normalize lag column name ----
    if "lag" not in df.columns:
        if "lag_min" in df.columns:
            df = df.withColumnRenamed("lag_min", "lag")
        else:
            raise ValueError(f"Missing lag column: expected 'lag' or 'lag_min'. Found: {df.columns}")

    # ---- Validate required columns ----
    needed = {"symbol", "lag", "delta_vol", "delta_illiq", "n_shock_obs"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in lag_effects: {missing}. Found: {df.columns}")

    # ---- Filter unstable estimates ----
    MIN_SHOCK_OBS = int(os.environ.get("MIN_SHOCK_OBS", "200"))
    df = df.filter(F.col("n_shock_obs") >= F.lit(MIN_SHOCK_OBS))
    print(f"Keeping rows with n_shock_obs >= {MIN_SHOCK_OBS}")

    # ---- Best lag per symbol (two perspectives) ----
    best_vol = best_by_metric(df, "delta_vol")
    best_illiq = best_by_metric(df, "delta_illiq")

    # ---- Typical lag stats + histograms ----
    stats = typical_lag_stats(best_vol, best_illiq)
    hist_vol = best_lag_hist(best_vol, "delta_vol")
    hist_illiq = best_lag_hist(best_illiq, "delta_illiq")

    # ---- Write outputs ----
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

    # ---- Safe previews (small) ----
    print("\nTop 10 symbols by strongest volatility propagation (best_delta_vol):")
    best_vol.limit(10).show(truncate=False)

    print("\nTypical lag stats across symbols:")
    stats.show(truncate=False)

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
