# spark/rq2_best_lag_summary.py
"""
RQ2 Summary: Best lag + typical lag distribution

Input (produced by rq2_shock_propagation.py):
  - {derived_base}/rq2_results/lag_effects   (parquet)

Expected columns (your current pipeline):
  symbol,
  lag_min,
  delta_vol,
  delta_illiq,
  n_shock_obs,
  (optionally) vol_on_shock, vol_on_noshock, illiq_on_shock, illiq_on_noshock, n_noshock_obs

Output:
  - {derived_base}/rq2_results/best_lag_by_vol     (parquet + csv)
  - {derived_base}/rq2_results/best_lag_by_illiq   (parquet + csv)
  - {derived_base}/rq2_results/lag_typical_stats   (parquet + csv)
  - {derived_base}/rq2_results/best_lag_hist_vol   (parquet + csv)
  - {derived_base}/rq2_results/best_lag_hist_illiq (parquet + csv)

Why this file exists:
  - Avoid big .show()/collect on driver (can OOM).
  - Produce stable, report-ready summary tables.
"""

import os
from pyspark.sql import SparkSession, functions as F, Window


# -------------------- Config helpers --------------------
def _get_derived_base() -> str:
    """
    Uses config.py if available.
    - Cluster: hdfs:///user/<id>/binance/derived
    - Local:   data/derived
    """
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


# -------------------- Core logic --------------------
def best_by_metric(df, metric_col: str):
    """
    For each symbol, pick the lag that maximizes `metric_col`.
    Tie-breaker: smaller lag (faster propagation).
    """
    w = Window.partitionBy("symbol").orderBy(F.col(metric_col).desc(), F.col("lag").asc())

    best = (
        df.withColumn("rn", F.row_number().over(w))
          .filter(F.col("rn") == 1)
          .drop("rn")
    )

    # Keep useful context columns if they exist
    cols = ["symbol", "lag", metric_col, "delta_vol", "delta_illiq", "n_shock_obs"]
    optional = ["vol_on_shock", "vol_on_noshock", "illiq_on_shock", "illiq_on_noshock", "n_noshock_obs"]
    cols += [c for c in optional if c in best.columns]

    # Rename the chosen metric column for clarity
    best = best.select(*cols).withColumnRenamed(metric_col, f"best_{metric_col}")

    return best.orderBy(F.col(f"best_{metric_col}").desc())


def typical_lag_stats(best_vol, best_illiq):
    """
    Typical lag summary stats across symbols:
      - mean lag
      - median lag
      - p25/p75 lag
      - number of symbols
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
    """How often each lag is 'best' across symbols."""
    return (
        best_df.groupBy("lag")
               .agg(F.count("*").alias("n_symbols"))
               .withColumn("metric", F.lit(label))
               .select("metric", "lag", "n_symbols")
               .orderBy("lag")
    )


def write_parquet_and_csv(df, out_base: str):
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
    # Your file has lag_min. Some versions might have lag.
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

    # ---- Best lag per symbol ----
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

    # ---- Small safe previews ----
    print("\nTop 10 symbols by strongest volatility propagation (best delta_vol):")
    best_vol.limit(10).show(truncate=False)

    print("\nTypical lag stats across symbols:")
    stats.show(truncate=False)

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
