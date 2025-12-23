# spark/rq2_best_lag_summary.py

import os
from pyspark.sql import functions as F, Window
from spark.config import ensure_spark, derived_path

# -------------------- CONFIG --------------------
MIN_SHOCK_OBS = int(os.environ.get("MIN_SHOCK_OBS", "200"))

# Leveraged token filters
LEVERAGED_PATTERNS = ["UP", "DOWN", "BULL", "BEAR"]

# -------------------- HELPERS --------------------
def is_leveraged(col):
    """
    Returns a boolean column identifying leveraged tokens by symbol name.
    """
    cond = None
    for p in LEVERAGED_PATTERNS:
        c = col.contains(p)
        cond = c if cond is None else (cond | c)
    return cond


def best_by_metric(df, metric_col):
    """
    For each symbol, pick the lag with the strongest metric response.
    """
    w = Window.partitionBy("symbol").orderBy(
        F.col(metric_col).desc(),
        F.col("lag").asc()
    )

    best = (
        df.withColumn("rn", F.row_number().over(w))
          .filter(F.col("rn") == 1)
          .drop("rn")
          .withColumn("best_metric", F.col(metric_col))
          .withColumn("best_metric_name", F.lit(metric_col))
    )

    return best.orderBy(F.col("best_metric").desc(), F.col("lag").asc())


def lag_stats(df, metric_col):
    """
    Summary stats of best lag distribution.
    """
    return (
        df.groupBy("best_metric_name")
          .agg(
              F.count("*").alias("n_symbols"),
              F.mean("lag").alias("mean_lag"),
              F.expr("percentile(lag, 0.5)").alias("median_lag"),
              F.expr("percentile(lag, 0.25)").alias("p25_lag"),
              F.expr("percentile(lag, 0.75)").alias("p75_lag"),
          )
    )


# -------------------- MAIN --------------------
def main():
    spark = ensure_spark("rq2_best_lag_summary")

    in_path = derived_path("rq2_results/lag_effects")
    out_base = derived_path("rq2_results")

    print("Reading RQ2 lag effects from:", in_path)

    df = spark.read.parquet(in_path)

    # Rename lag_min -> lag if needed
    if "lag_min" in df.columns and "lag" not in df.columns:
        df = df.withColumnRenamed("lag_min", "lag")

    # -------------------- FILTER 1: Enough observations --------------------
    print(f"Keeping rows with n_shock_obs >= {MIN_SHOCK_OBS}")
    df = df.filter(F.col("n_shock_obs") >= MIN_SHOCK_OBS)

    # -------------------- FILTER 2: Remove leveraged tokens --------------------
    print("Removing leveraged tokens (UP/DOWN/BULL/BEAR)")
    df = df.filter(~is_leveraged(F.col("symbol")))

    # -------------------- BEST LAG PER METRIC --------------------
    best_vol = best_by_metric(df, "delta_vol")
    best_illiq = best_by_metric(df, "delta_illiq")

    print("Writing:", f"{out_base}/best_lag_by_vol")
    best_vol.write.mode("overwrite").parquet(f"{out_base}/best_lag_by_vol")

    print("Writing:", f"{out_base}/best_lag_by_illiq")
    best_illiq.write.mode("overwrite").parquet(f"{out_base}/best_lag_by_illiq")

    # -------------------- LAG DISTRIBUTION STATS --------------------
    stats = lag_stats(
        best_vol.select("symbol", "lag", "best_metric_name")
        .unionByName(best_illiq.select("symbol", "lag", "best_metric_name"))
    )

    print("Writing:", f"{out_base}/lag_typical_stats")
    stats.write.mode("overwrite").parquet(f"{out_base}/lag_typical_stats")

    # -------------------- HISTOGRAM DATA --------------------
    print("Writing:", f"{out_base}/best_lag_hist_vol")
    best_vol.select("lag").write.mode("overwrite").parquet(
        f"{out_base}/best_lag_hist_vol"
    )

    print("Writing:", f"{out_base}/best_lag_hist_illiq")
    best_illiq.select("lag").write.mode("overwrite").parquet(
        f"{out_base}/best_lag_hist_illiq"
    )

    # -------------------- PREVIEW --------------------
    print("\nTop 10 symbols by strongest propagation (spot only):")
    best_vol.show(10, truncate=False)

    print("\nTypical lag stats across symbols:")
    stats.show(truncate=False)

    print("Done.")


if __name__ == "__main__":
    main()
