"""
Validation script for features_1m dataset.

Purpose:
- Verify dataset completeness and correctness
- Check time and symbol coverage
- Inspect null rates in derived features
- Detect duplicate (symbol, open_time) keys

This script performs only aggregations and scans.
It does NOT modify or overwrite data.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    spark = (
        SparkSession.builder
        .appName("validate_features_1m")
        .getOrCreate()
    )

    # Path to derived dataset
    FEATURES_PATH = "hdfs:///user/s3702111/binance/derived/features_1m"

    df = spark.read.parquet(FEATURES_PATH)

    print("\n=== BASIC DATASET INFO ===")
    print(f"Rows: {df.count()}")
    print(f"Columns: {len(df.columns)}")

    print("\n=== TIME COVERAGE ===")
    df.select(
        F.min("open_time").alias("min_open_time"),
        F.max("open_time").alias("max_open_time")
    ).show(truncate=False)

    print("\n=== SYMBOL COVERAGE ===")
    df.select(
        F.countDistinct("symbol").alias("n_symbols"),
        F.countDistinct("base_asset").alias("n_base_assets")
    ).show(truncate=False)

    print("\n=== NULL RATE CHECK (DERIVED FEATURES) ===")

    derived_cols = [
        "log_return",
        "amihud_illiq",
        "parkinson_var_1m",
        "taker_ratio"
    ]

    null_rate_exprs = [
        (
            F.sum(F.col(c).isNull().cast("int")) / F.count("*")
        ).alias(f"{c}_null_rate")
        for c in derived_cols
    ]

    df.select(*null_rate_exprs).show(truncate=False)

    print("\n=== ZERO VOLUME FREQUENCY ===")
    df.select(
        (F.sum(F.col("zero_volume_flag").cast("int")) / F.count("*"))
        .alias("zero_volume_rate")
    ).show(truncate=False)

    print("\n=== DUPLICATE KEY CHECK ===")
    dup_count = (
        df.groupBy("symbol", "open_time")
          .count()
          .filter(F.col("count") > 1)
          .count()
    )

    print(f"Duplicate (symbol, open_time) rows: {dup_count}")

    if dup_count == 0:
        print("PASS: No duplicate keys detected.")
    else:
        print("WARNING: Duplicate keys found!")

    print("\n=== VALIDATION COMPLETE ===")

    spark.stop()


if __name__ == "__main__":
    main()
