from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import DATA_DERIVED, IS_LOCAL

# -------------------------
# Spark session
# -------------------------
spark = (
    SparkSession.builder
    .appName("RQ1_Stress_Liquidity_Volatility")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# -------------------------
# Paths
# -------------------------
FEATURES_1M = f"{DATA_DERIVED}/features_1m"
OUTPUT_PATH = f"{DATA_DERIVED}/rq1_results"

# -------------------------
# Load features
# -------------------------
print("Reading features_1m from:", FEATURES_1M)
df = spark.read.parquet(FEATURES_1M)

# -------------------------
# Tier classification
# -------------------------
df = df.withColumn(
    "tier",
    F.when(F.col("base_asset").isin("BTC", "ETH"), "LARGE_CAP")
     .otherwise("SMALL_CAP")
)

# -------------------------
# Create daily BTC volatility
# -------------------------
btc = df.filter(F.col("base_asset") == "BTC")

btc_daily_vol = (
    btc
    .withColumn("date", F.to_date("open_time"))
    .groupBy("date")
    .agg(
        F.avg("parkinson_var_1m").alias("btc_daily_vol")
    )
)

# -------------------------
# Identify stress days (top 5%)
# -------------------------
quantiles = btc_daily_vol.approxQuantile(
    "btc_daily_vol", [0.95], 0.01
)
stress_threshold = quantiles[0]

print("BTC stress threshold (95th percentile):", stress_threshold)

stress_days = (
    btc_daily_vol
    .filter(F.col("btc_daily_vol") >= stress_threshold)
    .select("date")
    .withColumn("is_stress", F.lit(1))
)

# -------------------------
# Join stress indicator back
# -------------------------
df = (
    df
    .withColumn("date", F.to_date("open_time"))
    .join(stress_days, on="date", how="inner")
)

# -------------------------
# Aggregate metrics by tier
# -------------------------
results = (
    df
    .groupBy("tier")
    .agg(
        # Liquidity
        F.avg("quote_asset_volume").alias("avg_quote_volume"),
        F.expr("percentile_approx(quote_asset_volume, 0.5)").alias("median_quote_volume"),
        F.avg("amihud_illiq").alias("avg_amihud"),
        F.expr("percentile_approx(amihud_illiq, 0.95)").alias("p95_amihud"),
        F.avg(F.col("zero_volume_flag").cast("int")).alias("zero_volume_ratio"),

        # Volatility
        F.avg("parkinson_var_1m").alias("avg_parkinson_vol"),
        F.expr("percentile_approx(parkinson_var_1m, 0.95)").alias("p95_parkinson_vol"),
        F.avg(F.abs("log_return")).alias("avg_abs_return")
    )
)

# -------------------------
# Show results
# -------------------------
print("\nRQ1 Results (Stress Periods Only):")
results.show(truncate=False)

# -------------------------
# Write output
# -------------------------
print("Writing RQ1 results to:", OUTPUT_PATH)
results.coalesce(1).write.mode("overwrite").parquet(OUTPUT_PATH)

spark.stop()
print("Done.")
