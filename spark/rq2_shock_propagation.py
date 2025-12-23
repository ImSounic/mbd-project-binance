"""
RQ2: Shock propagation from BTC / ETH trading activity
------------------------------------------------------

Goal:
- Identify extreme trade activity shocks in BTC and ETH
- Measure how these shocks propagate to SMALL_CAP altcoins
- Estimate typical time lag of liquidity & volatility response

Runs on:
- Spark cluster (YARN)
"""

from pyspark.sql import SparkSession, functions as F
from spark.config import DATA_DERIVED


# -------------------- CONFIG --------------------
MAX_LAG_MINUTES = 60          # propagation window
SHOCK_Q = 0.99                # trade shock percentile
OUTPUT_PATH = f"{DATA_DERIVED}/rq2_shock_propagation"


# -------------------- MAIN --------------------
def main():
    spark = (
        SparkSession.builder
        .appName("rq2_shock_propagation")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("Reading features_1m from:", f"{DATA_DERIVED}/features_1m")
    df = spark.read.parquet(f"{DATA_DERIVED}/features_1m")

    # -------------------- STEP 1: Identify BTC & ETH --------------------
    leaders = (
        df.filter(F.col("symbol").isin("BTC-USDT", "ETH-USDT"))
          .select(
              "open_time",
              "symbol",
              F.col("number_of_trades").alias("leader_trades")
          )
    )

    # -------------------- STEP 2: Compute shock thresholds --------------------
    thresholds = (
        leaders.groupBy("symbol")
        .agg(F.expr(f"percentile_approx(leader_trades, {SHOCK_Q})").alias("thr"))
        .collect()
    )

    thr_map = {r["symbol"]: r["thr"] for r in thresholds}
    print("Shock thresholds:", thr_map)

    # -------------------- STEP 3: Mark shock minutes --------------------
    shocks = (
        leaders
        .withColumn(
            "is_shock",
            F.when(
                ((F.col("symbol") == "BTC-USDT") & (F.col("leader_trades") >= thr_map["BTC-USDT"])) |
                ((F.col("symbol") == "ETH-USDT") & (F.col("leader_trades") >= thr_map["ETH-USDT"])),
                1
            ).otherwise(0)
        )
        .filter(F.col("is_shock") == 1)
        .withColumn("source_asset", F.col("symbol"))
        .select("open_time", "source_asset")
    )

    print("Number of shock minutes:", shocks.count())

    # -------------------- STEP 4: SMALL CAP altcoins --------------------
    altcoins = (
        df.filter(F.col("tier") == "SMALL_CAP")
          .select(
              "open_time",
              "amihud_illiq",
              "parkinson_var_1m",
              F.abs(F.col("log_return")).alias("abs_return")
          )
    )

    # -------------------- STEP 5: Lagged join --------------------
    results = []

    for lag in range(0, MAX_LAG_MINUTES + 1):
        lagged = (
            shocks
            .withColumn("target_time", F.col("open_time") + F.expr(f"INTERVAL {lag} MINUTES"))
            .join(
                altcoins,
                altcoins.open_time == F.col("target_time"),
                "inner"
            )
            .groupBy("source_asset")
            .agg(
                F.avg("amihud_illiq").alias("avg_alt_amihud"),
                F.avg("parkinson_var_1m").alias("avg_alt_volatility"),
                F.avg("abs_return").alias("avg_alt_abs_return"),
                F.count("*").alias("n_obs")
            )
            .withColumn("lag_minutes", F.lit(lag))
        )

        results.append(lagged)

    final = results[0]
    for r in results[1:]:
        final = final.unionByName(r)

    # -------------------- STEP 6: Write output --------------------
    print("Writing RQ2 results to:", OUTPUT_PATH)
    (
        final
        .orderBy("source_asset", "lag_minutes")
        .write
        .mode("overwrite")
        .parquet(OUTPUT_PATH)
    )

    final.show(10, truncate=False)
    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
