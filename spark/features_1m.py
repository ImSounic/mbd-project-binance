# spark/features_1m.py
import os
import glob
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import IS_LOCAL, raw_path, derived_path

# Local dev safety: how many files to read from binance-dataset/
LOCAL_MAX_FILES = int(os.getenv("LOCAL_MAX_FILES", "10"))

def build_spark(app_name: str) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)

    if IS_LOCAL:
        builder = builder.master("local[*]")

    # Reasonable defaults (you can still override via spark-submit --conf)
    builder = (
        builder
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "400")
    )

    return builder.getOrCreate()

def read_raw_with_symbol(spark: SparkSession):
    in_path = raw_path()
    print("Reading from:", in_path)

    if IS_LOCAL:
        # Read a subset of files locally to avoid memory blow-ups
        files = sorted(glob.glob(os.path.join(in_path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files found under: {in_path}")

        files = files[:LOCAL_MAX_FILES]
        print(f"Local mode: reading {len(files)} parquet files (LOCAL_MAX_FILES={LOCAL_MAX_FILES})")

        df = spark.read.parquet(*files)

        # Symbol from local path ".../AAVE-BUSD.parquet" -> "AAVE-BUSD"
        df = df.withColumn(
            "symbol",
            F.regexp_extract(F.input_file_name(), r"([^/]+)\.parquet$", 1)
        )
    else:
        # Cluster mode: read directory in one shot (no 1000 unions)
        # input_file_name() gives ".../raw/AAVE-BUSD.parquet"
        df = spark.read.parquet(in_path).withColumn(
            "symbol",
            F.regexp_extract(F.input_file_name(), r"/([^/]+)\.parquet$", 1)
        )

    df = (
        df.withColumn("base_asset", F.split("symbol", "-").getItem(0))
          .withColumn("quote_asset", F.split("symbol", "-").getItem(1))
    )

    return df

def compute_features_1m(df):
    # Ensure consistent numeric types (Spark sometimes infers differently)
    df = (
        df.withColumn("open", F.col("open").cast("double"))
          .withColumn("high", F.col("high").cast("double"))
          .withColumn("low", F.col("low").cast("double"))
          .withColumn("close", F.col("close").cast("double"))
          .withColumn("volume", F.col("volume").cast("double"))
          .withColumn("quote_asset_volume", F.col("quote_asset_volume").cast("double"))
          .withColumn("number_of_trades", F.col("number_of_trades").cast("long"))
          .withColumn("taker_buy_base_asset_volume", F.col("taker_buy_base_asset_volume").cast("double"))
          .withColumn("taker_buy_quote_asset_volume", F.col("taker_buy_quote_asset_volume").cast("double"))
    )

    # Window by symbol/time
    w = Window.partitionBy("symbol").orderBy("open_time")

    # Log return = log(close_t / close_{t-1})
    prev_close = F.lag("close", 1).over(w)
    log_return = F.when(
        (prev_close.isNotNull()) & (F.col("close") > 0) & (prev_close > 0),
        F.log(F.col("close") / prev_close)
    ).otherwise(F.lit(None).cast("double"))

    # Taker ratio = taker_buy_quote_asset_volume / quote_asset_volume
    taker_ratio = F.when(
        F.col("quote_asset_volume") > 0,
        F.col("taker_buy_quote_asset_volume") / F.col("quote_asset_volume")
    ).otherwise(F.lit(0.0))

    # Amihud illiquidity proxy (minute): |return| / dollar volume (here quote_asset_volume)
    amihud_illiq = F.when(
        (F.col("quote_asset_volume") > 0) & log_return.isNotNull(),
        F.abs(log_return) / F.col("quote_asset_volume")
    ).otherwise(F.lit(None).cast("double"))

    # Zero-volume minute flag
    zero_volume_flag = (F.col("volume") == 0) | F.col("volume").isNull()

    # Parkinson variance proxy for 1-minute:
    # (1 / (4*ln(2))) * (ln(high/low))^2
    ln2 = F.lit(0.6931471805599453)
    parkinson_var_1m = F.when(
        (F.col("high") > 0) & (F.col("low") > 0) & (F.col("high") >= F.col("low")),
        (F.lit(1.0) / (F.lit(4.0) * ln2)) * F.pow(F.log(F.col("high") / F.col("low")), 2)
    ).otherwise(F.lit(None).cast("double"))

    out = (
        df
        .withColumn("log_return", log_return)
        .withColumn("taker_ratio", taker_ratio)
        .withColumn("amihud_illiq", amihud_illiq)
        .withColumn("zero_volume_flag", zero_volume_flag)
        .withColumn("parkinson_var_1m", parkinson_var_1m)
    )

    # Helpful to stabilize the window computation / reduce executor randomness:
    # One shuffle up-front, then sort locally inside partitions.
    out = out.repartition(200, "symbol").sortWithinPartitions("symbol", "open_time")

    return out

def main():
    spark = build_spark("binance_features_1m")

    df = read_raw_with_symbol(spark)

    # Select only needed columns (keeps memory lower)
    df = df.select(
        "open_time", "symbol", "base_asset", "quote_asset",
        "open", "high", "low", "close",
        "volume", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    )

    features = compute_features_1m(df)

    out_path = derived_path("features_1m")
    print("Writing:", out_path)

    # Keep file count sane:
    # - Local: small
    # - Cluster: still not thousands
    if IS_LOCAL:
        final_to_write = features.coalesce(8)
    else:
        final_to_write = features.coalesce(400)

    final_to_write.write.mode("overwrite").parquet(out_path)

    print("Preview of derived features_1m:")
    features.show(5, truncate=False)

    spark.stop()
    print("Done.")

if __name__ == "__main__":
    main()
