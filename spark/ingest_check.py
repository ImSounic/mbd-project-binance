from pyspark.sql import SparkSession
from spark.config import DATA_RAW

spark = SparkSession.builder.appName("binance_ingest_check").getOrCreate()

print("Reading:", DATA_RAW)
df = spark.read.parquet(DATA_RAW)

print("Schema:")
df.printSchema()

print("Sample:")
df.show(5, truncate=False)

# Avoid full scan at first (expensive). Uncomment later if needed:
# print("Count:", df.count())

spark.stop()
