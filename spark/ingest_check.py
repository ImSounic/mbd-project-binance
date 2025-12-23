from pyspark.sql import SparkSession
from config import raw_path

spark = (
    SparkSession.builder
    .appName("binance_ingest_check_local")
    .master("local[*]")   # local mode
    .getOrCreate()
)

path = raw_path()
print("Reading:", path)

df = spark.read.parquet(path)
df.printSchema()
df.show(5, truncate=False)

spark.stop()
