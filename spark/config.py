# spark/config.py
import os

# Edit these if the "data owner" differs.
DATA_OWNER = os.getenv("DATA_OWNER", "s3702111")

# HDFS locations (cluster)
DATA_RAW = f"hdfs:///user/{DATA_OWNER}/binance/raw"
DATA_DERIVED = f"hdfs:///user/{DATA_OWNER}/binance/derived"
DATA_RESULTS = f"hdfs:///user/{DATA_OWNER}/binance/results"

# Local locations (your laptop)
# Your repo has: binance-dataset/ (parquet files)
DATA_RAW_LOCAL = os.getenv("DATA_RAW_LOCAL", "binance-dataset")

# Derived output locally goes into repo/data/derived
DATA_DERIVED_LOCAL = os.getenv("DATA_DERIVED_LOCAL", "data/derived")

# Toggle local vs cluster:
# - On Mac: export IS_LOCAL=1 (or set True below)
# - On cluster: export IS_LOCAL=0 (or set False below)
IS_LOCAL = os.getenv("IS_LOCAL", "1").lower() in ("1", "true", "yes")

def raw_path() -> str:
    return DATA_RAW_LOCAL if IS_LOCAL else DATA_RAW

def derived_path(subdir: str) -> str:
    if IS_LOCAL:
        return os.path.join(DATA_DERIVED_LOCAL, subdir)
    return f"{DATA_DERIVED}/{subdir}"
