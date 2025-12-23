# Edit these if the "data owner" differs.
DATA_OWNER = "s3702111"

DATA_RAW = f"hdfs:///user/{DATA_OWNER}/binance/raw"
DATA_DERIVED = f"hdfs:///user/{DATA_OWNER}/binance/derived"
DATA_RESULTS = f"hdfs:///user/{DATA_OWNER}/binance/results"

# Local path (my(SOUNIC) VS Code folder has binance-dataset/)
DATA_RAW_LOCAL = "binance-dataset"

# Toggle this when running locally vs cluster
IS_LOCAL = True

def raw_path() -> str:
    return DATA_RAW_LOCAL if IS_LOCAL else DATA_RAW

def derived_path() -> str:
    return "data/derived" if IS_LOCAL else DATA_DERIVED

def results_path() -> str:
    return "data/results" if IS_LOCAL else DATA_RESULTS
