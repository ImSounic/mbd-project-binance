import os


DATA_OWNER = os.environ.get("DATA_OWNER", "s3702111")

DATA_RAW = f"hdfs:///user/{DATA_OWNER}/binance/raw"
DATA_DERIVED = f"hdfs:///user/{DATA_OWNER}/binance/derived"
DATA_RESULTS = f"hdfs:///user/{DATA_OWNER}/binance/results"

# Local paths
DATA_RAW_LOCAL = os.environ.get("DATA_RAW_LOCAL", "binance-dataset")
DATA_DERIVED_LOCAL = os.environ.get("DATA_DERIVED_LOCAL", "data/derived")
DATA_RESULTS_LOCAL = os.environ.get("DATA_RESULTS_LOCAL", "data/results")

IS_LOCAL = os.environ.get("IS_LOCAL", "0") == "1"


def raw_path() -> str:
    return DATA_RAW_LOCAL if IS_LOCAL else DATA_RAW


def derived_path() -> str:
    return DATA_DERIVED_LOCAL if IS_LOCAL else DATA_DERIVED


def results_path() -> str:
    return DATA_RESULTS_LOCAL if IS_LOCAL else DATA_RESULTS


def features_1m_path() -> str:
    # features_1m is produced into derived/features_1m
    return f"{derived_path()}/features_1m"
