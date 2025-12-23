#!/usr/bin/env python3
"""
RQ1 Visualisations
------------------
Creates PNG figures for RQ1, using:
- Summary results from rq1_compare (small table)
- Optional sampled distributions from features_1m (to show spread/tails)

Outputs:
- PNGs saved to local folder:
    data/results/rq1_figures   (IS_LOCAL=True)
    rq1_figures                (cluster driver local FS)
- If running on cluster, also uploads to HDFS:
    hdfs:///user/<owner>/binance/results/rq1_figures

Why this matters in the report:
- Tables show averages; plots show distributions and tail risk.
- Stress vs non-stress comparison is the "amplification" evidence.
"""

import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F, Window

# ---- config import ----
try:
    from config import IS_LOCAL, DATA_DERIVED, DATA_RESULTS
except Exception:
    from spark.config import IS_LOCAL, DATA_DERIVED, DATA_RESULTS  # type: ignore


# -------------------- SETTINGS --------------------
MAX_PLOT_POINTS = int(os.environ.get("MAX_PLOT_POINTS", "200000"))
SAMPLE_FRACTION = float(os.environ.get("SAMPLE_FRACTION", "0.001"))  # distribution sampling
LOCAL_MAX_FILES = int(os.environ.get("LOCAL_MAX_FILES", "0"))


def derived_features_path() -> str:
    return "data/derived/features_1m" if IS_LOCAL else f"{DATA_DERIVED}/features_1m"


def results_rq1_compare_path() -> str:
    return "data/results/rq1_compare" if IS_LOCAL else f"{DATA_RESULTS}/rq1_compare"


def results_fig_dir_local() -> str:
    # where PNGs are written on the machine running the driver
    return "data/results/rq1_figures" if IS_LOCAL else "rq1_figures"


def results_fig_dir_hdfs() -> str:
    # where to upload PNGs on cluster
    return f"{DATA_RESULTS}/rq1_figures"


def ensure_spark(app: str) -> SparkSession:
    builder = SparkSession.builder.appName(app)
    if IS_LOCAL:
        builder = builder.master("local[*]")
    builder = builder.config("spark.sql.session.timeZone", "UTC")
    return builder.getOrCreate()


def select_input_files_for_local(path: str) -> str:
    if not IS_LOCAL or LOCAL_MAX_FILES <= 0:
        return path

    spark = SparkSession.getActiveSession()
    if spark is None:
        return path

    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hconf)
    p = jvm.org.apache.hadoop.fs.Path(path)

    if not fs.exists(p):
        return path

    statuses = fs.listStatus(p)
    files = []
    for st in statuses:
        name = st.getPath().getName()
        if name.endswith(".parquet"):
            files.append(st.getPath().toString())

    files = sorted(files)[:LOCAL_MAX_FILES]
    if not files:
        return path

    print(f"Local mode: reading {len(files)} parquet files (LOCAL_MAX_FILES={LOCAL_MAX_FILES})")
    return ",".join(files)


def save_barplot(df_pd, x, y, hue, title, ylabel, out_file, logy=False):
    """
    Simple grouped bar chart using pandas DataFrame produced from Spark aggregation.
    """
    # Pivot to make grouped bars easy
    pivot = df_pd.pivot(index=x, columns=hue, values=y)

    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.legend(title=hue)
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


def save_boxplot(sample_pd, value_col, group_cols, title, ylabel, out_file, logy=False):
    """
    Boxplot for sampled distributions.
    group_cols must be exactly 2 columns (e.g., tier and is_stress).
    """
    # create combined group label
    g1, g2 = group_cols
    sample_pd["group"] = sample_pd[g1].astype(str) + " | " + sample_pd[g2].astype(str)

    groups = [g for g in sorted(sample_pd["group"].unique())]
    data = [sample_pd.loc[sample_pd["group"] == g, value_col].dropna().values for g in groups]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=groups, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


def maybe_upload_to_hdfs(local_dir: str, hdfs_dir: str):
    """
    Upload PNGs to HDFS when on cluster.
    """
    if IS_LOCAL:
        return

    # Create HDFS dir and put images
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", hdfs_dir], check=False)
    subprocess.run(["hdfs", "dfs", "-put", "-f", f"{local_dir}/*.png", hdfs_dir], check=False)
    print(f"Uploaded figures to: {hdfs_dir}")


def main():
    spark = ensure_spark("rq1_visualisations")

    # ---------- 1) Read summary results from rq1_compare ----------
    compare_path = results_rq1_compare_path()
    print(f"Reading RQ1 compare summary from: {compare_path}")
    summary = spark.read.parquet(compare_path)

    # Convert to pandas (small)
    summary_pd = summary.toPandas()

    # Output directory for PNGs
    fig_dir = Path(results_fig_dir_local())
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 2) Bar plots from summary (Stress vs Non-Stress) ----------
    save_barplot(
        summary_pd,
        x="tier",
        y="median_quote_volume",
        hue="is_stress",
        title="Median Quote Volume by Tier (Stress vs Non-Stress)",
        ylabel="Median quote_asset_volume",
        out_file=str(fig_dir / "rq1_median_volume.png"),
        logy=True
    )

    save_barplot(
        summary_pd,
        x="tier",
        y="zero_volume_ratio",
        hue="is_stress",
        title="Zero-Volume Ratio by Tier (Stress vs Non-Stress)",
        ylabel="Share of minutes with zero volume",
        out_file=str(fig_dir / "rq1_zero_volume_ratio.png"),
        logy=False
    )

    save_barplot(
        summary_pd,
        x="tier",
        y="avg_parkinson_vol",
        hue="is_stress",
        title="Avg Parkinson Volatility by Tier (Stress vs Non-Stress)",
        ylabel="Avg parkinson_var_1m",
        out_file=str(fig_dir / "rq1_avg_parkinson_vol.png"),
        logy=True
    )

    save_barplot(
        summary_pd,
        x="tier",
        y="p95_amihud",
        hue="is_stress",
        title="95th Percentile Amihud Illiquidity by Tier (Stress vs Non-Stress)",
        ylabel="p95 amihud_illiq (higher = worse liquidity)",
        out_file=str(fig_dir / "rq1_p95_amihud.png"),
        logy=True
    )

    # ---------- 3) Optional distribution plots using sampling ----------
    # Why: show skew and tail behavior; tables hide distribution differences.
    feat_path = derived_features_path()
    feat_path = select_input_files_for_local(feat_path)
    print(f"Reading features_1m (for sampling) from: {feat_path}")

    feat = spark.read.parquet(feat_path).select(
        "open_time", "tier", "is_stress",
        "quote_asset_volume", "amihud_illiq", "parkinson_var_1m", "log_return", "zero_volume_flag"
    )

    # Sample per group (tier, is_stress) approximately, to keep collect safe
    # (We don't need exact sampling; we need a representative distribution picture.)
    sampled = (
        feat
        .withColumn("abs_return", F.abs("log_return"))
        .sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=42)
        .limit(MAX_PLOT_POINTS)
    )

    sampled_pd = sampled.toPandas()

    # Boxplots (log scale for heavy tails)
    save_boxplot(
        sampled_pd,
        value_col="quote_asset_volume",
        group_cols=("tier", "is_stress"),
        title="Distribution: Quote Volume (sampled)",
        ylabel="quote_asset_volume",
        out_file=str(fig_dir / "rq1_dist_quote_volume.png"),
        logy=True
    )

    save_boxplot(
        sampled_pd,
        value_col="amihud_illiq",
        group_cols=("tier", "is_stress"),
        title="Distribution: Amihud Illiquidity (sampled)",
        ylabel="amihud_illiq (higher = worse liquidity)",
        out_file=str(fig_dir / "rq1_dist_amihud.png"),
        logy=True
    )

    save_boxplot(
        sampled_pd,
        value_col="parkinson_var_1m",
        group_cols=("tier", "is_stress"),
        title="Distribution: Parkinson Volatility (sampled)",
        ylabel="parkinson_var_1m",
        out_file=str(fig_dir / "rq1_dist_parkinson.png"),
        logy=True
    )

    save_boxplot(
        sampled_pd,
        value_col="abs_return",
        group_cols=("tier", "is_stress"),
        title="Distribution: Absolute Log Return (sampled)",
        ylabel="abs(log_return)",
        out_file=str(fig_dir / "rq1_dist_abs_return.png"),
        logy=True
    )

    # ---------- 4) Upload PNGs to HDFS (cluster) ----------
    maybe_upload_to_hdfs(str(fig_dir), results_fig_dir_hdfs())

    spark.stop()
    print(f"Saved figures to: {fig_dir.resolve()}")
    if not IS_LOCAL:
        print(f"And uploaded to HDFS: {results_fig_dir_hdfs()}")
    print("Done.")


if __name__ == "__main__":
    main()
