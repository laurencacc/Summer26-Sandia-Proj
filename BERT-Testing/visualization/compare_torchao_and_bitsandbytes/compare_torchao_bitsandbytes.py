from pathlib import Path
import os
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_TORCHAO_CSV = Path(
    "/home/lecacca/VSCode Environment/BERT-Testing/visualization/"
    "scatter_plot_torchao/results/job_1399997/manual_vs_torchao_results.csv"
)

DEFAULT_BNB_CSV = Path(
    "/home/lecacca/VSCode Environment/BERT-Testing/visualization/"
    "scatter_plot_bitsandbytes/results/job_1399993/manual_vs_bitsandbytes_results.csv"
)

DEFAULT_OUT_DIR = Path(
    "/home/lecacca/VSCode Environment/BERT-Testing/visualization/"
    "compare_torchao_and_bitsandbytes/results"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare TorchAO and BitsAndBytes quantization results, "
            "and sanity-check whether the manual quantization path matches "
            "between the two notebooks."
        )
    )

    parser.add_argument(
        "--torchao-csv",
        type=Path,
        default=DEFAULT_TORCHAO_CSV,
        help="Path to manual_vs_torchao_results.csv",
    )

    parser.add_argument(
        "--bnb-csv",
        type=Path,
        default=DEFAULT_BNB_CSV,
        help="Path to manual_vs_bitsandbytes_results.csv",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. If omitted, uses RESULTS_DIR if set; "
            "otherwise uses the default compare_torchao_and_bitsandbytes/results directory."
        ),
    )

    return parser.parse_args()


def get_out_dir(args):
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = Path(os.environ.get("RESULTS_DIR", DEFAULT_OUT_DIR))

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_results(torchao_csv, bnb_csv):
    if not torchao_csv.exists():
        raise FileNotFoundError(f"TorchAO CSV not found: {torchao_csv}")

    if not bnb_csv.exists():
        raise FileNotFoundError(f"BitsAndBytes CSV not found: {bnb_csv}")

    torchao_df = pd.read_csv(torchao_csv)
    bnb_df = pd.read_csv(bnb_csv)

    print("Loaded input CSVs")
    print(f"TorchAO CSV:      {torchao_csv}")
    print(f"BitsAndBytes CSV: {bnb_csv}")
    print(f"TorchAO rows:      {len(torchao_df)}")
    print(f"BitsAndBytes rows: {len(bnb_df)}")
    print()

    return torchao_df, bnb_df


def save_input_summaries(torchao_df, bnb_df, out_dir):
    torchao_counts = (
        torchao_df.groupby(["block_index", "sub_layer"])
        .size()
        .reset_index(name="n_rows")
    )

    bnb_counts = (
        bnb_df.groupby(["block_index", "sub_layer"])
        .size()
        .reset_index(name="n_rows")
    )

    torchao_counts_csv = out_dir / "torchao_input_counts_by_sublayer.csv"
    bnb_counts_csv = out_dir / "bitsandbytes_input_counts_by_sublayer.csv"

    torchao_counts.to_csv(torchao_counts_csv, index=False)
    bnb_counts.to_csv(bnb_counts_csv, index=False)

    print("Saved input count summaries")
    print(f"  {torchao_counts_csv}")
    print(f"  {bnb_counts_csv}")
    print()


def merge_results(torchao_df, bnb_df, out_dir):
    merge_cols = [
        "block_index",
        "sub_layer",
        "text_index",
        "text",
    ]

    comparison_df = torchao_df.merge(
        bnb_df,
        on=merge_cols,
        how="inner",
        suffixes=("_torchao_file", "_bnb_file"),
    )

    print("Merge summary")
    print(f"Merged rows:        {len(comparison_df)}")
    print(f"TorchAO rows:       {len(torchao_df)}")
    print(f"BitsAndBytes rows:  {len(bnb_df)}")

    expected_rows = min(len(torchao_df), len(bnb_df))
    if len(comparison_df) != expected_rows:
        print(
            "WARNING: merged row count differs from min(input row counts). "
            "Writing unmatched row files."
        )

    torchao_keys = torchao_df[merge_cols].drop_duplicates()
    bnb_keys = bnb_df[merge_cols].drop_duplicates()

    torchao_only = torchao_keys.merge(
        bnb_keys,
        on=merge_cols,
        how="left",
        indicator=True,
    )
    torchao_only = torchao_only[
        torchao_only["_merge"] == "left_only"
    ].drop(columns=["_merge"])

    bnb_only = bnb_keys.merge(
        torchao_keys,
        on=merge_cols,
        how="left",
        indicator=True,
    )
    bnb_only = bnb_only[
        bnb_only["_merge"] == "left_only"
    ].drop(columns=["_merge"])

    torchao_only_csv = out_dir / "rows_only_in_torchao.csv"
    bnb_only_csv = out_dir / "rows_only_in_bitsandbytes.csv"

    torchao_only.to_csv(torchao_only_csv, index=False)
    bnb_only.to_csv(bnb_only_csv, index=False)

    print(f"Rows only in TorchAO:      {len(torchao_only)}")
    print(f"Rows only in BitsAndBytes: {len(bnb_only)}")
    print(f"Saved: {torchao_only_csv}")
    print(f"Saved: {bnb_only_csv}")
    print()

    comparison_df = comparison_df.rename(
        columns={
            "manual_top1_torchao_file": "manual_top1_from_torchao_notebook",
            "manual_top1_bnb_file": "manual_top1_from_bnb_notebook",
            "manual_top1_match_torchao_file": "manual_top1_match_from_torchao_notebook",
            "manual_top1_match_bnb_file": "manual_top1_match_from_bnb_notebook",
            "manual_mean_abs_logit_diff_torchao_file": (
                "manual_mean_abs_logit_diff_from_torchao_notebook"
            ),
            "manual_mean_abs_logit_diff_bnb_file": (
                "manual_mean_abs_logit_diff_from_bnb_notebook"
            ),
            "manual_max_abs_logit_diff_torchao_file": (
                "manual_max_abs_logit_diff_from_torchao_notebook"
            ),
            "manual_max_abs_logit_diff_bnb_file": (
                "manual_max_abs_logit_diff_from_bnb_notebook"
            ),
        }
    )

    return comparison_df

def add_baseline_sanity_columns(comparison_df):
    """
    Check whether the FP32 baseline top-1 prediction agrees between
    the TorchAO CSV and the BitsAndBytes CSV.

    Note:
    This checks only baseline top-1, because the current CSVs do not contain
    full baseline logits or baseline checksum values.
    """

    comparison_df["baseline_top1_same_between_files"] = (
        comparison_df["baseline_top1_torchao_file"]
        == comparison_df["baseline_top1_bnb_file"]
    )

    # Keep a convenient unified baseline_top1 column.
    comparison_df["baseline_top1"] = comparison_df["baseline_top1_torchao_file"]

    return comparison_df


def summarize_baseline_sanity(comparison_df, out_dir):
    baseline_sanity_summary = (
        comparison_df.groupby(["block_index", "sub_layer"])
        .agg(
            n_rows=("text_index", "count"),
            baseline_top1_same_rate=("baseline_top1_same_between_files", "mean"),
            n_baseline_top1_mismatches=(
                "baseline_top1_same_between_files",
                lambda x: (~x).sum(),
            ),
        )
        .reset_index()
    )

    baseline_mismatches = comparison_df[
        ~comparison_df["baseline_top1_same_between_files"]
    ].copy()

    baseline_sanity_csv = out_dir / "baseline_sanity_summary_by_sublayer.csv"
    baseline_mismatches_csv = out_dir / "baseline_top1_mismatches.csv"

    baseline_sanity_summary.to_csv(baseline_sanity_csv, index=False)
    baseline_mismatches.to_csv(baseline_mismatches_csv, index=False)

    print("Baseline sanity summary")
    print(baseline_sanity_summary.to_string(index=False))
    print()
    print(f"Saved: {baseline_sanity_csv}")
    print(f"Saved: {baseline_mismatches_csv}")
    print()

    return baseline_sanity_summary

def validate_sanity_checks(comparison_df):
    """
    Print high-level sanity-check results.

    This does not stop the script. It reports whether the comparison is clean.
    """

    print("=" * 80)
    print("SANITY CHECK RESULTS")
    print("=" * 80)

    n_rows = len(comparison_df)

    # ------------------------------------------------------------------
    # Baseline sanity
    # ------------------------------------------------------------------

    n_baseline_mismatches = (
        ~comparison_df["baseline_top1_same_between_files"]
    ).sum()

    print(f"Rows compared: {n_rows}")
    print(f"Baseline top-1 mismatches: {n_baseline_mismatches}")

    if n_baseline_mismatches == 0:
        print("PASS: Baseline top-1 predictions match between files.")
    else:
        print("WARNING: Baseline top-1 predictions do not always match.")

    # ------------------------------------------------------------------
    # Manual top-1 sanity
    # ------------------------------------------------------------------

    n_manual_top1_mismatches = (
        ~comparison_df["manual_top1_same_between_notebooks"]
    ).sum()

    print(f"Manual top-1 mismatches between notebooks: {n_manual_top1_mismatches}")

    if n_manual_top1_mismatches == 0:
        print("PASS: Manual top-1 predictions match between notebooks.")
    else:
        print("WARNING: Manual top-1 predictions differ between notebooks.")

    # ------------------------------------------------------------------
    # Manual logit-drift sanity
    # ------------------------------------------------------------------

    mean_abs_manual_delta = (
        comparison_df["manual_mean_abs_logit_diff_abs_delta_between_notebooks"]
        .mean()
    )

    max_abs_manual_delta = (
        comparison_df["manual_mean_abs_logit_diff_abs_delta_between_notebooks"]
        .max()
    )

    print(
        "Manual mean-abs-logit-diff absolute delta between notebooks, mean: "
        f"{mean_abs_manual_delta:.10g}"
    )

    print(
        "Manual mean-abs-logit-diff absolute delta between notebooks, max: "
        f"{max_abs_manual_delta:.10g}"
    )

    if max_abs_manual_delta < 1e-8:
        print("PASS: Manual logit-drift values are effectively identical.")
    else:
        print(
            "WARNING: Manual logit-drift values differ between notebooks. "
            "This means the manual reference path is not identical."
        )

    print("=" * 80)
    print()

def add_manual_sanity_columns(comparison_df):
    comparison_df["manual_top1_same_between_notebooks"] = (
        comparison_df["manual_top1_from_torchao_notebook"]
        == comparison_df["manual_top1_from_bnb_notebook"]
    )

    comparison_df["manual_top1_match_same_between_notebooks"] = (
        comparison_df["manual_top1_match_from_torchao_notebook"]
        == comparison_df["manual_top1_match_from_bnb_notebook"]
    )

    comparison_df["manual_mean_abs_logit_diff_delta_bnb_minus_torchao_notebook"] = (
        comparison_df["manual_mean_abs_logit_diff_from_bnb_notebook"]
        - comparison_df["manual_mean_abs_logit_diff_from_torchao_notebook"]
    )

    comparison_df["manual_max_abs_logit_diff_delta_bnb_minus_torchao_notebook"] = (
        comparison_df["manual_max_abs_logit_diff_from_bnb_notebook"]
        - comparison_df["manual_max_abs_logit_diff_from_torchao_notebook"]
    )

    comparison_df["manual_mean_abs_logit_diff_abs_delta_between_notebooks"] = (
        comparison_df["manual_mean_abs_logit_diff_delta_bnb_minus_torchao_notebook"].abs()
    )

    comparison_df["manual_max_abs_logit_diff_abs_delta_between_notebooks"] = (
        comparison_df["manual_max_abs_logit_diff_delta_bnb_minus_torchao_notebook"].abs()
    )

    comparison_df["manual_mean_abs_logit_diff_ratio_bnb_over_torchao_notebook"] = (
        comparison_df["manual_mean_abs_logit_diff_from_bnb_notebook"]
        / comparison_df["manual_mean_abs_logit_diff_from_torchao_notebook"].replace(0, np.nan)
    )

    comparison_df["manual_max_abs_logit_diff_ratio_bnb_over_torchao_notebook"] = (
        comparison_df["manual_max_abs_logit_diff_from_bnb_notebook"]
        / comparison_df["manual_max_abs_logit_diff_from_torchao_notebook"].replace(0, np.nan)
    )

    return comparison_df


def summarize_manual_sanity(comparison_df, out_dir):
    manual_sanity_summary = (
        comparison_df.groupby(["block_index", "sub_layer"])
        .agg(
            n_rows=("text_index", "count"),

            manual_top1_same_rate=("manual_top1_same_between_notebooks", "mean"),
            manual_top1_match_same_rate=(
                "manual_top1_match_same_between_notebooks",
                "mean",
            ),

            manual_mean_abs_logit_diff_torchao_notebook_mean=(
                "manual_mean_abs_logit_diff_from_torchao_notebook",
                "mean",
            ),
            manual_mean_abs_logit_diff_bnb_notebook_mean=(
                "manual_mean_abs_logit_diff_from_bnb_notebook",
                "mean",
            ),
            manual_mean_abs_logit_diff_delta_mean=(
                "manual_mean_abs_logit_diff_delta_bnb_minus_torchao_notebook",
                "mean",
            ),
            manual_mean_abs_logit_diff_abs_delta_mean=(
                "manual_mean_abs_logit_diff_abs_delta_between_notebooks",
                "mean",
            ),
            manual_mean_abs_logit_diff_abs_delta_max=(
                "manual_mean_abs_logit_diff_abs_delta_between_notebooks",
                "max",
            ),
            manual_mean_abs_logit_diff_ratio_bnb_over_torchao_mean=(
                "manual_mean_abs_logit_diff_ratio_bnb_over_torchao_notebook",
                "mean",
            ),

            manual_max_abs_logit_diff_torchao_notebook_mean=(
                "manual_max_abs_logit_diff_from_torchao_notebook",
                "mean",
            ),
            manual_max_abs_logit_diff_bnb_notebook_mean=(
                "manual_max_abs_logit_diff_from_bnb_notebook",
                "mean",
            ),
            manual_max_abs_logit_diff_delta_mean=(
                "manual_max_abs_logit_diff_delta_bnb_minus_torchao_notebook",
                "mean",
            ),
            manual_max_abs_logit_diff_abs_delta_mean=(
                "manual_max_abs_logit_diff_abs_delta_between_notebooks",
                "mean",
            ),
            manual_max_abs_logit_diff_abs_delta_max=(
                "manual_max_abs_logit_diff_abs_delta_between_notebooks",
                "max",
            ),
            manual_max_abs_logit_diff_ratio_bnb_over_torchao_mean=(
                "manual_max_abs_logit_diff_ratio_bnb_over_torchao_notebook",
                "mean",
            ),
        )
        .reset_index()
    )

    manual_sanity_csv = out_dir / "manual_quantization_sanity_summary_by_sublayer.csv"
    manual_sanity_summary.to_csv(manual_sanity_csv, index=False)

    largest_manual_mean_disagreements = comparison_df.sort_values(
        "manual_mean_abs_logit_diff_abs_delta_between_notebooks",
        ascending=False,
    ).head(25)

    largest_manual_max_disagreements = comparison_df.sort_values(
        "manual_max_abs_logit_diff_abs_delta_between_notebooks",
        ascending=False,
    ).head(25)

    largest_mean_csv = out_dir / "largest_manual_mean_abs_logit_diff_disagreements.csv"
    largest_max_csv = out_dir / "largest_manual_max_abs_logit_diff_disagreements.csv"

    largest_manual_mean_disagreements.to_csv(largest_mean_csv, index=False)
    largest_manual_max_disagreements.to_csv(largest_max_csv, index=False)

    print("Manual quantization sanity summary")
    print(manual_sanity_summary.to_string(index=False))
    print()
    print(f"Saved: {manual_sanity_csv}")
    print(f"Saved: {largest_mean_csv}")
    print(f"Saved: {largest_max_csv}")
    print()

    return manual_sanity_summary


def add_backend_comparison_columns(comparison_df):
    comparison_df["bnb_minus_torchao_mean_abs_logit_diff"] = (
        comparison_df["bnb_mean_abs_logit_diff"]
        - comparison_df["torchao_mean_abs_logit_diff"]
    )

    comparison_df["bnb_minus_torchao_max_abs_logit_diff"] = (
        comparison_df["bnb_max_abs_logit_diff"]
        - comparison_df["torchao_max_abs_logit_diff"]
    )

    comparison_df["bnb_over_torchao_mean_abs_logit_diff"] = (
        comparison_df["bnb_mean_abs_logit_diff"]
        / comparison_df["torchao_mean_abs_logit_diff"].replace(0, np.nan)
    )

    comparison_df["bnb_over_torchao_max_abs_logit_diff"] = (
        comparison_df["bnb_max_abs_logit_diff"]
        / comparison_df["torchao_max_abs_logit_diff"].replace(0, np.nan)
    )

    comparison_df["backend_top1_same"] = (
        comparison_df["bnb_top1"] == comparison_df["torchao_top1"]
    )

    comparison_df["backend_top1_match_same"] = (
        comparison_df["bnb_top1_match"] == comparison_df["torchao_top1_match"]
    )

    return comparison_df


def summarize_backend_comparison(comparison_df, out_dir):
    backend_summary = (
        comparison_df.groupby(["block_index", "sub_layer"])
        .agg(
            n_rows=("text_index", "count"),

            torchao_mean_abs_logit_diff_mean=("torchao_mean_abs_logit_diff", "mean"),
            bnb_mean_abs_logit_diff_mean=("bnb_mean_abs_logit_diff", "mean"),
            bnb_minus_torchao_mean_abs_logit_diff_mean=(
                "bnb_minus_torchao_mean_abs_logit_diff",
                "mean",
            ),
            bnb_over_torchao_mean_abs_logit_diff_mean=(
                "bnb_over_torchao_mean_abs_logit_diff",
                "mean",
            ),

            torchao_max_abs_logit_diff_mean=("torchao_max_abs_logit_diff", "mean"),
            bnb_max_abs_logit_diff_mean=("bnb_max_abs_logit_diff", "mean"),
            bnb_minus_torchao_max_abs_logit_diff_mean=(
                "bnb_minus_torchao_max_abs_logit_diff",
                "mean",
            ),
            bnb_over_torchao_max_abs_logit_diff_mean=(
                "bnb_over_torchao_max_abs_logit_diff",
                "mean",
            ),

            torchao_top1_match_rate=("torchao_top1_match", "mean"),
            bnb_top1_match_rate=("bnb_top1_match", "mean"),
            backend_top1_same_rate=("backend_top1_same", "mean"),
            backend_top1_match_same_rate=("backend_top1_match_same", "mean"),
        )
        .reset_index()
    )

    backend_summary_csv = out_dir / "torchao_vs_bitsandbytes_summary_by_sublayer.csv"
    backend_summary.to_csv(backend_summary_csv, index=False)

    largest_bnb_over_torchao_mean = comparison_df.sort_values(
        "bnb_minus_torchao_mean_abs_logit_diff",
        ascending=False,
    ).head(25)

    largest_bnb_over_torchao_max = comparison_df.sort_values(
        "bnb_minus_torchao_max_abs_logit_diff",
        ascending=False,
    ).head(25)

    largest_mean_csv = out_dir / "largest_bnb_minus_torchao_mean_abs_logit_diff.csv"
    largest_max_csv = out_dir / "largest_bnb_minus_torchao_max_abs_logit_diff.csv"

    largest_bnb_over_torchao_mean.to_csv(largest_mean_csv, index=False)
    largest_bnb_over_torchao_max.to_csv(largest_max_csv, index=False)

    print("TorchAO vs BitsAndBytes summary")
    print(backend_summary.to_string(index=False))
    print()
    print(f"Saved: {backend_summary_csv}")
    print(f"Saved: {largest_mean_csv}")
    print(f"Saved: {largest_max_csv}")
    print()

    return backend_summary


def save_full_comparison(comparison_df, out_dir):
    comparison_csv = out_dir / "torchao_vs_bitsandbytes_rowwise_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Saved full rowwise comparison: {comparison_csv}")
    print()


def save_overall_summary(comparison_df, out_dir):
    overall = pd.DataFrame(
        [
            {
                "n_rows": len(comparison_df),
                "n_sublayers": comparison_df["sub_layer"].nunique(),
                "n_texts": comparison_df["text_index"].nunique(),

                "manual_top1_same_rate_between_notebooks": (
                    comparison_df["manual_top1_same_between_notebooks"].mean()
                ),
                "manual_mean_abs_logit_diff_abs_delta_mean_between_notebooks": (
                    comparison_df["manual_mean_abs_logit_diff_abs_delta_between_notebooks"].mean()
                ),
                "manual_mean_abs_logit_diff_abs_delta_max_between_notebooks": (
                    comparison_df["manual_mean_abs_logit_diff_abs_delta_between_notebooks"].max()
                ),

                "torchao_mean_abs_logit_diff_mean": (
                    comparison_df["torchao_mean_abs_logit_diff"].mean()
                ),
                "bnb_mean_abs_logit_diff_mean": (
                    comparison_df["bnb_mean_abs_logit_diff"].mean()
                ),
                "bnb_minus_torchao_mean_abs_logit_diff_mean": (
                    comparison_df["bnb_minus_torchao_mean_abs_logit_diff"].mean()
                ),
                "bnb_over_torchao_mean_abs_logit_diff_mean": (
                    comparison_df["bnb_over_torchao_mean_abs_logit_diff"].mean()
                ),

                "torchao_max_abs_logit_diff_mean": (
                    comparison_df["torchao_max_abs_logit_diff"].mean()
                ),
                "bnb_max_abs_logit_diff_mean": (
                    comparison_df["bnb_max_abs_logit_diff"].mean()
                ),
                "bnb_minus_torchao_max_abs_logit_diff_mean": (
                    comparison_df["bnb_minus_torchao_max_abs_logit_diff"].mean()
                ),
                "bnb_over_torchao_max_abs_logit_diff_mean": (
                    comparison_df["bnb_over_torchao_max_abs_logit_diff"].mean()
                ),

                "torchao_top1_match_rate": comparison_df["torchao_top1_match"].mean(),
                "bnb_top1_match_rate": comparison_df["bnb_top1_match"].mean(),
                "backend_top1_same_rate": comparison_df["backend_top1_same"].mean(),
            }
        ]
    )

    overall_csv = out_dir / "overall_comparison_summary.csv"
    overall.to_csv(overall_csv, index=False)

    print("Overall summary")
    print(overall.to_string(index=False))
    print()
    print(f"Saved: {overall_csv}")
    print()


def make_scatter_plot(
    df,
    x,
    y,
    xlabel,
    ylabel,
    title,
    out_path,
):
    plt.figure(figsize=(9, 7))

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="sub_layer",
        style="sub_layer",
        s=80,
        alpha=0.85,
    )

    max_val = max(df[x].max(), df[y].max())

    plt.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="y = x")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def make_plots(comparison_df, out_dir):
    sns.set_theme(style="whitegrid")

    make_scatter_plot(
        df=comparison_df,
        x="manual_mean_abs_logit_diff_from_torchao_notebook",
        y="manual_mean_abs_logit_diff_from_bnb_notebook",
        xlabel="Manual mean abs logit diff from TorchAO notebook",
        ylabel="Manual mean abs logit diff from BitsAndBytes notebook",
        title="Manual Quantization Sanity Check\nMean absolute [MASK] logit drift",
        out_path=out_dir / "scatter_manual_sanity_mean_abs_logit_diff.png",
    )

    make_scatter_plot(
        df=comparison_df,
        x="manual_max_abs_logit_diff_from_torchao_notebook",
        y="manual_max_abs_logit_diff_from_bnb_notebook",
        xlabel="Manual max abs logit diff from TorchAO notebook",
        ylabel="Manual max abs logit diff from BitsAndBytes notebook",
        title="Manual Quantization Sanity Check\nMax absolute [MASK] logit drift",
        out_path=out_dir / "scatter_manual_sanity_max_abs_logit_diff.png",
    )

    make_scatter_plot(
        df=comparison_df,
        x="torchao_mean_abs_logit_diff",
        y="bnb_mean_abs_logit_diff",
        xlabel="TorchAO mean abs logit diff",
        ylabel="BitsAndBytes mean abs logit diff",
        title="BitsAndBytes vs TorchAO INT8 Quantization\nMean absolute [MASK] logit drift",
        out_path=out_dir / "scatter_bnb_vs_torchao_mean_abs_logit_diff.png",
    )

    make_scatter_plot(
        df=comparison_df,
        x="torchao_max_abs_logit_diff",
        y="bnb_max_abs_logit_diff",
        xlabel="TorchAO max abs logit diff",
        ylabel="BitsAndBytes max abs logit diff",
        title="BitsAndBytes vs TorchAO INT8 Quantization\nMax absolute [MASK] logit drift",
        out_path=out_dir / "scatter_bnb_vs_torchao_max_abs_logit_diff.png",
    )

    print()


def main():
    args = parse_args()
    out_dir = get_out_dir(args)

    print("Compare TorchAO and BitsAndBytes")
    print("Output directory:", out_dir)
    print()

    torchao_df, bnb_df = load_results(args.torchao_csv, args.bnb_csv)

    save_input_summaries(torchao_df, bnb_df, out_dir)

    comparison_df = merge_results(torchao_df, bnb_df, out_dir)

    comparison_df = add_baseline_sanity_columns(comparison_df)
    comparison_df = add_manual_sanity_columns(comparison_df)
    comparison_df = add_backend_comparison_columns(comparison_df)

    summarize_baseline_sanity(comparison_df, out_dir)
    summarize_manual_sanity(comparison_df, out_dir)

    validate_sanity_checks(comparison_df)

    summarize_backend_comparison(comparison_df, out_dir)
    save_overall_summary(comparison_df, out_dir)
    save_full_comparison(comparison_df, out_dir)
    make_plots(comparison_df, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()