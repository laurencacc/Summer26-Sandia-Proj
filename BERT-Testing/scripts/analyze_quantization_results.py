import pandas as pd

INPUT_TSV = "quantization_results.tsv"

df = pd.read_csv(INPUT_TSV, sep="\t")


def frac_to_num(value):
    """
    Converts strings like '5/5' to 5.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if "/" not in value:
        return None

    return int(value.split("/")[0])


# Convert overlap columns to numeric scores
df["Top-5 Set Overlap Num"] = df["Top-5 Set Overlap"].apply(frac_to_num)
df["Top-5 Rank Match Num"] = df["Top-5 Rank Match"].apply(frac_to_num)

# Convert Top-1 to numeric
df["Top-1 Match Num"] = df["Top-1 Match"].astype(str).str.upper().map(
    {"TRUE": 1, "FALSE": 0}
)

# Ensure key metric columns are numeric
numeric_cols = [
    "Max Logit Diff",
    "Mean Logit Diff",
    "Cosine Sim",
    "Local Rel Σ|q-b|",
    "Final Rel Σ|q-b|",
    "Final Rel |Σq-Σb|",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


print("\n=== Overall correlation with overlap ===")
corr_cols = [
    "Top-1 Match Num",
    "Top-5 Set Overlap Num",
    "Top-5 Rank Match Num",
    "Max Logit Diff",
    "Mean Logit Diff",
    "Cosine Sim",
    "Local Rel Σ|q-b|",
    "Final Rel Σ|q-b|",
    "Final Rel |Σq-Σb|",
]

print(df[corr_cols].corr(method="spearman"))


print("\n=== Average metrics grouped by Top-5 Set Overlap ===")
grouped_set = df.groupby("Top-5 Set Overlap").agg(
    count=("Sublayer", "count"),
    mean_final_rel_error=("Final Rel Σ|q-b|", "mean"),
    median_final_rel_error=("Final Rel Σ|q-b|", "median"),
    min_final_rel_error=("Final Rel Σ|q-b|", "min"),
    max_final_rel_error=("Final Rel Σ|q-b|", "max"),
    mean_max_logit_diff=("Max Logit Diff", "mean"),
    mean_cosine=("Cosine Sim", "mean"),
)

print(grouped_set.sort_index(ascending=False))


print("\n=== Average metrics grouped by Top-5 Rank Match ===")
grouped_rank = df.groupby("Top-5 Rank Match").agg(
    count=("Sublayer", "count"),
    mean_final_rel_error=("Final Rel Σ|q-b|", "mean"),
    median_final_rel_error=("Final Rel Σ|q-b|", "median"),
    min_final_rel_error=("Final Rel Σ|q-b|", "min"),
    max_final_rel_error=("Final Rel Σ|q-b|", "max"),
    mean_max_logit_diff=("Max Logit Diff", "mean"),
    mean_cosine=("Cosine Sim", "mean"),
)

print(grouped_rank.sort_index(ascending=False))


print("\n=== Average metrics grouped by Top-1 Match ===")
grouped_top1 = df.groupby("Top-1 Match").agg(
    count=("Sublayer", "count"),
    mean_final_rel_error=("Final Rel Σ|q-b|", "mean"),
    median_final_rel_error=("Final Rel Σ|q-b|", "median"),
    min_final_rel_error=("Final Rel Σ|q-b|", "min"),
    max_final_rel_error=("Final Rel Σ|q-b|", "max"),
    mean_max_logit_diff=("Max Logit Diff", "mean"),
    mean_cosine=("Cosine Sim", "mean"),
)

print(grouped_top1)


print("\n=== Worst rows by Final Rel Σ|q-b| ===")
cols_to_show = [
    "Quantized precision",
    "Sentence",
    "Block",
    "Sublayer",
    "Top-1 Match",
    "Top-5 Set Overlap",
    "Top-5 Rank Match",
    "Max Logit Diff",
    "Mean Logit Diff",
    "Cosine Sim",
    "Final Rel Σ|q-b|",
]

print(
    df.sort_values("Final Rel Σ|q-b|", ascending=False)[cols_to_show].head(20)
)


print("\n=== Rows where Top-1 changed ===")
print(
    df[df["Top-1 Match"].astype(str).str.upper() == "FALSE"][cols_to_show]
    .sort_values("Final Rel Σ|q-b|", ascending=False)
)


print("\n=== Rows with high error but 5/5 set overlap ===")
high_error_5of5 = df[
    (df["Top-5 Set Overlap"] == "5/5")
    & (df["Final Rel Σ|q-b|"] > 0.05)
]

print(
    high_error_5of5[cols_to_show]
    .sort_values("Final Rel Σ|q-b|", ascending=False)
)


print("\n=== Rows with low error but less than 5/5 overlap ===")
low_error_bad_overlap = df[
    (df["Top-5 Set Overlap"] != "5/5")
    & (df["Final Rel Σ|q-b|"] < 0.03)
]

print(
    low_error_bad_overlap[cols_to_show]
    .sort_values("Final Rel Σ|q-b|", ascending=True)
)