import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_evidence_reconciliation_v1"
    / "final_evidence_reconciliation_v1.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "powerbi_dashboard_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "powerbi_eeg_evidence_dataset_v1.csv"

print("=" * 90)
print("CREATING POWER BI DASHBOARD DATASET V1")
print("=" * 90)

df = pd.read_csv(INPUT)

print(f"Input rows:    {len(df)}")
print(f"Input columns: {len(df.columns)}")

# ------------------------------------------------------------
# HUMAN-READABLE FEATURE GROUP
# ------------------------------------------------------------

def feature_group(feature):
    f = feature.lower()

    if "frontoparietal" in f:
        return "Regional Difference / Ratio"

    if "ratio" in f:
        return "Frequency Ratio"

    if f.endswith("_rel"):
        return "Relative Band Power"

    if f.endswith("_abs"):
        return "Absolute Band Power"

    regions = [
        "frontal",
        "central",
        "parietal",
        "occipital",
        "temporal"
    ]

    if any(r in f for r in regions):
        return "Regional Band Power"

    return "Other"

df["feature_group"] = df["feature"].apply(feature_group)

# ------------------------------------------------------------
# FREQUENCY BAND
# ------------------------------------------------------------

def frequency_band(feature):
    f = feature.lower()

    if "delta" in f:
        return "Delta"
    if "theta" in f:
        return "Theta"
    if "alpha" in f:
        return "Alpha"
    if "beta" in f:
        return "Beta"
    if "gamma" in f:
        return "Gamma"

    return "Mixed / Multi-band"

df["frequency_band"] = df["feature"].apply(frequency_band)

# ------------------------------------------------------------
# REGION
# ------------------------------------------------------------

def region_label(feature):
    f = feature.lower()

    if "frontoparietal" in f:
        return "Frontoparietal"

    for region in [
        "frontal",
        "central",
        "parietal",
        "occipital",
        "temporal"
    ]:
        if region in f:
            return region.capitalize()

    return "Global / Mixed"

df["region"] = df["feature"].apply(region_label)

# ------------------------------------------------------------
# SIMPLE SIGNIFICANCE FLAGS
# ------------------------------------------------------------

df["perturbation_significant_flag"] = (
    df["perturbation_significant"] > 0
).astype(int)

df["r_fdr_significant_flag"] = (
    df["r_fdr_significant"] > 0
).astype(int)

df["null_significant_flag"] = (
    df["null_significant"] > 0
).astype(int)

# ------------------------------------------------------------
# LOSO STABILITY RATE
# ------------------------------------------------------------

df["loso_stability_rate"] = (
    df["loso_stable_count"] /
    df["loso_rows"].replace(0, pd.NA)
).fillna(0)

# ------------------------------------------------------------
# EVIDENCE LEVEL
# ------------------------------------------------------------

def evidence_level(score):
    if score >= 0.70:
        return "High"
    elif score >= 0.50:
        return "Moderate"
    elif score >= 0.30:
        return "Low"
    else:
        return "Very Low"

df["evidence_level"] = df["final_evidence_score"].apply(
    evidence_level
)

# ------------------------------------------------------------
# TOP FEATURE FLAG
# ------------------------------------------------------------

df["top_10_flag"] = (
    df["final_rank"] <= 10
).astype(int)

df["top_20_flag"] = (
    df["final_rank"] <= 20
).astype(int)

# ------------------------------------------------------------
# FINAL COLUMN ORDER
# ------------------------------------------------------------

columns = [
    "final_rank",
    "feature",
    "feature_group",
    "frequency_band",
    "region",

    "final_evidence_score",
    "final_evidence_class",
    "evidence_level",

    "perturbation_significant",
    "perturbation_min_p",
    "perturbation_max_abs_d",

    "cross_target_pattern",

    "ml_mean_importance",
    "ml_max_importance",
    "ml_mean_sign_consistency",

    "loso_stable_count",
    "loso_rows",
    "loso_stability_rate",
    "loso_mean_effect",

    "null_significant",
    "null_min_p",

    "r_fdr_significant",
    "r_min_p",

    "score_perturbation",
    "score_ml",
    "score_loso",
    "score_consistency",
    "score_r",
    "score_null",

    "top_10_flag",
    "top_20_flag"
]

df = df[columns]

# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

df = df.sort_values(
    "final_rank",
    ascending=True
).reset_index(drop=True)

# ------------------------------------------------------------
# QC
# ------------------------------------------------------------

numeric_cols = df.select_dtypes(include="number").columns

nan_count = int(df[numeric_cols].isna().sum().sum())
inf_count = int(
    df[numeric_cols]
    .replace([float("inf"), float("-inf")], pd.NA)
    .isna()
    .sum()
    .sum()
)

duplicate_features = int(
    df["feature"].duplicated().sum()
)

print()
print("=" * 90)
print("POWER BI DATASET QC")
print("=" * 90)

print(f"Rows:                 {len(df)}")
print(f"Columns:              {len(df.columns)}")
print(f"Unique features:      {df['feature'].nunique()}")
print(f"NaN numeric cells:    {nan_count}")
print(f"Inf numeric cells:    {inf_count}")
print(f"Duplicate features:   {duplicate_features}")

if nan_count != 0:
    raise ValueError("NaN values detected.")

if inf_count != 0:
    raise ValueError("Inf values detected.")

if duplicate_features != 0:
    raise ValueError("Duplicate feature rows detected.")

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 90)
print("SAVED")
print("=" * 90)

print(OUTPUT)

print()
print("=" * 90)
print("POWER BI DASHBOARD DATASET V1 COMPLETE")
print("=" * 90)
print("STATUS: PASS - DASHBOARD-READY DATASET CREATED")
