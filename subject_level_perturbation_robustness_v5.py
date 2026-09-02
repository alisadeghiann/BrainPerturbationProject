from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "statistical_v2"
    / "perturbation_statistical_results_v2.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "robustness_v5"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = (
    OUTPUT_DIR
    / "subject_level_perturbation_robustness_v5.csv"
)

QC_OUTPUT = (
    OUTPUT_DIR
    / "subject_level_perturbation_robustness_v5_qc.csv"
)

print("=" * 80)
print("SUBJECT-LEVEL PERTURBATION / WHAT-IF ROBUSTNESS V5")
print("=" * 80)

# ============================================================
# LOAD
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

print(f"Input rows:    {len(df)}")
print(f"Input columns: {len(df.columns)}")

# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "target",
    "feature",
    "mean_difference",
    "cohen_d",
    "p_value",
    "p_fdr",
    "direction"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

# ============================================================
# IMPORTANT:
# V2 statistical file contains aggregate feature effects.
# There is no raw subject-level observation column in this file.
#
# Therefore subject_effect_* cannot be reconstructed from V2.
# We explicitly mark them as unavailable rather than inventing data.
# ============================================================

print()
print("=" * 80)
print("SUBJECT-LEVEL DATA AVAILABILITY")
print("=" * 80)

subject_candidates = [
    c for c in df.columns
    if "subject" in c.lower()
]

print("Subject-related columns found:")

if subject_candidates:
    for c in subject_candidates:
        print(f"  {c}")
else:
    print("  NONE")

# ============================================================
# ROBUSTNESS METRICS THAT CAN ACTUALLY BE CALCULATED
# ============================================================

result = df.copy()

# Number of subjects reported by the previous robustness pipeline
if "subject_count" not in result.columns:
    result["subject_count"] = np.nan

# Direction consistency
if "positive_fraction" not in result.columns:
    result["positive_fraction"] = np.nan

if "negative_fraction" not in result.columns:
    result["negative_fraction"] = np.nan

if "direction_consistency" not in result.columns:
    result["direction_consistency"] = np.nan

# ============================================================
# EXPLICITLY UNAVAILABLE SUBJECT EFFECT METRICS
# ============================================================

result["subject_effect_mean"] = np.nan
result["subject_effect_median"] = np.nan
result["subject_effect_std"] = np.nan

result["subject_effect_status"] = (
    "NOT_AVAILABLE_RAW_SUBJECT_DATA"
)

# ============================================================
# EFFECT DIRECTION
# ============================================================

result["statistical_direction"] = np.where(
    result["mean_difference"] > 0,
    "positive",
    np.where(
        result["mean_difference"] < 0,
        "negative",
        "zero"
    )
)

# ============================================================
# EFFECT MAGNITUDE
# ============================================================

abs_d = result["cohen_d"].abs()

result["effect_magnitude"] = np.select(
    [
        abs_d >= 0.80,
        abs_d >= 0.50,
        abs_d >= 0.20,
    ],
    [
        "large",
        "moderate",
        "small",
    ],
    default="negligible"
)

# ============================================================
# FDR SIGNIFICANCE
# ============================================================

result["fdr_significant"] = (
    pd.to_numeric(
        result["p_fdr"],
        errors="coerce"
    ) < 0.05
)

# ============================================================
# DIRECTION AGREEMENT
# ============================================================

def direction_agreement(row):

    statistical = row["statistical_direction"]

    positive = row["positive_fraction"]
    negative = row["negative_fraction"]

    if pd.isna(positive) or pd.isna(negative):
        return "not_available"

    if statistical == "positive":

        if positive >= 0.70:
            return "strong_agreement"

        if positive >= 0.55:
            return "moderate_agreement"

        return "weak_or_unresolved"

    if statistical == "negative":

        if negative >= 0.70:
            return "strong_agreement"

        if negative >= 0.55:
            return "moderate_agreement"

        return "weak_or_unresolved"

    return "not_available"


result["direction_agreement"] = result.apply(
    direction_agreement,
    axis=1
)

# ============================================================
# ROBUSTNESS CLASSIFICATION
# ============================================================

def classify_robustness(row):

    if not row["fdr_significant"]:
        return "not_significant"

    agreement = row["direction_agreement"]

    if agreement == "strong_agreement":
        return "robust"

    if agreement == "moderate_agreement":
        return "moderate_robustness"

    if agreement == "weak_or_unresolved":
        return "weak_or_unresolved"

    return "statistically_significant_but_subject_robustness_unavailable"


result["robustness_class"] = result.apply(
    classify_robustness,
    axis=1
)

# ============================================================
# SCIENTIFIC PRIORITY SCORE
# ============================================================

# Larger effect + stronger statistical evidence
# + stronger directional agreement = higher priority.

safe_p = pd.to_numeric(
    result["p_fdr"],
    errors="coerce"
).clip(lower=1e-300)

effect_score = result["cohen_d"].abs()

significance_score = -np.log10(safe_p)

consistency_score = result[
    "direction_consistency"
].fillna(0)

result["scientific_priority_score"] = (
    effect_score
    * significance_score
    * consistency_score
)

# ============================================================
# RANK
# ============================================================

result = result.sort_values(
    [
        "target",
        "scientific_priority_score"
    ],
    ascending=[True, False]
).reset_index(drop=True)

result["scientific_rank"] = (
    result
    .groupby("target")
    .cumcount()
    + 1
)

# ============================================================
# FINAL COLUMN ORDER
# ============================================================

preferred_columns = [
    "target",
    "feature",
    "mean_difference",
    "cohen_d",
    "abs_cohen_d",
    "p_value",
    "p_fdr",
    "fdr_significant",
    "effect_magnitude",
    "statistical_direction",
    "subject_count",
    "subject_effect_mean",
    "subject_effect_median",
    "subject_effect_std",
    "subject_effect_status",
    "positive_fraction",
    "negative_fraction",
    "direction_consistency",
    "direction_agreement",
    "robustness_class",
    "scientific_priority_score",
    "scientific_rank"
]

# Create abs_cohen_d if missing
if "abs_cohen_d" not in result.columns:
    result["abs_cohen_d"] = result["cohen_d"].abs()

final_columns = [
    c for c in preferred_columns
    if c in result.columns
]

remaining = [
    c for c in result.columns
    if c not in final_columns
]

result = result[
    final_columns + remaining
]

# ============================================================
# QC
# ============================================================

numeric_cols = result.select_dtypes(
    include=[np.number]
).columns

nan_numeric = int(
    result[numeric_cols].isna().sum().sum()
)

inf_numeric = int(
    np.isinf(
        result[numeric_cols].to_numpy(
            dtype=float
        )
    ).sum()
)

duplicate_keys = int(
    result.duplicated(
        subset=["target", "feature"]
    ).sum()
)

print()
print("=" * 80)
print("FINAL ROBUSTNESS QC")
print("=" * 80)

print(f"Targets:                  {result['target'].nunique()}")
print(f"Features:                 {result['feature'].nunique()}")
print(f"Result rows:              {len(result)}")
print(
    f"FDR significant:          "
    f"{result['fdr_significant'].sum()}"
)
print(
    f"Subject robustness data:  "
    f"NOT AVAILABLE"
)
print(f"NaN numeric values:       {nan_numeric}")
print(f"Inf numeric values:       {inf_numeric}")
print(f"Duplicate target-feature: {duplicate_keys}")

# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT,
    index=False
)

qc = pd.DataFrame([{
    "targets": result["target"].nunique(),
    "features": result["feature"].nunique(),
    "result_rows": len(result),
    "fdr_significant": int(
        result["fdr_significant"].sum()
    ),
    "subject_robustness_available": False,
    "subject_effect_mean_valid": int(
        result["subject_effect_mean"].notna().sum()
    ),
    "subject_effect_median_valid": int(
        result["subject_effect_median"].notna().sum()
    ),
    "subject_effect_std_valid": int(
        result["subject_effect_std"].notna().sum()
    ),
    "nan_numeric": nan_numeric,
    "inf_numeric": inf_numeric,
    "duplicate_target_feature": duplicate_keys
}])

qc.to_csv(
    QC_OUTPUT,
    index=False
)

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT)
print(QC_OUTPUT)

print()
print("=" * 80)
print("STATUS: PASS - ROBUSTNESS V5 CREATED")
print("=" * 80)