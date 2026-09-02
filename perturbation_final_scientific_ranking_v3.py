# =============================================================================
# FINAL SCIENTIFIC PERTURBATION / WHAT-IF RANKING V3
# =============================================================================
#
# Purpose:
#   Integrate:
#     1. Statistical perturbation results
#     2. Subject-level robustness results
#     3. Effect sizes
#     4. FDR significance
#     5. Direction consistency
#
# Important:
#   - No ML model is trained here.
#   - No target-derived predictor is introduced.
#   - This is a scientific interpretation/ranking stage.
#   - REMEMBER and CORRECT are analyzed separately.
#
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

STAT_FILE = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "statistical_v2"
    / "perturbation_statistical_results_v2.csv"
)

ROBUST_FILE = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "robustness_v3"
    / "subject_level_perturbation_robustness_v3.csv"
)

OUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_ranking_v3"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RANKING = OUT_DIR / "final_scientific_perturbation_ranking_v3.csv"
OUT_SUMMARY = OUT_DIR / "final_scientific_perturbation_summary_v3.csv"
OUT_QC = OUT_DIR / "final_scientific_perturbation_ranking_qc_v3.csv"


# =============================================================================
# HELPERS
# =============================================================================

def find_column(df, candidates):
    """
    Find the first available column from a list of possible names.
    """
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def numeric_series(df, column):
    """
    Safely convert a column to numeric.
    """
    if column is None:
        return pd.Series(np.nan, index=df.index)

    return pd.to_numeric(df[column], errors="coerce")


def safe_mean(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


# =============================================================================
# HEADER
# =============================================================================

print("=" * 88)
print("FINAL SCIENTIFIC PERTURBATION / WHAT-IF RANKING V3")
print("=" * 88)

print(f"Statistical input: {STAT_FILE}")
print(f"Robustness input:  {ROBUST_FILE}")


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

if not STAT_FILE.exists():
    raise FileNotFoundError(
        f"Statistical result file not found:\n{STAT_FILE}"
    )

if not ROBUST_FILE.exists():
    raise FileNotFoundError(
        f"Subject-level robustness file not found:\n{ROBUST_FILE}"
    )


# =============================================================================
# LOAD
# =============================================================================

stat = pd.read_csv(STAT_FILE)
robust = pd.read_csv(ROBUST_FILE)

print("\n" + "=" * 88)
print("INPUT VALIDATION")
print("=" * 88)

print(f"Statistical rows: {len(stat):,}")
print(f"Statistical columns: {len(stat.columns):,}")

print(f"Robustness rows: {len(robust):,}")
print(f"Robustness columns: {len(robust.columns):,}")


# =============================================================================
# IDENTIFY IMPORTANT COLUMNS
# =============================================================================

stat_target_col = find_column(
    stat,
    ["target", "target_name", "outcome"]
)

stat_feature_col = find_column(
    stat,
    ["feature", "feature_name", "predictor"]
)

stat_p_col = find_column(
    stat,
    ["p_value", "pvalue", "p"]
)

stat_fdr_col = find_column(
    stat,
    ["p_fdr", "fdr", "q_value", "qvalue", "adjusted_p"]
)

stat_effect_col = find_column(
    stat,
    ["cohen_d", "effect_size", "effect", "standardized_effect"]
)

stat_diff_col = find_column(
    stat,
    ["mean_difference", "mean_diff", "difference", "delta"]
)

stat_direction_col = find_column(
    stat,
    ["direction", "effect_direction"]
)

robust_target_col = find_column(
    robust,
    ["target", "target_name", "outcome"]
)

robust_feature_col = find_column(
    robust,
    ["feature", "feature_name", "predictor"]
)

print("\n" + "=" * 88)
print("COLUMN DETECTION")
print("=" * 88)

print(f"Stat target column:      {stat_target_col}")
print(f"Stat feature column:     {stat_feature_col}")
print(f"Stat p-value column:     {stat_p_col}")
print(f"Stat FDR column:         {stat_fdr_col}")
print(f"Stat effect column:      {stat_effect_col}")
print(f"Stat difference column:  {stat_diff_col}")
print(f"Stat direction column:   {stat_direction_col}")

print(f"Robust target column:    {robust_target_col}")
print(f"Robust feature column:   {robust_feature_col}")


# =============================================================================
# REQUIRED COLUMN CHECK
# =============================================================================

if stat_target_col is None:
    raise RuntimeError("Could not identify statistical target column.")

if stat_feature_col is None:
    raise RuntimeError("Could not identify statistical feature column.")

if stat_effect_col is None:
    raise RuntimeError("Could not identify Cohen's d / effect-size column.")

if stat_fdr_col is None:
    raise RuntimeError("Could not identify FDR column.")

if robust_target_col is None:
    raise RuntimeError("Could not identify robustness target column.")

if robust_feature_col is None:
    raise RuntimeError("Could not identify robustness feature column.")


# =============================================================================
# STANDARDIZE STATISTICAL TABLE
# =============================================================================

stat_work = pd.DataFrame()

stat_work["target"] = stat[stat_target_col].astype(str)
stat_work["feature"] = stat[stat_feature_col].astype(str)

stat_work["p_value"] = numeric_series(stat, stat_p_col)
stat_work["p_fdr"] = numeric_series(stat, stat_fdr_col)
stat_work["effect_size"] = numeric_series(stat, stat_effect_col)

if stat_diff_col is not None:
    stat_work["mean_difference"] = numeric_series(stat, stat_diff_col)
else:
    stat_work["mean_difference"] = np.nan

if stat_direction_col is not None:
    stat_work["direction"] = stat[stat_direction_col].astype(str)
else:
    stat_work["direction"] = np.where(
        stat_work["effect_size"] > 0,
        "positive",
        np.where(
            stat_work["effect_size"] < 0,
            "negative",
            "zero"
        )
    )


# =============================================================================
# STANDARDIZE ROBUSTNESS TABLE
# =============================================================================

rob_work = pd.DataFrame()

rob_work["target"] = robust[robust_target_col].astype(str)
rob_work["feature"] = robust[robust_feature_col].astype(str)


# =============================================================================
# DISCOVER ROBUSTNESS COLUMNS
# =============================================================================

rob_numeric_candidates = [
    "n_subjects",
    "subjects",
    "subject_count",
    "mean_effect",
    "mean_difference",
    "median_effect",
    "direction_consistency",
    "positive_fraction",
    "negative_fraction",
    "fraction_positive",
    "fraction_negative",
    "sign_consistency",
    "consistency",
    "stability"
]

for col in rob_numeric_candidates:
    detected = find_column(robust, [col])

    if detected is not None:
        rob_work[col] = numeric_series(robust, detected)


# =============================================================================
# DIRECTION CONSISTENCY DETECTION
# =============================================================================

consistency_col = None

for candidate in [
    "direction_consistency",
    "sign_consistency",
    "consistency",
    "stability",
    "fraction_positive",
    "positive_fraction"
]:
    detected = find_column(robust, [candidate])

    if detected is not None:
        consistency_col = detected
        break


positive_col = find_column(
    robust,
    ["positive_fraction", "fraction_positive", "positive_consistency"]
)

negative_col = find_column(
    robust,
    ["negative_fraction", "fraction_negative", "negative_consistency"]
)

subject_count_col = find_column(
    robust,
    ["n_subjects", "subjects", "subject_count"]
)


# =============================================================================
# BUILD ROBUSTNESS SUMMARY
# =============================================================================

rob_summary = (
    rob_work
    .groupby(["target", "feature"], as_index=False)
    .agg({
        col: "mean"
        for col in rob_work.columns
        if col not in ["target", "feature"]
    })
)

if len(rob_summary) == 0:
    rob_summary = (
        rob_work[["target", "feature"]]
        .drop_duplicates()
        .copy()
    )


# =============================================================================
# ADD EXPLICIT CONSISTENCY METRICS
# =============================================================================

if positive_col is not None and negative_col is not None:

    positive_values = numeric_series(
        robust,
        positive_col
    )

    negative_values = numeric_series(
        robust,
        negative_col
    )

    temp_consistency = pd.DataFrame({
        "target": robust[robust_target_col].astype(str),
        "feature": robust[robust_feature_col].astype(str),
        "_positive": positive_values,
        "_negative": negative_values
    })

    temp_consistency["_consistency"] = (
        temp_consistency[["_positive", "_negative"]]
        .max(axis=1)
    )

    consistency_summary = (
        temp_consistency
        .groupby(["target", "feature"], as_index=False)["_consistency"]
        .mean()
        .rename(columns={"_consistency": "direction_consistency"})
    )

    rob_summary = rob_summary.merge(
        consistency_summary,
        on=["target", "feature"],
        how="left"
    )

elif consistency_col is not None:

    consistency_values = numeric_series(
        robust,
        consistency_col
    )

    consistency_summary = pd.DataFrame({
        "target": robust[robust_target_col].astype(str),
        "feature": robust[robust_feature_col].astype(str),
        "direction_consistency": consistency_values
    })

    consistency_summary = (
        consistency_summary
        .groupby(["target", "feature"], as_index=False)
        ["direction_consistency"]
        .mean()
    )

    rob_summary = rob_summary.merge(
        consistency_summary,
        on=["target", "feature"],
        how="left"
    )


# =============================================================================
# SUBJECT COUNT
# =============================================================================

if subject_count_col is not None:

    subject_count = pd.DataFrame({
        "target": robust[robust_target_col].astype(str),
        "feature": robust[robust_feature_col].astype(str),
        "n_subjects": numeric_series(
            robust,
            subject_count_col
        )
    })

    subject_count = (
        subject_count
        .groupby(["target", "feature"], as_index=False)
        ["n_subjects"]
        .max()
    )

    rob_summary = rob_summary.merge(
        subject_count,
        on=["target", "feature"],
        how="left"
    )


# =============================================================================
# MERGE STATISTICAL + ROBUSTNESS RESULTS
# =============================================================================

final = stat_work.merge(
    rob_summary,
    on=["target", "feature"],
    how="left",
    suffixes=("", "_robust")
)


# =============================================================================
# CLEAN DUPLICATE COLUMNS
# =============================================================================

duplicate_columns = [
    c for c in final.columns
    if c.endswith("_robust")
]

for col in duplicate_columns:
    base_col = col[:-7]

    if base_col in final.columns:
        final[base_col] = final[base_col].fillna(final[col])

        final.drop(columns=[col], inplace=True)


# =============================================================================
# SIGNIFICANCE
# =============================================================================

final["fdr_significant"] = (
    pd.to_numeric(final["p_fdr"], errors="coerce") < 0.05
)


# =============================================================================
# ABSOLUTE EFFECT
# =============================================================================

final["abs_effect_size"] = (
    pd.to_numeric(
        final["effect_size"],
        errors="coerce"
    )
    .abs()
)


# =============================================================================
# EFFECT MAGNITUDE
# =============================================================================

def classify_effect(value):

    if pd.isna(value):
        return "unknown"

    value = abs(float(value))

    if value < 0.10:
        return "negligible"

    if value < 0.20:
        return "small"

    if value < 0.50:
        return "moderate"

    if value < 0.80:
        return "large"

    return "very_large"


final["effect_magnitude"] = final["effect_size"].apply(
    classify_effect
)


# =============================================================================
# DIRECTION
# =============================================================================

def classify_direction(value):

    if pd.isna(value):
        return "unknown"

    value = float(value)

    if value > 0:
        return "positive"

    if value < 0:
        return "negative"

    return "zero"


final["direction_final"] = final["effect_size"].apply(
    classify_direction
)


# =============================================================================
# ROBUSTNESS CLASSIFICATION
# =============================================================================

def classify_robustness(row):

    consistency = row.get(
        "direction_consistency",
        np.nan
    )

    if pd.isna(consistency):
        return "not_available"

    consistency = float(consistency)

    if consistency >= 0.80:
        return "strong"

    if consistency >= 0.65:
        return "moderate"

    if consistency >= 0.50:
        return "weak"

    return "unstable"


final["robustness_class"] = final.apply(
    classify_robustness,
    axis=1
)


# =============================================================================
# SCIENTIFIC PRIORITY SCORE
# =============================================================================
#
# This is NOT a machine-learning score.
#
# It combines:
#   - statistical significance
#   - effect magnitude
#   - subject-level directional consistency
#
# The purpose is ranking scientific candidates for interpretation.
#
# =============================================================================

def calculate_priority(row):

    effect = row["abs_effect_size"]

    if pd.isna(effect):
        effect_score = 0.0
    else:
        effect_score = min(float(effect) / 0.50, 1.0)

    fdr_score = 1.0 if row["fdr_significant"] else 0.0

    consistency = row.get(
        "direction_consistency",
        np.nan
    )

    if pd.isna(consistency):
        consistency_score = 0.0
    else:
        consistency_score = min(
            max(float(consistency), 0.0),
            1.0
        )

    score = (
        0.45 * effect_score
        + 0.30 * fdr_score
        + 0.25 * consistency_score
    )

    return score


final["scientific_priority_score"] = final.apply(
    calculate_priority,
    axis=1
)


# =============================================================================
# FINAL RANK
# =============================================================================

final = final.sort_values(
    [
        "target",
        "scientific_priority_score",
        "abs_effect_size"
    ],
    ascending=[True, False, False]
).reset_index(drop=True)

final["scientific_rank"] = (
    final.groupby("target")
    .cumcount()
    + 1
)


# =============================================================================
# TARGET SUMMARY
# =============================================================================

summary_rows = []

for target_name, group in final.groupby("target"):

    n_features = group["feature"].nunique()

    n_fdr = int(
        group["fdr_significant"].sum()
    )

    mean_abs_effect = safe_mean(
        group["abs_effect_size"]
    )

    max_abs_effect = (
        group["abs_effect_size"]
        .max()
    )

    strong_robustness = 0

    if "robustness_class" in group.columns:
        strong_robustness = int(
            (group["robustness_class"] == "strong").sum()
        )

    summary_rows.append({
        "target": target_name,
        "features": n_features,
        "fdr_significant_features": n_fdr,
        "strong_robust_features": strong_robustness,
        "mean_abs_effect_size": mean_abs_effect,
        "max_abs_effect_size": max_abs_effect
    })


summary = pd.DataFrame(summary_rows)


# =============================================================================
# TOP FEATURES
# =============================================================================

print("\n" + "=" * 88)
print("TOP SCIENTIFIC PERTURBATION FEATURES")
print("=" * 88)

for target_name in sorted(
    final["target"].unique()
):

    print(f"\nTARGET: {target_name.upper()}")
    print("-" * 88)

    top = final[
        final["target"] == target_name
    ].head(20)

    display_columns = [
        "scientific_rank",
        "feature",
        "effect_size",
        "abs_effect_size",
        "p_fdr",
        "fdr_significant",
        "direction_final",
        "effect_magnitude",
        "direction_consistency",
        "robustness_class",
        "scientific_priority_score"
    ]

    available_display = [
        c for c in display_columns
        if c in top.columns
    ]

    print(
        top[available_display].to_string(
            index=False
        )
    )


# =============================================================================
# FINAL QC
# =============================================================================

numeric_columns = final.select_dtypes(
    include=[np.number]
).columns.tolist()

nan_numeric = int(
    final[numeric_columns]
    .isna()
    .sum()
    .sum()
)

inf_numeric = int(
    np.isinf(
        final[numeric_columns]
        .to_numpy(dtype=float)
    )
    .sum()
)

duplicate_pairs = int(
    final[
        ["target", "feature"]
    ]
    .duplicated()
    .sum()
)

print("\n" + "=" * 88)
print("FINAL SCIENTIFIC RANKING QC")
print("=" * 88)

print(
    f"Targets:                    "
    f"{final['target'].nunique():,}"
)

print(
    f"Features:                   "
    f"{final['feature'].nunique():,}"
)

print(
    f"Statistical result rows:    "
    f"{len(stat_work):,}"
)

print(
    f"Final result rows:          "
    f"{len(final):,}"
)

print(
    f"FDR-significant rows:       "
    f"{int(final['fdr_significant'].sum()):,}"
)

print(
    f"NaN numeric values:         "
    f"{nan_numeric:,}"
)

print(
    f"Inf numeric values:         "
    f"{inf_numeric:,}"
)

print(
    f"Duplicate target-feature:   "
    f"{duplicate_pairs:,}"
)


# =============================================================================
# SAVE
# =============================================================================

final.to_csv(
    OUT_RANKING,
    index=False
)

summary.to_csv(
    OUT_SUMMARY,
    index=False
)


qc = pd.DataFrame([{
    "targets": int(
        final["target"].nunique()
    ),
    "features": int(
        final["feature"].nunique()
    ),
    "statistical_rows": int(
        len(stat_work)
    ),
    "final_rows": int(
        len(final)
    ),
    "fdr_significant_rows": int(
        final["fdr_significant"].sum()
    ),
    "nan_numeric_values": nan_numeric,
    "inf_numeric_values": inf_numeric,
    "duplicate_target_feature_pairs": duplicate_pairs,
    "status": (
        "PASS"
        if (
            nan_numeric == 0
            and inf_numeric == 0
            and duplicate_pairs == 0
        )
        else "REVIEW_REQUIRED"
    )
}])

qc.to_csv(
    OUT_QC,
    index=False
)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n" + "=" * 88)
print("FINAL SCIENTIFIC PERTURBATION / WHAT-IF RANKING V3 COMPLETE")
print("=" * 88)

print("\nResults:")
print(OUT_RANKING)

print("\nSummary:")
print(OUT_SUMMARY)

print("\nQC:")
print(OUT_QC)

print("\n" + "=" * 88)
print("STATUS: PASS - FINAL SCIENTIFIC PERTURBATION RANKING CREATED")
print("=" * 88)