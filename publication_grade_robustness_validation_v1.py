from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# PUBLICATION-GRADE ROBUSTNESS / STABILITY VALIDATION V1
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_scientific_ranking_v2"
    / "final_scientific_evidence_ranking_v2.csv"
)

SUBJECT_EFFECTS = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "perturbation_subject_effects.csv"
)

OUTDIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "publication_grade_robustness_v1"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTDIR / "publication_grade_robustness_v1.csv"
SUMMARY = OUTDIR / "publication_grade_robustness_summary_v1.csv"
QC = OUTDIR / "publication_grade_robustness_qc_v1.csv"

print("=" * 90)
print("PUBLICATION-GRADE ROBUSTNESS / STABILITY VALIDATION V1")
print("=" * 90)

# =============================================================================
# LOAD DATA
# =============================================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Final scientific ranking not found:\n{INPUT}"
    )

if not SUBJECT_EFFECTS.exists():
    raise FileNotFoundError(
        f"Subject-level perturbation effects not found:\n{SUBJECT_EFFECTS}"
    )

df = pd.read_csv(INPUT)
subj = pd.read_csv(SUBJECT_EFFECTS)

print("\nINPUT SUMMARY")
print("-" * 90)
print(f"Scientific ranking rows: {len(df)}")
print(f"Scientific ranking columns: {len(df.columns)}")
print(f"Subject-effect rows: {len(subj)}")

# =============================================================================
# VALIDATE SUBJECT EFFECT DATA
# =============================================================================

required_subject_cols = {
    "subject",
    "feature",
    "remember_effect_size",
    "correct_effect_size",
    "remember_abs_effect",
    "correct_abs_effect",
}

missing = required_subject_cols - set(subj.columns)

if missing:
    raise ValueError(
        f"Missing subject-effect columns: {sorted(missing)}"
    )

print("\nSUBJECT-LEVEL DATA VALIDATION")
print("-" * 90)

subjects = sorted(subj["subject"].dropna().unique())
features = sorted(subj["feature"].dropna().unique())

print(f"Subjects: {len(subjects)}")
print(f"Features: {len(features)}")

# =============================================================================
# NUMERIC CLEANING
# =============================================================================

effect_cols = [
    "remember_effect_size",
    "correct_effect_size",
    "remember_abs_effect",
    "correct_abs_effect",
]

for col in effect_cols:
    subj[col] = pd.to_numeric(subj[col], errors="coerce")

# =============================================================================
# DUPLICATE CHECK
# =============================================================================

duplicates = subj.duplicated(
    subset=["subject", "feature"],
    keep=False
)

duplicate_count = int(duplicates.sum())

print("\nSUBJECT-FEATURE DUPLICATION")
print("-" * 90)
print(f"Duplicate subject-feature rows: {duplicate_count}")

if duplicate_count > 0:
    raise ValueError(
        "Duplicate subject-feature rows detected. "
        "Robustness calculation cannot safely continue."
    )

# =============================================================================
# IDENTIFY STATISTICAL COLUMNS
# =============================================================================

possible_target_col = None

if "target" in df.columns:
    possible_target_col = "target"
elif "target_label" in df.columns:
    possible_target_col = "target_label"

if possible_target_col is None:
    raise ValueError(
        "No target column found in final scientific ranking."
    )

if "feature" not in df.columns:
    raise ValueError(
        "Feature column not found in final scientific ranking."
    )

print("\nTARGET VALIDATION")
print("-" * 90)
print(f"Target column: {possible_target_col}")
print("Targets:")
print(df[possible_target_col].value_counts(dropna=False))

# =============================================================================
# SUBJECT-LEVEL ROBUSTNESS AGGREGATION
# =============================================================================

rows = []

for _, row in df.iterrows():

    target = str(row[possible_target_col])
    feature = str(row["feature"])

    target_lower = target.lower()

    if target_lower == "remember":
        effect_column = "remember_effect_size"
        abs_column = "remember_abs_effect"
    elif target_lower == "correct":
        effect_column = "correct_effect_size"
        abs_column = "correct_abs_effect"
    else:
        continue

    s = subj[subj["feature"] == feature].copy()

    if s.empty:
        rows.append({
            "target": target,
            "feature": feature,
            "n_subjects": 0,
            "direction_consistency": np.nan,
            "positive_fraction": np.nan,
            "negative_fraction": np.nan,
            "median_effect": np.nan,
            "mean_effect": np.nan,
            "std_effect": np.nan,
            "median_abs_effect": np.nan,
            "nonzero_subject_fraction": np.nan,
            "robustness_class": "not_available",
        })
        continue

    effects = pd.to_numeric(
        s[effect_column],
        errors="coerce"
    ).dropna()

    abs_effects = pd.to_numeric(
        s[abs_column],
        errors="coerce"
    ).dropna()

    if len(effects) == 0:
        rows.append({
            "target": target,
            "feature": feature,
            "n_subjects": 0,
            "direction_consistency": np.nan,
            "positive_fraction": np.nan,
            "negative_fraction": np.nan,
            "median_effect": np.nan,
            "mean_effect": np.nan,
            "std_effect": np.nan,
            "median_abs_effect": np.nan,
            "nonzero_subject_fraction": np.nan,
            "robustness_class": "not_available",
        })
        continue

    positive_fraction = float((effects > 0).mean())
    negative_fraction = float((effects < 0).mean())

    dominant_direction_fraction = max(
        positive_fraction,
        negative_fraction
    )

    nonzero_fraction = float((effects != 0).mean())

    if dominant_direction_fraction >= 0.80:
        robustness_class = "strong_directional_robustness"
    elif dominant_direction_fraction >= 0.65:
        robustness_class = "moderate_directional_robustness"
    elif dominant_direction_fraction >= 0.50:
        robustness_class = "weak_directional_robustness"
    else:
        robustness_class = "inconsistent"

    rows.append({
        "target": target,
        "feature": feature,
        "n_subjects": int(len(effects)),
        "direction_consistency": dominant_direction_fraction,
        "positive_fraction": positive_fraction,
        "negative_fraction": negative_fraction,
        "median_effect": float(effects.median()),
        "mean_effect": float(effects.mean()),
        "std_effect": float(effects.std(ddof=1)) if len(effects) > 1 else 0.0,
        "median_abs_effect": float(abs_effects.median()) if len(abs_effects) else np.nan,
        "nonzero_subject_fraction": nonzero_fraction,
        "robustness_class": robustness_class,
    })

robust = pd.DataFrame(rows)

# =============================================================================
# MERGE WITH SCIENTIFIC RANKING
# =============================================================================

merge_cols = ["target", "feature"]

result = df.merge(
    robust,
    on=merge_cols,
    how="left",
    validate="one_to_one"
)

# =============================================================================
# PUBLICATION-GRADE COMPOSITE VALIDATION
# =============================================================================

def get_fdr(row):
    for col in ["p_fdr", "fdr_p", "q_value"]:
        if col in row.index:
            return pd.to_numeric(row[col], errors="coerce")
    return np.nan


def get_cohen_d(row):
    for col in ["cohen_d", "effect_size", "mean_effect_size"]:
        if col in row.index:
            return pd.to_numeric(row[col], errors="coerce")
    return np.nan


def get_ml_stability(row):
    candidates = [
        "stability_score",
        "ml_stability_score",
        "feature_stability",
    ]

    for col in candidates:
        if col in row.index:
            return pd.to_numeric(row[col], errors="coerce")

    return np.nan


publication_classes = []
publication_scores = []

for _, row in result.iterrows():

    p = get_fdr(row)
    d = get_cohen_d(row)
    consistency = row.get("direction_consistency", np.nan)
    ml_stability = get_ml_stability(row)

    score = 0.0

    # Statistical significance
    if pd.notna(p):
        if p < 0.001:
            score += 2
        elif p < 0.05:
            score += 1

    # Effect size
    if pd.notna(d):
        ad = abs(d)

        if ad >= 0.20:
            score += 2
        elif ad >= 0.10:
            score += 1

    # Subject-level directional robustness
    if pd.notna(consistency):

        if consistency >= 0.80:
            score += 2
        elif consistency >= 0.65:
            score += 1

    # ML stability if available
    if pd.notna(ml_stability):

        if ml_stability >= 0.01:
            score += 1

    if score >= 6:
        cls = "publication_strong_candidate"
    elif score >= 4:
        cls = "publication_supported_candidate"
    elif score >= 2:
        cls = "scientifically_interesting"
    else:
        cls = "exploratory"

    publication_scores.append(score)
    publication_classes.append(cls)

result["publication_robustness_score"] = publication_scores
result["publication_robustness_class"] = publication_classes

# =============================================================================
# SORT
# =============================================================================

result = result.sort_values(
    by=[
        "publication_robustness_score",
        "direction_consistency",
    ],
    ascending=[False, False],
    na_position="last"
).reset_index(drop=True)

# =============================================================================
# SUMMARY
# =============================================================================

summary_rows = []

for target in sorted(result["target"].dropna().unique()):

    x = result[result["target"] == target]

    summary_rows.append({
        "target": target,
        "rows": len(x),
        "strong_candidates": int(
            (x["publication_robustness_class"]
             == "publication_strong_candidate").sum()
        ),
        "supported_candidates": int(
            (x["publication_robustness_class"]
             == "publication_supported_candidate").sum()
        ),
        "scientifically_interesting": int(
            (x["publication_robustness_class"]
             == "scientifically_interesting").sum()
        ),
        "exploratory": int(
            (x["publication_robustness_class"]
             == "exploratory").sum()
        ),
        "mean_direction_consistency": float(
            x["direction_consistency"].mean()
        ) if x["direction_consistency"].notna().any() else np.nan,
    })

summary = pd.DataFrame(summary_rows)

# =============================================================================
# QC
# =============================================================================

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

duplicate_target_feature = int(
    result.duplicated(
        subset=["target", "feature"]
    ).sum()
)

strong_count = int(
    (
        result["publication_robustness_class"]
        == "publication_strong_candidate"
    ).sum()
)

supported_count = int(
    (
        result["publication_robustness_class"]
        == "publication_supported_candidate"
    ).sum()
)

qc = pd.DataFrame([{
    "rows": len(result),
    "unique_features": result["feature"].nunique(),
    "targets": result["target"].nunique(),
    "publication_strong_candidates": strong_count,
    "publication_supported_candidates": supported_count,
    "nan_numeric_cells": nan_numeric,
    "inf_numeric_cells": inf_numeric,
    "duplicate_target_feature": duplicate_target_feature,
    "subject_effect_rows": len(subj),
    "subjects": subj["subject"].nunique(),
    "subject_features": subj["feature"].nunique(),
}])

# =============================================================================
# PRINT RESULTS
# =============================================================================

print("\n" + "=" * 90)
print("FINAL PUBLICATION-GRADE ROBUSTNESS QC")
print("=" * 90)

print(f"Rows:                         {len(result)}")
print(f"Unique features:              {result['feature'].nunique()}")
print(f"Targets:                      {result['target'].nunique()}")
print(f"Publication-strong:           {strong_count}")
print(f"Publication-supported:        {supported_count}")
print(f"NaN numeric cells:            {nan_numeric}")
print(f"Inf numeric cells:            {inf_numeric}")
print(f"Duplicate target-feature:     {duplicate_target_feature}")

print("\n" + "=" * 90)
print("TOP PUBLICATION ROBUSTNESS CANDIDATES")
print("=" * 90)

display_cols = [
    "target",
    "feature",
    "publication_robustness_score",
    "publication_robustness_class",
    "direction_consistency",
    "median_effect",
]

available_display = [
    c for c in display_cols
    if c in result.columns
]

print(
    result[available_display]
    .head(30)
    .to_string(index=False)
)

# =============================================================================
# SAVE
# =============================================================================

result.to_csv(
    OUTPUT,
    index=False
)

summary.to_csv(
    SUMMARY,
    index=False
)

qc.to_csv(
    QC,
    index=False
)

print("\n" + "=" * 90)
print("SAVED")
print("=" * 90)

print(OUTPUT)
print(SUMMARY)
print(QC)

print("\n" + "=" * 90)
print("PUBLICATION-GRADE ROBUSTNESS / STABILITY VALIDATION V1 COMPLETE")
print("=" * 90)

if duplicate_target_feature == 0 and inf_numeric == 0:
    print("STATUS: PASS - PUBLICATION ROBUSTNESS VALIDATION CREATED")
else:
    print("STATUS: REVIEW_REQUIRED")