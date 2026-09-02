# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# PUBLICATION-GRADE PERTURBATION / WHAT-IF EFFECT VALIDATION V2
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT_CANDIDATES = [
    BASE / "features" / "perturbation_analysis" / "final_scientific_ranking_v2"
    / "final_scientific_evidence_ranking_v2.csv",

    BASE / "features" / "perturbation_analysis" / "final_master_v1"
    / "final_perturbation_master_v1.csv",

    BASE / "features" / "perturbation_analysis" / "final_evidence_v2"
    / "final_scientific_perturbation_evidence_v2.csv",
]

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "publication_grade_effect_validation_v2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RESULTS = OUTPUT_DIR / "publication_grade_effect_validation_v2.csv"
OUT_SUMMARY = OUTPUT_DIR / "publication_grade_effect_validation_summary_v2.csv"
OUT_QC = OUTPUT_DIR / "publication_grade_effect_validation_qc_v2.csv"


# =============================================================================
# HELPERS
# =============================================================================

def find_input_file():
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "\nNo valid scientific perturbation input file was found.\n"
        "Checked:\n"
        + "\n".join(str(p) for p in INPUT_CANDIDATES)
    )


def first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


# =============================================================================
# LOAD
# =============================================================================

print("=" * 90)
print("PUBLICATION-GRADE PERTURBATION / WHAT-IF EFFECT VALIDATION V2")
print("=" * 90)
print(f"Project root: {BASE}")

INPUT = find_input_file()

print("\n" + "=" * 90)
print("INPUT FILE")
print("=" * 90)
print(INPUT)

df = pd.read_csv(INPUT)

print("\n" + "=" * 90)
print("INPUT SUMMARY")
print("=" * 90)
print(f"Rows:       {len(df):,}")
print(f"Columns:    {len(df.columns)}")

print("\nColumns:")
print(list(df.columns))


# =============================================================================
# COLUMN DISCOVERY
# =============================================================================

print("\n" + "=" * 90)
print("COLUMN DISCOVERY")
print("=" * 90)

feature_col = first_existing_column(
    df,
    ["feature", "Feature", "feature_name"]
)

target_col = first_existing_column(
    df,
    ["target", "Target", "target_name"]
)

effect_col = first_existing_column(
    df,
    [
        "cohen_d",
        "effect_size",
        "mean_effect_size",
        "effect_magnitude"
    ]
)

p_col = first_existing_column(
    df,
    [
        "p_fdr",
        "p_value_fdr",
        "fdr_p",
        "adjusted_p",
        "q_value"
    ]
)

direction_consistency_col = first_existing_column(
    df,
    [
        "direction_consistency",
        "target_direction_consistency",
        "positive_fraction",
        "direction_agreement",
        "subject_direction_consistency"
    ]
)

positive_fraction_col = first_existing_column(
    df,
    [
        "positive_fraction",
        "positive_subject_fraction"
    ]
)

negative_fraction_col = first_existing_column(
    df,
    [
        "negative_fraction",
        "negative_subject_fraction"
    ]
)

robustness_col = first_existing_column(
    df,
    [
        "target_robustness_class",
        "robustness_class",
        "subject_robustness_class"
    ]
)

cross_target_col = first_existing_column(
    df,
    [
        "cross_target_pattern",
        "cross_target_class",
        "cross_target_relationship"
    ]
)

scientific_class_col = first_existing_column(
    df,
    [
        "scientific_evidence_class",
        "final_evidence_class",
        "evidence_class"
    ]
)

priority_col = first_existing_column(
    df,
    [
        "final_scientific_priority_score",
        "scientific_priority_score",
        "final_priority_score_v1",
        "scientific_ml_rank"
    ]
)

print(f"Feature column:                {feature_col}")
print(f"Target column:                 {target_col}")
print(f"Effect column:                 {effect_col}")
print(f"FDR column:                    {p_col}")
print(f"Direction consistency column:  {direction_consistency_col}")
print(f"Positive fraction column:      {positive_fraction_col}")
print(f"Negative fraction column:      {negative_fraction_col}")
print(f"Robustness column:             {robustness_col}")
print(f"Cross-target column:           {cross_target_col}")
print(f"Scientific class column:       {scientific_class_col}")
print(f"Priority column:               {priority_col}")


# =============================================================================
# REQUIRED VALIDATION
# =============================================================================

if feature_col is None:
    raise ValueError("No feature column found.")

if target_col is None:
    raise ValueError("No target column found.")

if effect_col is None:
    raise ValueError("No effect-size column found.")

if p_col is None:
    raise ValueError("No FDR/p-value column found.")


# =============================================================================
# NORMALIZE CORE COLUMNS
# =============================================================================

work = pd.DataFrame()

work["feature"] = df[feature_col].astype(str)
work["target"] = df[target_col].astype(str)

work["effect_size"] = pd.to_numeric(
    df[effect_col],
    errors="coerce"
)

work["p_fdr"] = pd.to_numeric(
    df[p_col],
    errors="coerce"
)


# =============================================================================
# DIRECTION CONSISTENCY RECONSTRUCTION
# =============================================================================

print("\n" + "=" * 90)
print("DIRECTION CONSISTENCY VALIDATION")
print("=" * 90)

if direction_consistency_col is not None:

    direction_values = pd.to_numeric(
        df[direction_consistency_col],
        errors="coerce"
    )

    # If the selected column is actually positive_fraction,
    # convert it into consistency only if possible.
    if (
        direction_consistency_col == positive_fraction_col
        and negative_fraction_col is not None
    ):
        pos = pd.to_numeric(df[positive_fraction_col], errors="coerce")
        neg = pd.to_numeric(df[negative_fraction_col], errors="coerce")

        direction_values = pd.concat(
            [pos, neg],
            axis=1
        ).max(axis=1)

        print(
            "Direction consistency reconstructed from "
            "positive/negative fractions."
        )
    else:
        print(
            f"Using existing column: {direction_consistency_col}"
        )

else:

    print(
        "No direct direction-consistency column found."
    )

    if (
        positive_fraction_col is not None
        and negative_fraction_col is not None
    ):

        pos = pd.to_numeric(
            df[positive_fraction_col],
            errors="coerce"
        )

        neg = pd.to_numeric(
            df[negative_fraction_col],
            errors="coerce"
        )

        direction_values = pd.concat(
            [pos, neg],
            axis=1
        ).max(axis=1)

        print(
            "Direction consistency reconstructed from "
            "positive_fraction and negative_fraction."
        )

    else:

        direction_values = pd.Series(
            np.nan,
            index=df.index
        )

        print(
            "WARNING: Direction consistency unavailable."
        )

work["direction_consistency"] = direction_values


# =============================================================================
# OPTIONAL COLUMNS
# =============================================================================

if robustness_col is not None:
    work["robustness_class"] = df[robustness_col].astype(str)
else:
    work["robustness_class"] = "not_available"


if cross_target_col is not None:
    work["cross_target_pattern"] = df[cross_target_col].astype(str)
else:
    work["cross_target_pattern"] = "not_available"


if scientific_class_col is not None:
    work["scientific_evidence_class"] = (
        df[scientific_class_col].astype(str)
    )
else:
    work["scientific_evidence_class"] = "not_available"


if priority_col is not None:
    work["priority_score"] = pd.to_numeric(
        df[priority_col],
        errors="coerce"
    )
else:
    work["priority_score"] = np.nan


# =============================================================================
# SIGNIFICANCE
# =============================================================================

work["fdr_significant"] = (
    work["p_fdr"] < 0.05
)

work["abs_effect_size"] = (
    work["effect_size"].abs()
)


# =============================================================================
# EFFECT MAGNITUDE
# =============================================================================

def classify_effect(x):

    if pd.isna(x):
        return "not_available"

    x = abs(float(x))

    if x < 0.10:
        return "negligible"

    elif x < 0.20:
        return "small"

    elif x < 0.50:
        return "small_to_moderate"

    elif x < 0.80:
        return "moderate"

    else:
        return "large"


work["effect_magnitude"] = (
    work["effect_size"].apply(classify_effect)
)


# =============================================================================
# DIRECTION
# =============================================================================

def classify_direction(x):

    if pd.isna(x):
        return "not_available"

    if x > 0:
        return "positive"

    if x < 0:
        return "negative"

    return "zero"


work["effect_direction"] = (
    work["effect_size"].apply(classify_direction)
)


# =============================================================================
# DIRECTION CONSISTENCY CLASS
# =============================================================================

def classify_consistency(x):

    if pd.isna(x):
        return "not_available"

    x = abs(float(x))

    if x >= 0.80:
        return "strong"

    elif x >= 0.60:
        return "moderate"

    elif x >= 0.50:
        return "weak"

    return "inconsistent"


work["direction_consistency_class"] = (
    work["direction_consistency"]
    .apply(classify_consistency)
)


# =============================================================================
# PUBLICATION-GRADE VALIDATION CLASS
# =============================================================================

def validation_class(row):

    sig = bool(row["fdr_significant"])

    effect = row["abs_effect_size"]

    consistency = row["direction_consistency"]

    robustness = str(row["robustness_class"]).lower()

    cross_target = str(
        row["cross_target_pattern"]
    ).lower()

    if not sig:
        return "non_significant"

    if pd.isna(effect):
        return "insufficient_effect_data"

    # Strong statistical + directional evidence
    if (
        effect >= 0.20
        and not pd.isna(consistency)
        and consistency >= 0.80
        and (
            "strong" in robustness
            or "moderate" in robustness
            or "replicated" in robustness
        )
    ):
        return "publication_strong"

    # Moderate evidence
    if (
        effect >= 0.10
        and not pd.isna(consistency)
        and consistency >= 0.60
    ):
        return "publication_supported"

    # Statistically significant but weak effect
    if (
        sig
        and effect < 0.10
    ):
        return "statistically_significant_weak_effect"

    # Significant but inconsistent
    if (
        sig
        and not pd.isna(consistency)
        and consistency < 0.60
    ):
        return "significant_but_directionally_inconsistent"

    return "significant_supported"


work["publication_validation_class"] = (
    work.apply(validation_class, axis=1)
)


# =============================================================================
# PUBLICATION SCORE
# =============================================================================

def publication_score(row):

    score = 0.0

    # Statistical significance
    if row["fdr_significant"]:
        score += 2.0

    # Effect size
    effect = row["abs_effect_size"]

    if not pd.isna(effect):

        if effect >= 0.80:
            score += 4.0

        elif effect >= 0.50:
            score += 3.0

        elif effect >= 0.20:
            score += 2.0

        elif effect >= 0.10:
            score += 1.0

        else:
            score += 0.5

    # Directional consistency
    consistency = row["direction_consistency"]

    if not pd.isna(consistency):

        if consistency >= 0.80:
            score += 2.0

        elif consistency >= 0.60:
            score += 1.0

        elif consistency >= 0.50:
            score += 0.5

    # Robustness
    robustness = str(
        row["robustness_class"]
    ).lower()

    if "strong" in robustness:
        score += 2.0

    elif "moderate" in robustness:
        score += 1.0

    elif "weak" in robustness:
        score += 0.5

    # Cross-target evidence
    cross_target = str(
        row["cross_target_pattern"]
    ).lower()

    if "shared" in cross_target:
        score += 1.0

    elif "preferential" in cross_target:
        score += 0.5

    return score


work["publication_grade_score"] = (
    work.apply(publication_score, axis=1)
)


# =============================================================================
# RANKING
# =============================================================================

work = work.sort_values(
    [
        "publication_grade_score",
        "fdr_significant",
        "abs_effect_size"
    ],
    ascending=[False, False, False]
).reset_index(drop=True)

work["publication_grade_rank"] = (
    np.arange(1, len(work) + 1)
)


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("PUBLICATION-GRADE VALIDATION SUMMARY")
print("=" * 90)

print(
    f"Rows:                 {len(work):,}"
)

print(
    f"Features:             {work['feature'].nunique():,}"
)

print(
    f"Targets:              {work['target'].nunique():,}"
)

print(
    f"FDR significant:      {int(work['fdr_significant'].sum()):,}"
)

print(
    f"Publication strong:   "
    f"{(work['publication_validation_class'] == 'publication_strong').sum():,}"
)

print(
    f"Publication supported:"
    f" {(work['publication_validation_class'] == 'publication_supported').sum():,}"
)

print(
    f"Non-significant:      "
    f"{(work['publication_validation_class'] == 'non_significant').sum():,}"
)


# =============================================================================
# TOP EVIDENCE
# =============================================================================

print("\n" + "=" * 90)
print("TOP PUBLICATION-GRADE EVIDENCE")
print("=" * 90)

display_cols = [
    "publication_grade_rank",
    "target",
    "feature",
    "effect_size",
    "p_fdr",
    "direction_consistency",
    "effect_magnitude",
    "robustness_class",
    "cross_target_pattern",
    "publication_validation_class",
    "publication_grade_score"
]

print(
    work[display_cols]
    .head(30)
    .to_string(index=False)
)


# =============================================================================
# TARGET SUMMARY
# =============================================================================

target_summary = (
    work
    .groupby("target")
    .agg(
        rows=("feature", "count"),
        fdr_significant=("fdr_significant", "sum"),
        mean_abs_effect=("abs_effect_size", "mean"),
        max_abs_effect=("abs_effect_size", "max"),
        mean_direction_consistency=(
            "direction_consistency",
            "mean"
        ),
        publication_strong=(
            "publication_validation_class",
            lambda x: (
                x == "publication_strong"
            ).sum()
        ),
        publication_supported=(
            "publication_validation_class",
            lambda x: (
                x == "publication_supported"
            ).sum()
        ),
    )
    .reset_index()
)


# =============================================================================
# QC
# =============================================================================

numeric_cols = work.select_dtypes(
    include=[np.number]
).columns

nan_numeric = int(
    work[numeric_cols]
    .isna()
    .sum()
    .sum()
)

inf_numeric = int(
    np.isinf(
        work[numeric_cols]
        .to_numpy(
            dtype=float,
            na_value=np.nan
        )
    ).sum()
)

duplicate_target_feature = int(
    work.duplicated(
        subset=["target", "feature"]
    ).sum()
)

qc = pd.DataFrame(
    {
        "metric": [
            "input_file",
            "rows",
            "features",
            "targets",
            "fdr_significant",
            "publication_strong",
            "publication_supported",
            "nan_numeric_cells",
            "inf_numeric_cells",
            "duplicate_target_feature",
            "direction_consistency_source"
        ],
        "value": [
            str(INPUT),
            len(work),
            work["feature"].nunique(),
            work["target"].nunique(),
            int(work["fdr_significant"].sum()),
            int(
                (
                    work["publication_validation_class"]
                    == "publication_strong"
                ).sum()
            ),
            int(
                (
                    work["publication_validation_class"]
                    == "publication_supported"
                ).sum()
            ),
            nan_numeric,
            inf_numeric,
            duplicate_target_feature,
            direction_consistency_col
            if direction_consistency_col is not None
            else "reconstructed_or_unavailable"
        ]
    }
)


# =============================================================================
# SAVE
# =============================================================================

print("\n" + "=" * 90)
print("SAVING PUBLICATION-GRADE VALIDATION")
print("=" * 90)

work.to_csv(
    OUT_RESULTS,
    index=False
)

target_summary.to_csv(
    OUT_SUMMARY,
    index=False
)

qc.to_csv(
    OUT_QC,
    index=False
)

print("\nSaved:")
print(OUT_RESULTS)
print(OUT_SUMMARY)
print(OUT_QC)


# =============================================================================
# FINAL STATUS
# =============================================================================

print("\n" + "=" * 90)
print("FINAL PUBLICATION-GRADE EFFECT VALIDATION QC")
print("=" * 90)

print(
    f"Rows:                         {len(work)}"
)

print(
    f"Unique features:              {work['feature'].nunique()}"
)

print(
    f"Targets:                      {work['target'].nunique()}"
)

print(
    f"FDR-significant rows:         "
    f"{int(work['fdr_significant'].sum())}"
)

print(
    f"Publication-strong rows:      "
    f"{int((work['publication_validation_class'] == 'publication_strong').sum())}"
)

print(
    f"Publication-supported rows:   "
    f"{int((work['publication_validation_class'] == 'publication_supported').sum())}"
)

print(
    f"NaN numeric cells:             {nan_numeric}"
)

print(
    f"Inf numeric cells:             {inf_numeric}"
)

print(
    f"Duplicate target-feature:      {duplicate_target_feature}"
)

if (
    duplicate_target_feature == 0
    and inf_numeric == 0
):
    print("\nSTATUS: PASS - PUBLICATION-GRADE EFFECT VALIDATION CREATED")
else:
    print("\nSTATUS: REVIEW_REQUIRED")