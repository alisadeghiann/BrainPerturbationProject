from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# SUBJECT-LEVEL PERTURBATION / WHAT-IF ROBUSTNESS INTEGRATION V4
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
    / "robustness_integration_v4"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RESULTS = (
    OUT_DIR
    / "perturbation_robustness_integration_v4.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "perturbation_robustness_summary_v4.csv"
)

OUT_QC = (
    OUT_DIR
    / "perturbation_robustness_integration_qc_v4.csv"
)


# =============================================================================
# HELPERS
# =============================================================================

def find_column(df, candidates):

    lower_map = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in lower_map:
            return lower_map[key]

    return None


def numeric(df, column):

    if column is None:
        return pd.Series(np.nan, index=df.index)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def classify_effect(d):

    if pd.isna(d):
        return "unknown"

    x = abs(float(d))

    if x < 0.10:
        return "negligible"

    if x < 0.20:
        return "small"

    if x < 0.50:
        return "moderate"

    if x < 0.80:
        return "large"

    return "very_large"


def classify_robustness(consistency):

    if pd.isna(consistency):
        return "not_available"

    x = float(consistency)

    if x >= 0.80:
        return "strong"

    if x >= 0.65:
        return "moderate"

    if x >= 0.50:
        return "weak"

    return "unstable"


# =============================================================================
# HEADER
# =============================================================================

print("=" * 88)
print("SUBJECT-LEVEL PERTURBATION / WHAT-IF ROBUSTNESS INTEGRATION V4")
print("=" * 88)


# =============================================================================
# INPUT CHECK
# =============================================================================

if not STAT_FILE.exists():
    raise FileNotFoundError(
        f"Statistical file not found:\n{STAT_FILE}"
    )

if not ROBUST_FILE.exists():
    raise FileNotFoundError(
        f"Robustness file not found:\n{ROBUST_FILE}"
    )


# =============================================================================
# LOAD
# =============================================================================

stat = pd.read_csv(STAT_FILE)

rob = pd.read_csv(ROBUST_FILE)


print("\n" + "=" * 88)
print("INPUT DATA")
print("=" * 88)

print(f"Statistical rows: {len(stat):,}")
print(f"Robustness rows:  {len(rob):,}")

print(f"Statistical columns: {len(stat.columns):,}")
print(f"Robustness columns:  {len(rob.columns):,}")


# =============================================================================
# DETECT STATISTICAL COLUMNS
# =============================================================================

stat_target = find_column(
    stat,
    [
        "target",
        "target_name",
        "outcome"
    ]
)

stat_feature = find_column(
    stat,
    [
        "feature",
        "feature_name",
        "predictor"
    ]
)

stat_p = find_column(
    stat,
    [
        "p_value",
        "pvalue",
        "p"
    ]
)

stat_fdr = find_column(
    stat,
    [
        "p_fdr",
        "fdr",
        "q_value",
        "qvalue",
        "adjusted_p"
    ]
)

stat_effect = find_column(
    stat,
    [
        "cohen_d",
        "effect_size",
        "effect",
        "standardized_effect"
    ]
)

stat_difference = find_column(
    stat,
    [
        "mean_difference",
        "mean_diff",
        "difference",
        "delta"
    ]
)


# =============================================================================
# DETECT ROBUSTNESS COLUMNS
# =============================================================================

rob_target = find_column(
    rob,
    [
        "target",
        "target_name",
        "outcome"
    ]
)

rob_feature = find_column(
    rob,
    [
        "feature",
        "feature_name",
        "predictor"
    ]
)

rob_subject = find_column(
    rob,
    [
        "subject",
        "subject_id",
        "participant",
        "participant_id"
    ]
)

rob_effect = find_column(
    rob,
    [
        "effect",
        "effect_size",
        "subject_effect",
        "mean_difference",
        "difference",
        "effect_direction_value"
    ]
)

rob_positive = find_column(
    rob,
    [
        "positive_fraction",
        "fraction_positive",
        "positive_consistency",
        "positive_rate"
    ]
)

rob_negative = find_column(
    rob,
    [
        "negative_fraction",
        "fraction_negative",
        "negative_consistency",
        "negative_rate"
    ]
)

rob_consistency = find_column(
    rob,
    [
        "direction_consistency",
        "sign_consistency",
        "consistency",
        "stability"
    ]
)


print("\n" + "=" * 88)
print("COLUMN DETECTION")
print("=" * 88)

print(f"Stat target:       {stat_target}")
print(f"Stat feature:      {stat_feature}")
print(f"Stat p-value:      {stat_p}")
print(f"Stat FDR:          {stat_fdr}")
print(f"Stat effect:       {stat_effect}")
print(f"Stat difference:   {stat_difference}")

print(f"Rob target:        {rob_target}")
print(f"Rob feature:       {rob_feature}")
print(f"Rob subject:       {rob_subject}")
print(f"Rob effect:        {rob_effect}")
print(f"Rob positive:      {rob_positive}")
print(f"Rob negative:      {rob_negative}")
print(f"Rob consistency:   {rob_consistency}")


# =============================================================================
# REQUIRED STATISTICAL COLUMNS
# =============================================================================

required_stat = {
    "target": stat_target,
    "feature": stat_feature,
    "p_value": stat_p,
    "p_fdr": stat_fdr,
    "effect": stat_effect
}

missing_stat = [
    name
    for name, column in required_stat.items()
    if column is None
]

if missing_stat:

    raise RuntimeError(
        "Missing required statistical columns: "
        + ", ".join(missing_stat)
    )


# =============================================================================
# REQUIRED ROBUSTNESS COLUMNS
# =============================================================================

required_rob = {
    "target": rob_target,
    "feature": rob_feature
}

missing_rob = [
    name
    for name, column in required_rob.items()
    if column is None
]

if missing_rob:

    raise RuntimeError(
        "Missing required robustness columns: "
        + ", ".join(missing_rob)
    )


# =============================================================================
# STANDARDIZE STATISTICAL DATA
# =============================================================================

s = pd.DataFrame()

s["target"] = (
    stat[stat_target]
    .astype(str)
    .str.strip()
)

s["feature"] = (
    stat[stat_feature]
    .astype(str)
    .str.strip()
)

s["p_value"] = numeric(
    stat,
    stat_p
)

s["p_fdr"] = numeric(
    stat,
    stat_fdr
)

s["cohen_d"] = numeric(
    stat,
    stat_effect
)

if stat_difference is not None:

    s["mean_difference"] = numeric(
        stat,
        stat_difference
    )

else:

    s["mean_difference"] = np.nan


# =============================================================================
# STANDARDIZE ROBUSTNESS DATA
# =============================================================================

r = pd.DataFrame()

r["target"] = (
    rob[rob_target]
    .astype(str)
    .str.strip()
)

r["feature"] = (
    rob[rob_feature]
    .astype(str)
    .str.strip()
)

if rob_subject is not None:

    r["subject"] = (
        rob[rob_subject]
        .astype(str)
        .str.strip()
    )

else:

    r["subject"] = np.nan


# =============================================================================
# SUBJECT-LEVEL EFFECT
# =============================================================================

if rob_effect is not None:

    r["subject_effect"] = numeric(
        rob,
        rob_effect
    )

else:

    r["subject_effect"] = np.nan


# =============================================================================
# POSITIVE / NEGATIVE FRACTION
# =============================================================================

if rob_positive is not None:

    r["positive_fraction"] = numeric(
        rob,
        rob_positive
    )

else:

    r["positive_fraction"] = np.nan


if rob_negative is not None:

    r["negative_fraction"] = numeric(
        rob,
        rob_negative
    )

else:

    r["negative_fraction"] = np.nan


# =============================================================================
# EXPLICIT DIRECTION CONSISTENCY
# =============================================================================

if (
    r["positive_fraction"].notna().any()
    and r["negative_fraction"].notna().any()
):

    r["direction_consistency"] = (
        r[
            [
                "positive_fraction",
                "negative_fraction"
            ]
        ]
        .max(axis=1)
    )

else:

    r["direction_consistency"] = np.nan


# =============================================================================
# IF EXPLICIT CONSISTENCY EXISTS, USE IT
# =============================================================================

if rob_consistency is not None:

    explicit_consistency = numeric(
        rob,
        rob_consistency
    )

    r["direction_consistency"] = (
        r["direction_consistency"]
        .fillna(explicit_consistency)
    )


# =============================================================================
# SUBJECT-LEVEL AGGREGATION
# =============================================================================

group_columns = [
    "target",
    "feature"
]

robust_summary = (
    r
    .groupby(group_columns, as_index=False)
    .agg(
        subject_count=(
            "subject",
            "nunique"
        ),
        subject_effect_mean=(
            "subject_effect",
            "mean"
        ),
        subject_effect_median=(
            "subject_effect",
            "median"
        ),
        subject_effect_std=(
            "subject_effect",
            "std"
        ),
        direction_consistency=(
            "direction_consistency",
            "mean"
        ),
        positive_fraction=(
            "positive_fraction",
            "mean"
        ),
        negative_fraction=(
            "negative_fraction",
            "mean"
        )
    )
)


# =============================================================================
# DIRECT SIGN CONSISTENCY FROM SUBJECT EFFECT
# =============================================================================

if (
    r["subject_effect"].notna().any()
    and rob_subject is not None
):

    sign_table = r[
        [
            "target",
            "feature",
            "subject_effect"
        ]
    ].copy()

    sign_table["positive"] = (
        sign_table["subject_effect"] > 0
    ).astype(float)

    sign_table["negative"] = (
        sign_table["subject_effect"] < 0
    ).astype(float)

    sign_summary = (
        sign_table
        .groupby(
            [
                "target",
                "feature"
            ],
            as_index=False
        )
        .agg(
            calculated_positive_fraction=(
                "positive",
                "mean"
            ),
            calculated_negative_fraction=(
                "negative",
                "mean"
            )
        )
    )

    sign_summary["calculated_direction_consistency"] = (
        sign_summary[
            [
                "calculated_positive_fraction",
                "calculated_negative_fraction"
            ]
        ]
        .max(axis=1)
    )

    robust_summary = robust_summary.merge(
        sign_summary,
        on=[
            "target",
            "feature"
        ],
        how="left"
    )

    robust_summary["direction_consistency"] = (
        robust_summary[
            "direction_consistency"
        ]
        .fillna(
            robust_summary[
                "calculated_direction_consistency"
            ]
        )
    )


# =============================================================================
# MERGE STATISTICAL + ROBUSTNESS
# =============================================================================

final = s.merge(
    robust_summary,
    on=[
        "target",
        "feature"
    ],
    how="left"
)


# =============================================================================
# STATISTICAL FLAGS
# =============================================================================

final["fdr_significant"] = (
    final["p_fdr"] < 0.05
)

final["p_significant"] = (
    final["p_value"] < 0.05
)


# =============================================================================
# EFFECT METRICS
# =============================================================================

final["abs_cohen_d"] = (
    final["cohen_d"]
    .abs()
)

final["effect_magnitude"] = (
    final["cohen_d"]
    .apply(classify_effect)
)


# =============================================================================
# STATISTICAL DIRECTION
# =============================================================================

final["statistical_direction"] = np.where(
    final["cohen_d"] > 0,
    "positive",
    np.where(
        final["cohen_d"] < 0,
        "negative",
        "zero"
    )
)


# =============================================================================
# ROBUSTNESS CLASSIFICATION
# =============================================================================

final["robustness_class"] = (
    final["direction_consistency"]
    .apply(classify_robustness)
)


# =============================================================================
# DIRECTION AGREEMENT
# =============================================================================

def direction_agreement(row):

    stat_direction = row[
        "statistical_direction"
    ]

    consistency = row[
        "direction_consistency"
    ]

    if pd.isna(consistency):
        return "not_available"

    positive = row[
        "positive_fraction"
    ]

    negative = row[
        "negative_fraction"
    ]

    if (
        pd.notna(positive)
        and pd.notna(negative)
    ):

        if stat_direction == "positive":

            if positive >= negative:
                return "agreement"

            return "disagreement"

        if stat_direction == "negative":

            if negative >= positive:
                return "agreement"

            return "disagreement"

    return "not_available"


final["direction_agreement"] = (
    final.apply(
        direction_agreement,
        axis=1
    )
)


# =============================================================================
# SCIENTIFIC EVIDENCE CLASSIFICATION
# =============================================================================

def evidence_class(row):

    fdr = bool(
        row["fdr_significant"]
    )

    effect = row[
        "abs_cohen_d"
    ]

    consistency = row[
        "direction_consistency"
    ]

    agreement = row[
        "direction_agreement"
    ]

    if (
        fdr
        and pd.notna(effect)
        and effect >= 0.20
        and pd.notna(consistency)
        and consistency >= 0.80
        and agreement == "agreement"
    ):
        return "strong_scientific_candidate"

    if (
        fdr
        and pd.notna(consistency)
        and consistency >= 0.65
        and agreement == "agreement"
    ):
        return "moderate_scientific_candidate"

    if (
        fdr
        and agreement == "agreement"
    ):
        return "statistically_supported_candidate"

    if fdr:
        return "statistically_significant_but_less_robust"

    if (
        pd.notna(consistency)
        and consistency >= 0.80
        and pd.notna(effect)
        and effect >= 0.20
    ):
        return "robust_but_not_fdr_significant"

    return "weak_or_unresolved"


final["scientific_evidence_class"] = (
    final.apply(
        evidence_class,
        axis=1
    )
)


# =============================================================================
# SCIENTIFIC PRIORITY SCORE
# =============================================================================

def priority_score(row):

    effect = row["abs_cohen_d"]

    if pd.isna(effect):
        effect_score = 0.0
    else:
        effect_score = min(
            float(effect) / 0.50,
            1.0
        )

    fdr_score = (
        1.0
        if row["fdr_significant"]
        else 0.0
    )

    consistency = row[
        "direction_consistency"
    ]

    if pd.isna(consistency):
        consistency_score = 0.0
    else:
        consistency_score = min(
            max(float(consistency), 0.0),
            1.0
        )

    agreement_score = (
        1.0
        if row["direction_agreement"] == "agreement"
        else 0.0
    )

    return (
        0.35 * effect_score
        + 0.30 * fdr_score
        + 0.20 * consistency_score
        + 0.15 * agreement_score
    )


final["scientific_priority_score"] = (
    final.apply(
        priority_score,
        axis=1
    )
)


# =============================================================================
# RANK
# =============================================================================

final = final.sort_values(
    [
        "target",
        "scientific_priority_score",
        "abs_cohen_d"
    ],
    ascending=[
        True,
        False,
        False
    ]
).reset_index(drop=True)

final["scientific_rank"] = (
    final
    .groupby("target")
    .cumcount()
    + 1
)


# =============================================================================
# TARGET SUMMARY
# =============================================================================

summary_rows = []

for target, group in final.groupby(
    "target"
):

    summary_rows.append({

        "target": target,

        "features_analyzed": int(
            group["feature"].nunique()
        ),

        "fdr_significant": int(
            group["fdr_significant"].sum()
        ),

        "p_significant": int(
            group["p_significant"].sum()
        ),

        "strong_scientific_candidates": int(
            (
                group[
                    "scientific_evidence_class"
                ]
                == "strong_scientific_candidate"
            ).sum()
        ),

        "moderate_scientific_candidates": int(
            (
                group[
                    "scientific_evidence_class"
                ]
                == "moderate_scientific_candidate"
            ).sum()
        ),

        "mean_abs_cohen_d": float(
            group["abs_cohen_d"].mean()
        ),

        "max_abs_cohen_d": float(
            group["abs_cohen_d"].max()
        ),

        "mean_direction_consistency": float(
            group[
                "direction_consistency"
            ].mean()
        )
        if group[
            "direction_consistency"
        ].notna().any()
        else np.nan
    })


summary = pd.DataFrame(
    summary_rows
)


# =============================================================================
# PRINT TOP RESULTS
# =============================================================================

print("\n" + "=" * 88)
print("TOP SCIENTIFIC WHAT-IF CANDIDATES")
print("=" * 88)

for target in sorted(
    final["target"].unique()
):

    print(
        f"\nTARGET: {target.upper()}"
    )

    print("-" * 88)

    top = final[
        final["target"] == target
    ].head(20)

    columns = [
        "scientific_rank",
        "feature",
        "cohen_d",
        "p_fdr",
        "fdr_significant",
        "direction_consistency",
        "robustness_class",
        "direction_agreement",
        "scientific_evidence_class",
        "scientific_priority_score"
    ]

    columns = [
        c for c in columns
        if c in top.columns
    ]

    print(
        top[columns]
        .to_string(index=False)
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
        final[
            numeric_columns
        ]
        .to_numpy(
            dtype=float
        )
    )
    .sum()
)

duplicate_keys = int(
    final[
        [
            "target",
            "feature"
        ]
    ]
    .duplicated()
    .sum()
)

robustness_available = int(
    final[
        "direction_consistency"
    ]
    .notna()
    .sum()
)

fdr_count = int(
    final[
        "fdr_significant"
    ]
    .sum()
)


# =============================================================================
# QC STATUS
# =============================================================================

if (
    nan_numeric == 0
    and inf_numeric == 0
    and duplicate_keys == 0
    and robustness_available > 0
):

    status = "PASS"

else:

    status = "REVIEW_REQUIRED"


print("\n" + "=" * 88)
print("FINAL ROBUSTNESS INTEGRATION QC")
print("=" * 88)

print(
    f"Targets:                  "
    f"{final['target'].nunique():,}"
)

print(
    f"Features:                 "
    f"{final['feature'].nunique():,}"
)

print(
    f"Result rows:              "
    f"{len(final):,}"
)

print(
    f"FDR significant:          "
    f"{fdr_count:,}"
)

print(
    f"Robustness available:     "
    f"{robustness_available:,}"
)

print(
    f"NaN numeric values:       "
    f"{nan_numeric:,}"
)

print(
    f"Inf numeric values:       "
    f"{inf_numeric:,}"
)

print(
    f"Duplicate target-feature:"
    f" {duplicate_keys:,}"
)


# =============================================================================
# SAVE
# =============================================================================

final.to_csv(
    OUT_RESULTS,
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

    "result_rows": int(
        len(final)
    ),

    "fdr_significant": fdr_count,

    "robustness_available_rows":
        robustness_available,

    "nan_numeric_values":
        nan_numeric,

    "inf_numeric_values":
        inf_numeric,

    "duplicate_target_feature":
        duplicate_keys,

    "status":
        status

}])

qc.to_csv(
    OUT_QC,
    index=False
)


# =============================================================================
# COMPLETE
# =============================================================================

print("\n" + "=" * 88)
print("SUBJECT-LEVEL PERTURBATION / WHAT-IF ROBUSTNESS INTEGRATION V4 COMPLETE")
print("=" * 88)

print("\nResults:")
print(OUT_RESULTS)

print("\nSummary:")
print(OUT_SUMMARY)

print("\nQC:")
print(OUT_QC)

print("\n" + "=" * 88)
print(
    f"STATUS: {status} - "
    "ROBUSTNESS INTEGRATION V4 CREATED"
)
print("=" * 88)