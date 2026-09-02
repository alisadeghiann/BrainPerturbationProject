# ================================================================
# FINAL SCIENTIFIC PERTURBATION / WHAT-IF EVIDENCE V2
# ================================================================
# Purpose:
#   Create the clean final EEG-only scientific evidence table
#   from the integrated perturbation evidence.
#
# Important:
#   - memory_cond is NOT an EEG feature -> excluded from EEG ranking
#   - target variables are NEVER used as predictors
#   - remember and correct are analyzed separately
#   - previous files are READ-ONLY
# ================================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ================================================================
# PROJECT PATHS
# ================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_evidence_v1"
    / "final_perturbation_evidence_v1.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_evidence_v2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


OUTPUT_MAIN = (
    OUTPUT_DIR
    / "final_scientific_perturbation_evidence_v2.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "final_scientific_perturbation_evidence_summary_v2.csv"
)

OUTPUT_QC = (
    OUTPUT_DIR
    / "final_scientific_perturbation_evidence_qc_v2.csv"
)


# ================================================================
# EEG SCIENTIFIC FEATURES
# ================================================================
# These are the 55 EEG-derived scientific predictors retained
# after the previous target-independent feature-selection stage.
# ================================================================

EEG_FEATURES = [
    "delta_abs",
    "theta_abs",
    "alpha_abs",
    "beta_abs",
    "gamma_abs",

    "delta_rel",
    "theta_rel",
    "alpha_rel",
    "beta_rel",
    "gamma_rel",

    "theta_alpha_ratio",
    "theta_beta_ratio",
    "alpha_beta_ratio",
    "delta_theta_ratio",

    "delta_frontal",
    "theta_frontal",
    "alpha_frontal",
    "beta_frontal",
    "gamma_frontal",

    "delta_central",
    "theta_central",
    "alpha_central",
    "beta_central",
    "gamma_central",

    "delta_parietal",
    "theta_parietal",
    "alpha_parietal",
    "beta_parietal",
    "gamma_parietal",

    "delta_occipital",
    "theta_occipital",
    "alpha_occipital",
    "beta_occipital",
    "gamma_occipital",

    "delta_temporal",
    "theta_temporal",
    "alpha_temporal",
    "beta_temporal",
    "gamma_temporal",

    "theta_alpha_frontal_ratio",
    "alpha_beta_frontal_ratio",

    "theta_alpha_central_ratio",
    "alpha_beta_central_ratio",

    "theta_alpha_parietal_ratio",
    "alpha_beta_parietal_ratio",

    "theta_alpha_occipital_ratio",
    "alpha_beta_occipital_ratio",

    "theta_alpha_temporal_ratio",
    "alpha_beta_temporal_ratio",

    "delta_frontoparietal_diff",
    "delta_frontoparietal_ratio",

    "theta_frontoparietal_diff",
    "theta_frontoparietal_ratio",

    "alpha_frontoparietal_diff",
    "alpha_frontoparietal_ratio",

    "beta_frontoparietal_diff",
    "beta_frontoparietal_ratio",

    "gamma_frontoparietal_diff",
    "gamma_frontoparietal_ratio",
]


# ================================================================
# START
# ================================================================

print("=" * 80)
print("FINAL SCIENTIFIC PERTURBATION / WHAT-IF EVIDENCE V2")
print("=" * 80)

print(f"Project root: {BASE}")
print()


# ================================================================
# INPUT VALIDATION
# ================================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Integrated evidence file not found:\n{INPUT}"
    )

print("INPUT:")
print(INPUT)
print()


df = pd.read_csv(INPUT)


print("=" * 80)
print("INPUT SUMMARY")
print("=" * 80)

print(f"Rows:       {len(df):,}")
print(f"Columns:    {len(df.columns):,}")

if "target" in df.columns:
    print(
        f"Targets:    {df['target'].nunique():,}"
    )

if "feature" in df.columns:
    print(
        f"Features:   {df['feature'].nunique():,}"
    )

print()


# ================================================================
# REQUIRED COLUMNS
# ================================================================

REQUIRED = [
    "target",
    "feature",
]

missing = [
    c for c in REQUIRED
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

print("=" * 80)
print("REQUIRED COLUMN CHECK")
print("=" * 80)

print("PASS")
print()


# ================================================================
# DUPLICATE CHECK
# ================================================================

print("=" * 80)
print("DUPLICATE TARGET-FEATURE CHECK")
print("=" * 80)

duplicates = df.duplicated(
    subset=["target", "feature"]
).sum()

print(f"Duplicate target-feature rows: {duplicates}")

if duplicates > 0:
    raise RuntimeError(
        "Duplicate target-feature rows detected."
    )

print("PASS")
print()


# ================================================================
# EEG FEATURE FILTER
# ================================================================

print("=" * 80)
print("EEG SCIENTIFIC FEATURE FILTER")
print("=" * 80)

input_features = set(
    df["feature"].astype(str)
)

eeg_features_present = sorted(
    input_features.intersection(EEG_FEATURES)
)

non_eeg_features = sorted(
    input_features.difference(EEG_FEATURES)
)

print(
    f"Expected EEG scientific features: {len(EEG_FEATURES)}"
)

print(
    f"EEG scientific features present:   {len(eeg_features_present)}"
)

print(
    f"Non-EEG / excluded features:       {len(non_eeg_features)}"
)

if non_eeg_features:
    print()
    print("Excluded features:")
    for feature in non_eeg_features:
        print(f"  - {feature}")

print()


# ================================================================
# STRICT FEATURE VALIDATION
# ================================================================

missing_eeg = sorted(
    set(EEG_FEATURES)
    - set(eeg_features_present)
)

if missing_eeg:
    print("WARNING - EEG features missing from integrated evidence:")
    for feature in missing_eeg:
        print(f"  - {feature}")
    print()


# ================================================================
# CREATE EEG-ONLY DATASET
# ================================================================

scientific = df[
    df["feature"].isin(EEG_FEATURES)
].copy()

scientific = scientific.reset_index(drop=True)


# ================================================================
# TARGET VALIDATION
# ================================================================

print("=" * 80)
print("TARGET VALIDATION")
print("=" * 80)

targets = sorted(
    scientific["target"]
    .astype(str)
    .unique()
)

print("Targets:")
for target in targets:
    print(f"  - {target}")

print()


# ================================================================
# IMPORTANT SCIENTIFIC SEPARATION
# ================================================================

print("=" * 80)
print("SCIENTIFIC TARGET SEPARATION")
print("=" * 80)

for target in ["remember", "correct"]:
    count = (
        scientific["target"]
        .astype(str)
        .eq(target)
        .sum()
    )

    print(
        f"{target:10s}: {count:3d} EEG feature rows"
    )

print()


# ================================================================
# NUMERIC COLUMN QC
# ================================================================

numeric_columns = scientific.select_dtypes(
    include=[np.number]
).columns.tolist()

nan_count = int(
    scientific[numeric_columns]
    .isna()
    .sum()
    .sum()
)

inf_count = int(
    np.isinf(
        scientific[numeric_columns].to_numpy(
            dtype=float
        )
    ).sum()
)

print("=" * 80)
print("NUMERIC QC")
print("=" * 80)

print(
    f"Numeric columns: {len(numeric_columns)}"
)

print(
    f"NaN values:      {nan_count}"
)

print(
    f"Inf values:      {inf_count}"
)

if inf_count > 0:
    raise RuntimeError(
        "Infinite values detected."
    )

print()


# ================================================================
# SCIENTIFIC EVIDENCE CLASSIFICATION
# ================================================================

print("=" * 80)
print("SCIENTIFIC EVIDENCE CLASSIFICATION")
print("=" * 80)


def classify_evidence(row):

    robustness = str(
        row.get(
            "target_robustness_class",
            ""
        )
    ).lower()

    agreement = str(
        row.get(
            "subject_vs_statistical_direction",
            ""
        )
    ).lower()

    fdr = row.get(
        "fdr_significant",
        np.nan
    )

    if pd.isna(fdr):
        fdr_significant = False
    else:
        fdr_significant = bool(fdr)

    # ------------------------------------------------------------
    # Strong replicated evidence
    # ------------------------------------------------------------

    if (
        fdr_significant
        and robustness == "moderate"
        and agreement == "agreement"
    ):
        return "strong_replicated"

    # ------------------------------------------------------------
    # Moderate replicated evidence
    # ------------------------------------------------------------

    if (
        fdr_significant
        and robustness in [
            "moderate",
            "weak"
        ]
        and agreement == "agreement"
    ):
        return "moderate_replicated"

    # ------------------------------------------------------------
    # Significant but inconsistent
    # ------------------------------------------------------------

    if (
        fdr_significant
        and (
            agreement != "agreement"
            or robustness == "mixed"
        )
    ):
        return "significant_but_inconsistent"

    # ------------------------------------------------------------
    # Non-significant but directionally informative
    # ------------------------------------------------------------

    if (
        not fdr_significant
        and agreement == "agreement"
    ):
        return "non_significant_agreement"

    # ------------------------------------------------------------
    # Everything else
    # ------------------------------------------------------------

    return "weak_or_unresolved"


scientific[
    "final_evidence_class_v2"
] = scientific.apply(
    classify_evidence,
    axis=1
)


# ================================================================
# SCIENTIFIC PRIORITY SCORE
# ================================================================

def priority_score(row):

    evidence = row[
        "final_evidence_class_v2"
    ]

    abs_d = row.get(
        "abs_cohen_d",
        np.nan
    )

    if pd.isna(abs_d):
        abs_d = 0.0

    try:
        abs_d = float(abs_d)
    except Exception:
        abs_d = 0.0

    if evidence == "strong_replicated":
        base = 8.0

    elif evidence == "moderate_replicated":
        base = 7.0

    elif evidence == "significant_but_inconsistent":
        base = 5.0

    elif evidence == "non_significant_agreement":
        base = 3.0

    else:
        base = 1.0

    # Small continuous effect-size contribution
    score = base + min(abs_d, 1.0)

    return round(score, 6)


scientific[
    "scientific_priority_score_v2"
] = scientific.apply(
    priority_score,
    axis=1
)


# ================================================================
# RANK WITHIN EACH TARGET
# ================================================================

scientific = scientific.sort_values(
    by=[
        "target",
        "scientific_priority_score_v2",
        "abs_cohen_d"
    ],
    ascending=[
        True,
        False,
        False
    ]
).reset_index(drop=True)


scientific[
    "scientific_rank_v2"
] = (
    scientific
    .groupby("target")
    ["scientific_priority_score_v2"]
    .rank(
        method="first",
        ascending=False
    )
    .astype(int)
)


# ================================================================
# FEATURE TYPE LABEL
# ================================================================

scientific[
    "feature_type"
] = "EEG_scientific_feature"


# ================================================================
# FINAL COLUMN ORDER
# ================================================================

preferred_columns = [
    "target",
    "feature",
    "feature_type",

    "mean_difference",
    "cohen_d",
    "abs_cohen_d",

    "p_value",
    "p_fdr",
    "fdr_significant",

    "target_direction",
    "target_direction_consistency",

    "subject_count",
    "target_robustness_class",

    "subject_vs_statistical_direction",

    "scientific_evidence_class",
    "final_evidence_class_v2",

    "scientific_priority_score",
    "scientific_priority_score_v2",

    "scientific_rank",
    "scientific_rank_v2",
]

existing_preferred = [
    c for c in preferred_columns
    if c in scientific.columns
]

remaining_columns = [
    c for c in scientific.columns
    if c not in existing_preferred
]

scientific = scientific[
    existing_preferred
    + remaining_columns
]


# ================================================================
# FINAL DUPLICATE CHECK
# ================================================================

final_duplicates = scientific.duplicated(
    subset=["target", "feature"]
).sum()

if final_duplicates > 0:
    raise RuntimeError(
        "Duplicate target-feature pairs exist in final dataset."
    )


# ================================================================
# FINAL SUMMARY
# ================================================================

print("=" * 80)
print("FINAL SCIENTIFIC EEG EVIDENCE SUMMARY")
print("=" * 80)

print(
    f"Final EEG feature rows:       {len(scientific):,}"
)

print(
    f"Unique EEG features:          "
    f"{scientific['feature'].nunique():,}"
)

print(
    f"Targets:                      "
    f"{scientific['target'].nunique():,}"
)

print(
    f"Duplicate target-feature:     "
    f"{final_duplicates}"
)

print()


# ================================================================
# EVIDENCE CLASS COUNTS
# ================================================================

print("=" * 80)
print("EVIDENCE CLASS DISTRIBUTION")
print("=" * 80)

class_counts = (
    scientific[
        "final_evidence_class_v2"
    ]
    .value_counts()
)

print(class_counts.to_string())

print()


# ================================================================
# TARGET-SPECIFIC COUNTS
# ================================================================

print("=" * 80)
print("TARGET-SPECIFIC EVIDENCE")
print("=" * 80)

target_summary = (
    scientific
    .groupby(
        [
            "target",
            "final_evidence_class_v2"
        ]
    )
    .size()
    .reset_index(
        name="feature_count"
    )
)

print(
    target_summary.to_string(
        index=False
    )
)

print()


# ================================================================
# TOP FEATURES
# ================================================================

print("=" * 80)
print("TOP SCIENTIFIC EEG PERTURBATION / WHAT-IF FEATURES")
print("=" * 80)

top_columns = [
    "target",
    "scientific_rank_v2",
    "feature",
    "mean_difference",
    "cohen_d",
    "p_fdr",
    "target_direction",
    "target_robustness_class",
    "subject_vs_statistical_direction",
    "final_evidence_class_v2",
    "scientific_priority_score_v2",
]

top_columns = [
    c for c in top_columns
    if c in scientific.columns
]

print(
    scientific[top_columns]
    .head(40)
    .to_string(index=False)
)

print()


# ================================================================
# SUMMARY TABLE
# ================================================================

summary_rows = []

for target in sorted(
    scientific["target"].astype(str).unique()
):

    target_df = scientific[
        scientific["target"].astype(str)
        == target
    ].copy()

    row = {
        "target": target,
        "total_EEG_features": len(target_df),
        "FDR_significant": int(
            target_df[
                "fdr_significant"
            ].fillna(False).sum()
        )
        if "fdr_significant" in target_df.columns
        else np.nan,
        "strong_replicated": int(
            (
                target_df[
                    "final_evidence_class_v2"
                ]
                == "strong_replicated"
            ).sum()
        ),
        "moderate_replicated": int(
            (
                target_df[
                    "final_evidence_class_v2"
                ]
                == "moderate_replicated"
            ).sum()
        ),
        "significant_but_inconsistent": int(
            (
                target_df[
                    "final_evidence_class_v2"
                ]
                == "significant_but_inconsistent"
            ).sum()
        ),
        "non_significant_agreement": int(
            (
                target_df[
                    "final_evidence_class_v2"
                ]
                == "non_significant_agreement"
            ).sum()
        ),
        "weak_or_unresolved": int(
            (
                target_df[
                    "final_evidence_class_v2"
                ]
                == "weak_or_unresolved"
            ).sum()
        ),
    }

    summary_rows.append(row)


summary = pd.DataFrame(
    summary_rows
)


# ================================================================
# OVERALL SUMMARY ROW
# ================================================================

overall = {
    "target": "ALL",
    "total_EEG_features": len(scientific),
    "FDR_significant": (
        int(
            scientific[
                "fdr_significant"
            ].fillna(False).sum()
        )
        if "fdr_significant" in scientific.columns
        else np.nan
    ),
    "strong_replicated": int(
        (
            scientific[
                "final_evidence_class_v2"
            ]
            == "strong_replicated"
        ).sum()
    ),
    "moderate_replicated": int(
        (
            scientific[
                "final_evidence_class_v2"
            ]
            == "moderate_replicated"
        ).sum()
    ),
    "significant_but_inconsistent": int(
        (
            scientific[
                "final_evidence_class_v2"
            ]
            == "significant_but_inconsistent"
        ).sum()
    ),
    "non_significant_agreement": int(
        (
            scientific[
                "final_evidence_class_v2"
            ]
            == "non_significant_agreement"
        ).sum()
    ),
    "weak_or_unresolved": int(
        (
            scientific[
                "final_evidence_class_v2"
            ]
            == "weak_or_unresolved"
        ).sum()
    ),
}

summary = pd.concat(
    [
        summary,
        pd.DataFrame([overall])
    ],
    ignore_index=True
)


# ================================================================
# FINAL QC TABLE
# ================================================================

qc_rows = [
    {
        "metric": "input_rows",
        "value": len(df)
    },
    {
        "metric": "final_EEG_rows",
        "value": len(scientific)
    },
    {
        "metric": "input_unique_features",
        "value": df["feature"].nunique()
    },
    {
        "metric": "final_unique_EEG_features",
        "value": scientific["feature"].nunique()
    },
    {
        "metric": "expected_EEG_features",
        "value": len(EEG_FEATURES)
    },
    {
        "metric": "missing_EEG_features",
        "value": len(missing_eeg)
    },
    {
        "metric": "excluded_non_EEG_features",
        "value": len(non_eeg_features)
    },
    {
        "metric": "targets",
        "value": scientific["target"].nunique()
    },
    {
        "metric": "duplicate_target_feature",
        "value": final_duplicates
    },
    {
        "metric": "NaN_numeric_values",
        "value": nan_count
    },
    {
        "metric": "Inf_numeric_values",
        "value": inf_count
    },
    {
        "metric": "memory_cond_excluded",
        "value": int(
            "memory_cond" not in set(
                scientific["feature"]
            )
        )
    },
]


qc = pd.DataFrame(qc_rows)


# ================================================================
# SAVE
# ================================================================

scientific.to_csv(
    OUTPUT_MAIN,
    index=False
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

qc.to_csv(
    OUTPUT_QC,
    index=False
)


# ================================================================
# FINAL STATUS
# ================================================================

print("=" * 80)
print("FINAL SCIENTIFIC EEG PERTURBATION / WHAT-IF EVIDENCE V2 COMPLETE")
print("=" * 80)

print(
    f"EEG scientific features:   "
    f"{scientific['feature'].nunique()}"
)

print(
    f"Targets:                    "
    f"{scientific['target'].nunique()}"
)

print(
    f"Final rows:                 "
    f"{len(scientific):,}"
)

print(
    f"NaN numeric values:         "
    f"{nan_count}"
)

print(
    f"Inf numeric values:         "
    f"{inf_count}"
)

print(
    f"Duplicate target-feature:   "
    f"{final_duplicates}"
)

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_MAIN)
print(OUTPUT_SUMMARY)
print(OUTPUT_QC)

print()
print("=" * 80)
print("STATUS: PASS - FINAL EEG-ONLY SCIENTIFIC EVIDENCE CREATED")
print("=" * 80)