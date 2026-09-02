from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# CROSS-TARGET EEG PERTURBATION / WHAT-IF ANALYSIS V1
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_evidence_v2"
)

NETWORK_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "network_analysis_v1"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "cross_target_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVIDENCE_FILE = (
    INPUT_DIR
    / "final_scientific_perturbation_evidence_v2.csv"
)

NETWORK_FILE = (
    NETWORK_DIR
    / "scientific_perturbation_network_v1.csv"
)

# =============================================================================
# HELPERS
# =============================================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file was not found:\n{path}"
        )


def find_column(df, candidates, required=True):
    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise KeyError(
            f"Required column not found.\n"
            f"Candidates: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


# =============================================================================
# HEADER
# =============================================================================

print("=" * 90)
print("CROSS-TARGET EEG PERTURBATION / WHAT-IF ANALYSIS V1")
print("=" * 90)

print(f"Project root: {BASE}")

# =============================================================================
# INPUT SEARCH
# =============================================================================

print("\n" + "=" * 90)
print("SEARCHING FOR FINAL EEG PERTURBATION EVIDENCE")
print("=" * 90)

require_file(EVIDENCE_FILE)

print(f"Evidence input:\n{EVIDENCE_FILE}")

print("\n" + "=" * 90)
print("SEARCHING FOR NETWORK-LEVEL PERTURBATION RESULTS")
print("=" * 90)

require_file(NETWORK_FILE)

print(f"Network input:\n{NETWORK_FILE}")

# =============================================================================
# LOAD
# =============================================================================

evidence = pd.read_csv(EVIDENCE_FILE)
network = pd.read_csv(NETWORK_FILE)

print("\n" + "=" * 90)
print("INPUT DATA SUMMARY")
print("=" * 90)

print(f"Evidence rows:     {len(evidence):,}")
print(f"Evidence columns:  {len(evidence.columns):,}")
print(f"Network rows:      {len(network):,}")
print(f"Network columns:   {len(network.columns):,}")

# =============================================================================
# COLUMN DISCOVERY
# =============================================================================

feature_col = find_column(
    evidence,
    ["feature"]
)

target_col = find_column(
    evidence,
    ["target"]
)

effect_col = find_column(
    evidence,
    [
        "mean_difference",
        "cohen_d",
        "effect_size",
        "effect"
    ]
)

fdr_col = find_column(
    evidence,
    [
        "p_fdr",
        "fdr",
        "adjusted_p",
        "q_value"
    ],
    required=False
)

print("\n" + "=" * 90)
print("COLUMN VALIDATION")
print("=" * 90)

print(f"Feature column:    {feature_col}")
print(f"Target column:     {target_col}")
print(f"Effect column:     {effect_col}")
print(
    f"FDR column:        "
    f"{fdr_col if fdr_col else 'not available'}"
)

# =============================================================================
# TARGET VALIDATION
# =============================================================================

targets = sorted(
    evidence[target_col]
    .dropna()
    .astype(str)
    .str.lower()
    .unique()
)

print("\n" + "=" * 90)
print("TARGET VALIDATION")
print("=" * 90)

print("Targets found:")
for target in targets:
    print(f"  {target}")

if "remember" not in targets:
    raise ValueError(
        "Target 'remember' was not found."
    )

if "correct" not in targets:
    raise ValueError(
        "Target 'correct' was not found."
    )

# Normalize target values
evidence["_target_normalized"] = (
    evidence[target_col]
    .astype(str)
    .str.lower()
)

# =============================================================================
# NUMERIC CONVERSION
# =============================================================================

evidence[effect_col] = pd.to_numeric(
    evidence[effect_col],
    errors="coerce"
)

if fdr_col:
    evidence[fdr_col] = pd.to_numeric(
        evidence[fdr_col],
        errors="coerce"
    )

# =============================================================================
# DUPLICATE CHECK
# =============================================================================

duplicate_count = int(
    evidence.duplicated(
        subset=[
            "_target_normalized",
            feature_col
        ]
    ).sum()
)

print("\n" + "=" * 90)
print("DUPLICATE TARGET-FEATURE CHECK")
print("=" * 90)

print(
    f"Duplicate target-feature rows: "
    f"{duplicate_count}"
)

if duplicate_count > 0:
    raise ValueError(
        "Duplicate target-feature rows detected."
    )

# =============================================================================
# BUILD REMEMBER / CORRECT PIVOT
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING CROSS-TARGET EFFECT MATRIX")
print("=" * 90)

effect_pivot = evidence.pivot_table(
    index=feature_col,
    columns="_target_normalized",
    values=effect_col,
    aggfunc="mean"
)

fdr_pivot = None

if fdr_col:
    fdr_pivot = evidence.pivot_table(
        index=feature_col,
        columns="_target_normalized",
        values=fdr_col,
        aggfunc="min"
    )

# =============================================================================
# CROSS-TARGET COMPARISON
# =============================================================================

comparison_rows = []

for feature in effect_pivot.index:

    remember_effect = effect_pivot.loc[
        feature
    ].get("remember", np.nan)

    correct_effect = effect_pivot.loc[
        feature
    ].get("correct", np.nan)

    if pd.isna(remember_effect) or pd.isna(correct_effect):
        continue

    remember_abs = abs(
        remember_effect
    )

    correct_abs = abs(
        correct_effect
    )

    effect_difference = (
        correct_effect - remember_effect
    )

    absolute_effect_difference = abs(
        correct_effect - remember_effect
    )

    if (
        np.sign(remember_effect)
        == np.sign(correct_effect)
    ):
        direction_relationship = (
            "same_direction"
        )
    else:
        direction_relationship = (
            "opposite_direction"
        )

    if (
        remember_abs > correct_abs
        and remember_abs >= 0.05
    ):
        target_specificity = (
            "remember_dominant"
        )

    elif (
        correct_abs > remember_abs
        and correct_abs >= 0.05
    ):
        target_specificity = (
            "correct_dominant"
        )

    elif (
        remember_abs >= 0.05
        and correct_abs >= 0.05
    ):
        target_specificity = (
            "shared_effect"
        )

    else:
        target_specificity = (
            "weak_both_targets"
        )

    if (
        direction_relationship
        == "opposite_direction"
        and absolute_effect_difference >= 0.10
    ):
        cross_target_pattern = (
            "target_dissociated"
        )

    elif (
        direction_relationship
        == "same_direction"
        and min(
            remember_abs,
            correct_abs
        ) >= 0.05
    ):
        cross_target_pattern = (
            "target_shared"
        )

    elif (
        remember_abs >= 0.10
        or correct_abs >= 0.10
    ):
        cross_target_pattern = (
            "target_preferential"
        )

    else:
        cross_target_pattern = (
            "weak_cross_target_pattern"
        )

    # FDR information
    if fdr_pivot is not None:

        remember_fdr = fdr_pivot.loc[
            feature
        ].get("remember", np.nan)

        correct_fdr = fdr_pivot.loc[
            feature
        ].get("correct", np.nan)

        remember_sig = (
            not pd.isna(remember_fdr)
            and remember_fdr < 0.05
        )

        correct_sig = (
            not pd.isna(correct_fdr)
            and correct_fdr < 0.05
        )

    else:

        remember_fdr = np.nan
        correct_fdr = np.nan

        remember_sig = False
        correct_sig = False

    if remember_sig and correct_sig:
        significance_pattern = (
            "significant_both_targets"
        )

    elif remember_sig:
        significance_pattern = (
            "significant_remember_only"
        )

    elif correct_sig:
        significance_pattern = (
            "significant_correct_only"
        )

    else:
        significance_pattern = (
            "not_fdr_significant"
        )

    comparison_rows.append(
        {
            "feature": feature,
            "remember_effect": remember_effect,
            "correct_effect": correct_effect,
            "remember_abs_effect": remember_abs,
            "correct_abs_effect": correct_abs,
            "effect_difference_correct_minus_remember":
                effect_difference,
            "absolute_cross_target_difference":
                absolute_effect_difference,
            "direction_relationship":
                direction_relationship,
            "target_specificity":
                target_specificity,
            "cross_target_pattern":
                cross_target_pattern,
            "remember_p_fdr":
                remember_fdr,
            "correct_p_fdr":
                correct_fdr,
            "remember_fdr_significant":
                remember_sig,
            "correct_fdr_significant":
                correct_sig,
            "significance_pattern":
                significance_pattern
        }
    )

comparison = pd.DataFrame(
    comparison_rows
)

# =============================================================================
# SCIENTIFIC TARGET DISSOCIATION SCORE
# =============================================================================

comparison["target_dissociation_score"] = (
    comparison[
        "absolute_cross_target_difference"
    ]
    + comparison[
        "remember_abs_effect"
    ]
    + comparison[
        "correct_abs_effect"
    ]
)

comparison = comparison.sort_values(
    [
        "target_dissociation_score",
        "absolute_cross_target_difference"
    ],
    ascending=[False, False]
).reset_index(drop=True)

comparison["cross_target_rank"] = (
    np.arange(len(comparison)) + 1
)

# =============================================================================
# NETWORK INTEGRATION
# =============================================================================

network_cols = [
    c for c in [
        "feature",
        "frequency",
        "region"
    ]
    if c in network.columns
]

if len(network_cols) >= 1:

    network_feature_col = (
        "feature"
        if "feature" in network.columns
        else None
    )

    if network_feature_col:

        network_subset = network[
            network_cols
        ].drop_duplicates(
            subset=["feature"]
        )

        comparison = comparison.merge(
            network_subset,
            on="feature",
            how="left"
        )

# =============================================================================
# TARGET-SPECIFIC TABLES
# =============================================================================

remember_dominant = comparison[
    comparison["target_specificity"]
    == "remember_dominant"
].copy()

correct_dominant = comparison[
    comparison["target_specificity"]
    == "correct_dominant"
].copy()

shared_effects = comparison[
    comparison["target_specificity"]
    == "shared_effect"
].copy()

dissociated = comparison[
    comparison["cross_target_pattern"]
    == "target_dissociated"
].copy()

# =============================================================================
# PRINT TOP RESULTS
# =============================================================================

print("\n" + "=" * 90)
print("TOP CROSS-TARGET PERTURBATION PATTERNS")
print("=" * 90)

display_cols = [
    "feature",
    "remember_effect",
    "correct_effect",
    "remember_abs_effect",
    "correct_abs_effect",
    "direction_relationship",
    "target_specificity",
    "significance_pattern",
    "cross_target_pattern",
    "target_dissociation_score"
]

if "frequency" in comparison.columns:
    display_cols.insert(
        1,
        "frequency"
    )

if "region" in comparison.columns:
    display_cols.insert(
        2,
        "region"
    )

print(
    comparison[
        display_cols
    ].head(40).to_string(index=False)
)

# =============================================================================
# CATEGORY COUNTS
# =============================================================================

print("\n" + "=" * 90)
print("CROSS-TARGET CLASSIFICATION")
print("=" * 90)

print(
    comparison[
        "target_specificity"
    ].value_counts()
)

print("\nCross-target pattern:")
print(
    comparison[
        "cross_target_pattern"
    ].value_counts()
)

print("\nSignificance pattern:")
print(
    comparison[
        "significance_pattern"
    ].value_counts()
)

print("\nDirection relationship:")
print(
    comparison[
        "direction_relationship"
    ].value_counts()
)

# =============================================================================
# FINAL QC
# =============================================================================

numeric_columns = [
    "remember_effect",
    "correct_effect",
    "remember_abs_effect",
    "correct_abs_effect",
    "effect_difference_correct_minus_remember",
    "absolute_cross_target_difference",
    "target_dissociation_score"
]

nan_numeric = int(
    comparison[numeric_columns]
    .isna()
    .sum()
    .sum()
)

inf_numeric = int(
    np.isinf(
        comparison[numeric_columns]
        .to_numpy(
            dtype=float
        )
    ).sum()
)

duplicate_feature = int(
    comparison.duplicated(
        subset=["feature"]
    ).sum()
)

print("\n" + "=" * 90)
print("FINAL CROSS-TARGET SCIENTIFIC QC")
print("=" * 90)

print(
    f"Features compared:             "
    f"{len(comparison)}"
)

print(
    f"Remember-dominant features:    "
    f"{len(remember_dominant)}"
)

print(
    f"Correct-dominant features:     "
    f"{len(correct_dominant)}"
)

print(
    f"Shared-effect features:        "
    f"{len(shared_effects)}"
)

print(
    f"Target-dissociated features:   "
    f"{len(dissociated)}"
)

print(
    f"NaN numeric values:             "
    f"{nan_numeric}"
)

print(
    f"Inf numeric values:             "
    f"{inf_numeric}"
)

print(
    f"Duplicate feature rows:         "
    f"{duplicate_feature}"
)

# =============================================================================
# SAVE
# =============================================================================

comparison_file = (
    OUTPUT_DIR
    / "cross_target_perturbation_comparison_v1.csv"
)

remember_file = (
    OUTPUT_DIR
    / "remember_dominant_perturbations_v1.csv"
)

correct_file = (
    OUTPUT_DIR
    / "correct_dominant_perturbations_v1.csv"
)

shared_file = (
    OUTPUT_DIR
    / "shared_target_perturbations_v1.csv"
)

dissociated_file = (
    OUTPUT_DIR
    / "target_dissociated_perturbations_v1.csv"
)

qc_file = (
    OUTPUT_DIR
    / "cross_target_perturbation_qc_v1.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)

remember_dominant.to_csv(
    remember_file,
    index=False
)

correct_dominant.to_csv(
    correct_file,
    index=False
)

shared_effects.to_csv(
    shared_file,
    index=False
)

dissociated.to_csv(
    dissociated_file,
    index=False
)

status = (
    "PASS"
    if (
        nan_numeric == 0
        and inf_numeric == 0
        and duplicate_feature == 0
        and len(comparison) > 0
    )
    else "REVIEW_REQUIRED"
)

qc = pd.DataFrame(
    [
        {
            "features_compared":
                len(comparison),
            "remember_dominant":
                len(remember_dominant),
            "correct_dominant":
                len(correct_dominant),
            "shared_effects":
                len(shared_effects),
            "target_dissociated":
                len(dissociated),
            "nan_numeric":
                nan_numeric,
            "inf_numeric":
                inf_numeric,
            "duplicate_feature":
                duplicate_feature,
            "status":
                status
        }
    ]
)

qc.to_csv(
    qc_file,
    index=False
)

# =============================================================================
# COMPLETION
# =============================================================================

print("\n" + "=" * 90)
print("CROSS-TARGET EEG PERTURBATION / WHAT-IF ANALYSIS V1 COMPLETE")
print("=" * 90)

print("\nSaved:")
print(comparison_file)
print(remember_file)
print(correct_file)
print(shared_file)
print(dissociated_file)
print(qc_file)

print("\n" + "=" * 90)
print(
    f"STATUS: {status} - "
    "CROSS-TARGET PERTURBATION ANALYSIS CREATED"
)
print("=" * 90)