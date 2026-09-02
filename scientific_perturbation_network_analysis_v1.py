from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# SCIENTIFIC EEG PERTURBATION / WHAT-IF NETWORK ANALYSIS V1
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "scientific_interpretation_v1"
)

EVIDENCE_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_evidence_v2"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "network_analysis_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERPRETATION_FILE = (
    INPUT_DIR / "scientific_feature_interpretation_v1.csv"
)

PATTERN_FILE = (
    INPUT_DIR / "scientific_frequency_region_patterns_v1.csv"
)

EVIDENCE_FILE = (
    EVIDENCE_DIR / "final_scientific_perturbation_evidence_v2.csv"
)

# =============================================================================
# HELPERS
# =============================================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found:\n{path}")


def find_column(df, candidates, required=True):
    lower_map = {str(c).lower(): c for c in df.columns}

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


def safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def infer_frequency_region(feature):
    """
    Parse the project's established EEG feature naming convention.
    """

    f = str(feature).lower()

    frequencies = [
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma"
    ]

    regions = [
        "frontal",
        "central",
        "parietal",
        "occipital",
        "temporal",
        "frontoparietal",
        "global"
    ]

    frequency = "unknown"
    region = "global"

    for freq in frequencies:
        if f.startswith(freq + "_") or f == freq:
            frequency = freq
            break

    for reg in regions:
        if reg in f:
            region = reg
            break

    return frequency, region


# =============================================================================
# HEADER
# =============================================================================

print("=" * 90)
print("SCIENTIFIC EEG PERTURBATION / WHAT-IF NETWORK ANALYSIS V1")
print("=" * 90)
print(f"Project root: {BASE}")

# =============================================================================
# INPUT VALIDATION
# =============================================================================

print("\n" + "=" * 90)
print("SEARCHING FOR SCIENTIFIC INTERPRETATION")
print("=" * 90)

require_file(INTERPRETATION_FILE)

print(f"Interpretation input:\n{INTERPRETATION_FILE}")

print("\n" + "=" * 90)
print("SEARCHING FOR FREQUENCY / REGION PATTERNS")
print("=" * 90)

require_file(PATTERN_FILE)

print(f"Pattern input:\n{PATTERN_FILE}")

print("\n" + "=" * 90)
print("SEARCHING FOR FINAL EEG PERTURBATION EVIDENCE")
print("=" * 90)

require_file(EVIDENCE_FILE)

print(f"Evidence input:\n{EVIDENCE_FILE}")

# =============================================================================
# LOAD
# =============================================================================

interpretation = pd.read_csv(INTERPRETATION_FILE)
patterns = pd.read_csv(PATTERN_FILE)
evidence = pd.read_csv(EVIDENCE_FILE)

print("\n" + "=" * 90)
print("INPUT DATA SUMMARY")
print("=" * 90)

print(f"Interpretation rows: {len(interpretation):,}")
print(f"Pattern rows:        {len(patterns):,}")
print(f"Evidence rows:       {len(evidence):,}")

# =============================================================================
# COLUMN DISCOVERY
# =============================================================================

feature_col_i = find_column(
    interpretation,
    ["feature"]
)

target_col_i = find_column(
    interpretation,
    ["target"]
)

effect_col_i = find_column(
    interpretation,
    [
        "mean_difference",
        "effect",
        "effect_size",
        "cohen_d"
    ],
    required=False
)

fdr_col_i = find_column(
    interpretation,
    ["p_fdr", "fdr", "adjusted_p", "q_value"],
    required=False
)

if effect_col_i is None:
    raise KeyError(
        "No effect column was found in the interpretation file."
    )

interpretation = safe_numeric(
    interpretation,
    [effect_col_i] + ([fdr_col_i] if fdr_col_i else [])
)

print("\n" + "=" * 90)
print("INTERPRETATION COLUMN VALIDATION")
print("=" * 90)

print(f"Feature column:       {feature_col_i}")
print(f"Target column:        {target_col_i}")
print(f"Effect column:        {effect_col_i}")
print(f"FDR column:           {fdr_col_i if fdr_col_i else 'not available'}")

# =============================================================================
# BUILD FEATURE METADATA
# =============================================================================

metadata_rows = []

for feature in interpretation[feature_col_i].dropna().unique():

    frequency, region = infer_frequency_region(feature)

    metadata_rows.append(
        {
            "feature": feature,
            "frequency": frequency,
            "region": region
        }
    )

feature_metadata = pd.DataFrame(metadata_rows)

# =============================================================================
# MERGE METADATA
# =============================================================================

analysis = interpretation.merge(
    feature_metadata,
    left_on=feature_col_i,
    right_on="feature",
    how="left"
)

# =============================================================================
# BASIC QC
# =============================================================================

analysis[effect_col_i] = pd.to_numeric(
    analysis[effect_col_i],
    errors="coerce"
)

if fdr_col_i:
    analysis[fdr_col_i] = pd.to_numeric(
        analysis[fdr_col_i],
        errors="coerce"
    )

analysis["absolute_effect"] = analysis[effect_col_i].abs()

if fdr_col_i:
    analysis["fdr_significant"] = (
        analysis[fdr_col_i] < 0.05
    )
else:
    analysis["fdr_significant"] = False

analysis["effect_direction"] = np.where(
    analysis[effect_col_i] > 0,
    "positive",
    np.where(
        analysis[effect_col_i] < 0,
        "negative",
        "zero"
    )
)

# =============================================================================
# FEATURE-LEVEL NETWORK REPRESENTATION
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING FREQUENCY × REGION PERTURBATION NETWORK")
print("=" * 90)

network_rows = []

for (target, frequency, region), group in analysis.groupby(
    [target_col_i, "frequency", "region"],
    dropna=False
):

    valid_effects = group[effect_col_i].dropna()

    if len(valid_effects) == 0:
        continue

    positive_count = int((valid_effects > 0).sum())
    negative_count = int((valid_effects < 0).sum())

    total_directional = positive_count + negative_count

    if total_directional > 0:
        directional_consistency = (
            max(positive_count, negative_count)
            / total_directional
        )
    else:
        directional_consistency = np.nan

    significant = int(
        group["fdr_significant"].sum()
    )

    network_rows.append(
        {
            "target": target,
            "frequency": frequency,
            "region": region,
            "feature_count": len(group),
            "fdr_significant_features": significant,
            "mean_effect": valid_effects.mean(),
            "median_effect": valid_effects.median(),
            "mean_absolute_effect": valid_effects.abs().mean(),
            "max_absolute_effect": valid_effects.abs().max(),
            "positive_features": positive_count,
            "negative_features": negative_count,
            "directional_consistency": directional_consistency
        }
    )

network = pd.DataFrame(network_rows)

# =============================================================================
# NETWORK CLASSIFICATION
# =============================================================================

def classify_network(row):

    sig = row["fdr_significant_features"]
    consistency = row["directional_consistency"]
    magnitude = row["mean_absolute_effect"]

    if sig >= 2 and consistency >= 0.75:
        return "coherent_replicated_pattern"

    if sig >= 1 and consistency >= 0.60:
        return "supported_directional_pattern"

    if sig >= 1:
        return "statistical_pattern"

    if magnitude >= 0.10 and consistency >= 0.60:
        return "effect_candidate"

    return "no_clear_network_pattern"


network["network_class"] = network.apply(
    classify_network,
    axis=1
)

# =============================================================================
# TARGET × FREQUENCY SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING TARGET × FREQUENCY SUMMARY")
print("=" * 90)

frequency_summary = (
    analysis
    .groupby([target_col_i, "frequency"])
    .agg(
        features=("feature", "count"),
        fdr_significant=("fdr_significant", "sum"),
        mean_absolute_effect=("absolute_effect", "mean"),
        max_absolute_effect=("absolute_effect", "max")
    )
    .reset_index()
)

frequency_summary["significant_fraction"] = (
    frequency_summary["fdr_significant"]
    / frequency_summary["features"]
)

# =============================================================================
# TARGET × REGION SUMMARY
# =============================================================================

region_summary = (
    analysis
    .groupby([target_col_i, "region"])
    .agg(
        features=("feature", "count"),
        fdr_significant=("fdr_significant", "sum"),
        mean_absolute_effect=("absolute_effect", "mean"),
        max_absolute_effect=("absolute_effect", "max")
    )
    .reset_index()
)

region_summary["significant_fraction"] = (
    region_summary["fdr_significant"]
    / region_summary["features"]
)

# =============================================================================
# TOP NETWORK PATTERNS
# =============================================================================

print("\n" + "=" * 90)
print("TOP FREQUENCY × REGION PERTURBATION PATTERNS")
print("=" * 90)

top_network = network.sort_values(
    [
        "fdr_significant_features",
        "directional_consistency",
        "mean_absolute_effect"
    ],
    ascending=[False, False, False]
).head(30)

if len(top_network) > 0:
    print(
        top_network[
            [
                "target",
                "frequency",
                "region",
                "feature_count",
                "fdr_significant_features",
                "mean_absolute_effect",
                "directional_consistency",
                "network_class"
            ]
        ].to_string(index=False)
    )

# =============================================================================
# FEATURE DIRECTION CONSISTENCY
# =============================================================================

print("\n" + "=" * 90)
print("FEATURE DIRECTION CONSISTENCY")
print("=" * 90)

feature_consistency_rows = []

for feature, group in analysis.groupby(feature_col_i):

    effects = group[effect_col_i].dropna()

    if len(effects) == 0:
        continue

    positive = int((effects > 0).sum())
    negative = int((effects < 0).sum())

    total = positive + negative

    consistency = (
        max(positive, negative) / total
        if total > 0
        else np.nan
    )

    feature_consistency_rows.append(
        {
            "feature": feature,
            "frequency": group["frequency"].iloc[0],
            "region": group["region"].iloc[0],
            "targets_present": group[target_col_i].nunique(),
            "positive_targets": positive,
            "negative_targets": negative,
            "direction_consistency": consistency,
            "mean_absolute_effect": effects.abs().mean()
        }
    )

feature_consistency = pd.DataFrame(
    feature_consistency_rows
)

# =============================================================================
# CROSS-TARGET AGREEMENT
# =============================================================================

print("\n" + "=" * 90)
print("CROSS-TARGET PERTURBATION AGREEMENT")
print("=" * 90)

pivot = analysis.pivot_table(
    index=feature_col_i,
    columns=target_col_i,
    values=effect_col_i,
    aggfunc="mean"
)

agreement_rows = []

for feature, row in pivot.iterrows():

    remember_effect = (
        row.get("remember", np.nan)
    )

    correct_effect = (
        row.get("correct", np.nan)
    )

    if pd.isna(remember_effect) or pd.isna(correct_effect):
        agreement = "not_available"
    elif (
        np.sign(remember_effect)
        == np.sign(correct_effect)
    ):
        agreement = "same_direction"
    else:
        agreement = "opposite_direction"

    agreement_rows.append(
        {
            "feature": feature,
            "remember_effect": remember_effect,
            "correct_effect": correct_effect,
            "remember_abs_effect": (
                abs(remember_effect)
                if not pd.isna(remember_effect)
                else np.nan
            ),
            "correct_abs_effect": (
                abs(correct_effect)
                if not pd.isna(correct_effect)
                else np.nan
            ),
            "cross_target_agreement": agreement
        }
    )

cross_target = pd.DataFrame(
    agreement_rows
)

# =============================================================================
# SCIENTIFIC PRIORITY
# =============================================================================

network["scientific_network_score"] = (
    network["fdr_significant_features"] * 3
    + network["directional_consistency"].fillna(0) * 2
    + network["mean_absolute_effect"].fillna(0)
)

network = network.sort_values(
    "scientific_network_score",
    ascending=False
).reset_index(drop=True)

network["scientific_network_rank"] = (
    np.arange(len(network)) + 1
)

# =============================================================================
# FINAL QC
# =============================================================================

numeric_columns = [
    effect_col_i,
    "absolute_effect",
    "directional_consistency"
]

numeric_columns = [
    c for c in numeric_columns
    if c in analysis.columns
]

nan_numeric = int(
    analysis[numeric_columns].isna().sum().sum()
)

inf_numeric = int(
    np.isinf(
        analysis[numeric_columns]
        .select_dtypes(include=[np.number])
    ).sum().sum()
)

duplicate_feature_target = int(
    analysis.duplicated(
        subset=[target_col_i, feature_col_i]
    ).sum()
)

print("\n" + "=" * 90)
print("FINAL SCIENTIFIC NETWORK QC")
print("=" * 90)

print(
    f"Targets:                         "
    f"{analysis[target_col_i].nunique()}"
)

print(
    f"Features:                        "
    f"{analysis[feature_col_i].nunique()}"
)

print(
    f"Network rows:                    "
    f"{len(network)}"
)

print(
    f"FDR-significant feature rows:    "
    f"{int(analysis['fdr_significant'].sum())}"
)

print(
    f"NaN numeric values:              "
    f"{nan_numeric}"
)

print(
    f"Inf numeric values:              "
    f"{inf_numeric}"
)

print(
    f"Duplicate target-feature rows:   "
    f"{duplicate_feature_target}"
)

# =============================================================================
# SAVE
# =============================================================================

network_file = (
    OUTPUT_DIR
    / "scientific_perturbation_network_v1.csv"
)

frequency_file = (
    OUTPUT_DIR
    / "scientific_frequency_summary_v1.csv"
)

region_file = (
    OUTPUT_DIR
    / "scientific_region_summary_v1.csv"
)

consistency_file = (
    OUTPUT_DIR
    / "feature_direction_consistency_v1.csv"
)

cross_target_file = (
    OUTPUT_DIR
    / "cross_target_perturbation_agreement_v1.csv"
)

metadata_file = (
    OUTPUT_DIR
    / "feature_frequency_region_metadata_v1.csv"
)

qc_file = (
    OUTPUT_DIR
    / "scientific_perturbation_network_qc_v1.csv"
)

network.to_csv(
    network_file,
    index=False
)

frequency_summary.to_csv(
    frequency_file,
    index=False
)

region_summary.to_csv(
    region_file,
    index=False
)

feature_consistency.to_csv(
    consistency_file,
    index=False
)

cross_target.to_csv(
    cross_target_file,
    index=False
)

feature_metadata.to_csv(
    metadata_file,
    index=False
)

qc = pd.DataFrame(
    [
        {
            "targets": analysis[target_col_i].nunique(),
            "features": analysis[feature_col_i].nunique(),
            "network_rows": len(network),
            "fdr_significant_rows": int(
                analysis["fdr_significant"].sum()
            ),
            "nan_numeric_values": nan_numeric,
            "inf_numeric_values": inf_numeric,
            "duplicate_target_feature": duplicate_feature_target,
            "status": (
                "PASS"
                if (
                    nan_numeric == 0
                    and inf_numeric == 0
                    and duplicate_feature_target == 0
                )
                else "REVIEW_REQUIRED"
            )
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
print("SCIENTIFIC EEG PERTURBATION / WHAT-IF NETWORK ANALYSIS V1 COMPLETE")
print("=" * 90)

print("\nSaved:")
print(network_file)
print(frequency_file)
print(region_file)
print(consistency_file)
print(cross_target_file)
print(metadata_file)
print(qc_file)

print("\n" + "=" * 90)
print(
    "STATUS: "
    + qc.loc[0, "status"]
    + " - SCIENTIFIC PERTURBATION NETWORK ANALYSIS CREATED"
)
print("=" * 90)