# ================================================================
# SCIENTIFIC PERTURBATION / WHAT-IF INTERPRETATION V1
# ================================================================
# Purpose:
#   Convert the final EEG-only perturbation evidence into a
#   scientifically interpretable frequency × region evidence layer.
#
# IMPORTANT:
#   - This is NOT a causal inference analysis.
#   - This does NOT train another ML model.
#   - Target-derived variables are excluded.
#   - memory_cond is excluded.
#   - Results are based only on the final EEG scientific evidence.
# ================================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ================================================================
# PROJECT PATHS
# ================================================================

BASE = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_evidence_v2"
    / "final_scientific_perturbation_evidence_v2.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "scientific_interpretation_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FEATURE = (
    OUTPUT_DIR
    / "scientific_feature_interpretation_v1.csv"
)

OUTPUT_TARGET = (
    OUTPUT_DIR
    / "scientific_target_interpretation_v1.csv"
)

OUTPUT_PATTERN = (
    OUTPUT_DIR
    / "scientific_frequency_region_patterns_v1.csv"
)

OUTPUT_QC = (
    OUTPUT_DIR
    / "scientific_interpretation_qc_v1.csv"
)


# ================================================================
# START
# ================================================================

print("=" * 80)
print("SCIENTIFIC PERTURBATION / WHAT-IF INTERPRETATION V1")
print("=" * 80)

print(f"Project root: {BASE}")
print()


# ================================================================
# LOAD
# ================================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Final EEG evidence file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

print("=" * 80)
print("INPUT VALIDATION")
print("=" * 80)

print(f"Rows:       {len(df):,}")
print(f"Columns:    {len(df.columns):,}")

if "target" not in df.columns:
    raise RuntimeError(
        "Required column 'target' is missing."
    )

if "feature" not in df.columns:
    raise RuntimeError(
        "Required column 'feature' is missing."
    )

print("Required columns: PASS")
print()


# ================================================================
# DUPLICATE CHECK
# ================================================================

duplicates = df.duplicated(
    subset=["target", "feature"]
).sum()

print("=" * 80)
print("DUPLICATE CHECK")
print("=" * 80)

print(
    f"Duplicate target-feature rows: {duplicates}"
)

if duplicates > 0:
    raise RuntimeError(
        "Duplicate target-feature rows detected."
    )

print("PASS")
print()


# ================================================================
# EXCLUDE NON-EEG FEATURES
# ================================================================

NON_EEG_FEATURES = {
    "memory_cond",
    "target_remember",
    "target_correct",
    "remember",
    "correct",
    "target"
}

before_features = df["feature"].nunique()

df = df[
    ~df["feature"]
    .astype(str)
    .isin(NON_EEG_FEATURES)
].copy()

after_features = df["feature"].nunique()

print("=" * 80)
print("EEG-ONLY FILTER")
print("=" * 80)

print(
    f"Features before filtering: {before_features}"
)

print(
    f"Features after filtering:  {after_features}"
)

print()


# ================================================================
# NUMERIC QC
# ================================================================

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()

nan_count = int(
    df[numeric_columns]
    .isna()
    .sum()
    .sum()
)

inf_count = int(
    np.isinf(
        df[numeric_columns].to_numpy(
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
        "Infinite numeric values detected."
    )

print()


# ================================================================
# FEATURE PARSING
# ================================================================

BANDS = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma"
]

REGIONS = [
    "frontal",
    "central",
    "parietal",
    "occipital",
    "temporal",
    "frontoparietal"
]


def identify_band(feature):

    feature = str(feature)

    for band in BANDS:
        if feature.startswith(band + "_"):
            return band

    return "other"


def identify_region(feature):

    feature = str(feature)

    # Frontoparietal first because it contains
    # frontal + parietal terminology.
    if "frontoparietal" in feature:
        return "frontoparietal"

    for region in REGIONS:
        if region in feature:
            return region

    return "global"


def identify_measure(feature):

    feature = str(feature)

    if "_ratio" in feature:
        return "ratio"

    if "_diff" in feature:
        return "difference"

    if feature.endswith("_rel"):
        return "relative_power"

    if feature.endswith("_abs"):
        return "absolute_power"

    if (
        "_frontal" in feature
        or "_central" in feature
        or "_parietal" in feature
        or "_occipital" in feature
        or "_temporal" in feature
    ):
        return "regional_power"

    return "other"


df["frequency_band"] = df[
    "feature"
].apply(
    identify_band
)

df["brain_region"] = df[
    "feature"
].apply(
    identify_region
)

df["feature_measure"] = df[
    "feature"
].apply(
    identify_measure
)


# ================================================================
# EFFECT MAGNITUDE
# ================================================================

if "abs_cohen_d" in df.columns:

    df["abs_cohen_d_clean"] = pd.to_numeric(
        df["abs_cohen_d"],
        errors="coerce"
    )

else:

    if "cohen_d" in df.columns:
        df["abs_cohen_d_clean"] = (
            pd.to_numeric(
                df["cohen_d"],
                errors="coerce"
            )
            .abs()
        )

    else:
        df["abs_cohen_d_clean"] = np.nan


def effect_strength(d):

    if pd.isna(d):
        return "not_available"

    d = abs(float(d))

    if d >= 0.8:
        return "large"

    if d >= 0.5:
        return "medium"

    if d >= 0.2:
        return "small"

    return "negligible"


df["effect_strength_v1"] = (
    df["abs_cohen_d_clean"]
    .apply(effect_strength)
)


# ================================================================
# FDR STATUS
# ================================================================

if "fdr_significant" in df.columns:

    df["fdr_significant_clean"] = (
        df["fdr_significant"]
        .fillna(False)
        .astype(bool)
    )

else:

    if "p_fdr" in df.columns:

        df["p_fdr_numeric"] = pd.to_numeric(
            df["p_fdr"],
            errors="coerce"
        )

        df["fdr_significant_clean"] = (
            df["p_fdr_numeric"] < 0.05
        )

    else:

        df["fdr_significant_clean"] = False


# ================================================================
# EVIDENCE CLASS
# ================================================================

if "final_evidence_class_v2" in df.columns:

    df["evidence_class"] = (
        df["final_evidence_class_v2"]
        .astype(str)
    )

elif "scientific_evidence_class" in df.columns:

    df["evidence_class"] = (
        df["scientific_evidence_class"]
        .astype(str)
    )

else:

    df["evidence_class"] = "unknown"


# ================================================================
# SCIENTIFIC INTERPRETATION
# ================================================================

def interpret_feature(row):

    evidence = str(
        row["evidence_class"]
    )

    direction = str(
        row.get(
            "target_direction",
            ""
        )
    )

    band = str(
        row["frequency_band"]
    )

    region = str(
        row["brain_region"]
    )

    strength = str(
        row["effect_strength_v1"]
    )

    if evidence == "strong_replicated":

        level = "high_priority"

    elif evidence == "moderate_replicated":

        level = "replicated_candidate"

    elif evidence == "significant_but_inconsistent":

        level = "statistically_supported_but_inconsistent"

    elif evidence == "non_significant_agreement":

        level = "directional_candidate"

    else:

        level = "weak_or_unresolved"

    return (
        f"{level}; "
        f"{band}-band; "
        f"{region}-region; "
        f"{direction}; "
        f"{strength}-effect"
    )


df[
    "scientific_interpretation_v1"
] = df.apply(
    interpret_feature,
    axis=1
)


# ================================================================
# FEATURE-LEVEL TABLE
# ================================================================

feature_columns = [
    "target",
    "feature",
    "frequency_band",
    "brain_region",
    "feature_measure",

    "mean_difference",
    "cohen_d",
    "abs_cohen_d_clean",

    "p_value",
    "p_fdr",
    "fdr_significant_clean",

    "target_direction",
    "target_direction_consistency",

    "subject_count",
    "target_robustness_class",
    "subject_vs_statistical_direction",

    "evidence_class",
    "effect_strength_v1",

    "scientific_priority_score_v2",
    "scientific_rank_v2",

    "scientific_interpretation_v1"
]

feature_columns = [
    c for c in feature_columns
    if c in df.columns
]

feature_table = df[
    feature_columns
].copy()

feature_table = feature_table.sort_values(
    by=[
        "target",
        "scientific_priority_score_v2"
    ],
    ascending=[
        True,
        False
    ]
)


# ================================================================
# TARGET-LEVEL SUMMARY
# ================================================================

target_rows = []

for target in sorted(
    df["target"]
    .astype(str)
    .unique()
):

    sub = df[
        df["target"].astype(str)
        == target
    ].copy()

    row = {
        "target": target,
        "features": sub["feature"].nunique(),

        "FDR_significant": int(
            sub[
                "fdr_significant_clean"
            ].sum()
        ),

        "strong_replicated": int(
            (
                sub["evidence_class"]
                == "strong_replicated"
            ).sum()
        ),

        "moderate_replicated": int(
            (
                sub["evidence_class"]
                == "moderate_replicated"
            ).sum()
        ),

        "significant_but_inconsistent": int(
            (
                sub["evidence_class"]
                == "significant_but_inconsistent"
            ).sum()
        ),

        "directional_candidate": int(
            (
                sub["evidence_class"]
                == "non_significant_agreement"
            ).sum()
        ),

        "weak_or_unresolved": int(
            (
                sub["evidence_class"]
                == "weak_or_unresolved"
            ).sum()
        ),

        "mean_abs_cohen_d": (
            sub["abs_cohen_d_clean"]
            .mean()
        )
    }

    target_rows.append(row)


target_summary = pd.DataFrame(
    target_rows
)


# ================================================================
# FREQUENCY × REGION PATTERNS
# ================================================================

pattern = (
    df
    .groupby(
        [
            "target",
            "frequency_band",
            "brain_region"
        ]
    )
    .agg(
        feature_count=(
            "feature",
            "nunique"
        ),

        fdr_significant_count=(
            "fdr_significant_clean",
            "sum"
        ),

        mean_abs_cohen_d=(
            "abs_cohen_d_clean",
            "mean"
        ),

        max_abs_cohen_d=(
            "abs_cohen_d_clean",
            "max"
        ),

        positive_fraction=(
            "target_direction",
            lambda x: (
                x.astype(str)
                .str.lower()
                .eq("positive")
                .mean()
            )
        ),

        negative_fraction=(
            "target_direction",
            lambda x: (
                x.astype(str)
                .str.lower()
                .eq("negative")
                .mean()
            )
        )
    )
    .reset_index()
)


def classify_pattern(row):

    sig = row[
        "fdr_significant_count"
    ]

    count = row[
        "feature_count"
    ]

    mean_d = row[
        "mean_abs_cohen_d"
    ]

    if sig >= 2 and mean_d >= 0.2:
        return "strong_pattern"

    if sig >= 1 and mean_d >= 0.1:
        return "candidate_pattern"

    if sig >= 1:
        return "statistical_pattern"

    if count >= 2:
        return "weak_directional_pattern"

    return "no_clear_pattern"


pattern[
    "pattern_class_v1"
] = pattern.apply(
    classify_pattern,
    axis=1
)


pattern = pattern.sort_values(
    by=[
        "target",
        "fdr_significant_count",
        "mean_abs_cohen_d"
    ],
    ascending=[
        True,
        False,
        False
    ]
)


# ================================================================
# PRINT TOP RESULTS
# ================================================================

print("=" * 80)
print("TOP SCIENTIFIC INTERPRETATION RESULTS")
print("=" * 80)

display_columns = [
    "target",
    "feature",
    "frequency_band",
    "brain_region",
    "mean_difference",
    "cohen_d",
    "p_fdr",
    "evidence_class",
    "effect_strength_v1",
    "scientific_interpretation_v1"
]

display_columns = [
    c for c in display_columns
    if c in feature_table.columns
]

print(
    feature_table[
        display_columns
    ]
    .head(40)
    .to_string(index=False)
)

print()


# ================================================================
# PATTERN RESULTS
# ================================================================

print("=" * 80)
print("FREQUENCY × REGION PATTERNS")
print("=" * 80)

print(
    pattern
    .head(40)
    .to_string(index=False)
)

print()


# ================================================================
# FINAL QC
# ================================================================

final_duplicates = feature_table.duplicated(
    subset=[
        "target",
        "feature"
    ]
).sum()

qc = pd.DataFrame([
    {
        "metric": "input_rows",
        "value": len(pd.read_csv(INPUT))
    },
    {
        "metric": "final_rows",
        "value": len(feature_table)
    },
    {
        "metric": "unique_features",
        "value": feature_table[
            "feature"
        ].nunique()
    },
    {
        "metric": "targets",
        "value": feature_table[
            "target"
        ].nunique()
    },
    {
        "metric": "duplicate_target_feature",
        "value": final_duplicates
    },
    {
        "metric": "NaN_numeric_values",
        "value": int(
            feature_table
            .select_dtypes(
                include=[np.number]
            )
            .isna()
            .sum()
            .sum()
        )
    },
    {
        "metric": "Inf_numeric_values",
        "value": int(
            np.isinf(
                feature_table
                .select_dtypes(
                    include=[np.number]
                )
                .to_numpy(
                    dtype=float
                )
            ).sum()
        )
    },
    {
        "metric": "FDR_significant_total",
        "value": int(
            df[
                "fdr_significant_clean"
            ].sum()
        )
    }
])


# ================================================================
# SAVE
# ================================================================

feature_table.to_csv(
    OUTPUT_FEATURE,
    index=False
)

target_summary.to_csv(
    OUTPUT_TARGET,
    index=False
)

pattern.to_csv(
    OUTPUT_PATTERN,
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
print("SCIENTIFIC PERTURBATION / WHAT-IF INTERPRETATION V1 COMPLETE")
print("=" * 80)

print(
    f"Features analyzed:        "
    f"{feature_table['feature'].nunique()}"
)

print(
    f"Targets analyzed:         "
    f"{feature_table['target'].nunique()}"
)

print(
    f"Final rows:               "
    f"{len(feature_table)}"
)

print(
    f"FDR significant:          "
    f"{int(df['fdr_significant_clean'].sum())}"
)

print(
    f"Duplicate target-feature: "
    f"{final_duplicates}"
)

print()

print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_FEATURE)
print(OUTPUT_TARGET)
print(OUTPUT_PATTERN)
print(OUTPUT_QC)

print()

if (
    final_duplicates == 0
    and qc.loc[
        qc["metric"]
        == "Inf_numeric_values",
        "value"
    ].iloc[0] == 0
):

    print(
        "STATUS: PASS - SCIENTIFIC INTERPRETATION LAYER CREATED"
    )

else:

    print(
        "STATUS: REVIEW_REQUIRED"
    )

print("=" * 80)