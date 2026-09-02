from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "ml_ready_v2"
    / "ml_ready_dataset_v2.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "ml_ready_v2"
    / "feature_selection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DATASET = (
    OUTPUT_DIR
    / "ml_ready_dataset_v2_selected.csv"
)

OUTPUT_FEATURES = (
    OUTPUT_DIR
    / "selected_scientific_features_v2.csv"
)

OUTPUT_QC = (
    OUTPUT_DIR
    / "feature_selection_v2_qc.csv"
)

OUTPUT_CORR = (
    OUTPUT_DIR
    / "feature_correlation_v2.csv"
)


# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("SCIENTIFIC FEATURE SELECTION V2")
print("=" * 80)

df = pd.read_csv(INPUT)

print(f"Input rows:       {len(df):,}")
print(f"Input columns:    {len(df.columns):,}")


# =============================================================================
# REQUIRED KEYS
# =============================================================================

KEY_COLUMNS = [
    "subject",
    "run",
    "epoch",
]

missing_keys = [
    c for c in KEY_COLUMNS
    if c not in df.columns
]

if missing_keys:
    raise RuntimeError(
        f"Missing required key columns: {missing_keys}"
    )


# =============================================================================
# EXCLUDE NON-SCIENTIFIC / TARGET-DERIVED VARIABLES
# =============================================================================

EXCLUDE_COLUMNS = {
    # identifiers
    "subject",
    "run",
    "epoch",
    "trial",
    "file",

    # split information
    "split",

    # targets
    "target",
    "target_label",
    "target_remember",
    "target_correct",

    # behavioral variables
    "is_correct",
    "is_remembered",
    "is_ignored",

    "n_epochs",
    "has_fixation",
    "has_letters",
    "has_wm_period",
    "has_left_click",
    "has_right_click",
    "has_correct_feedback",
    "has_incorrect_feedback",
    "n_letters",
    "event_sequence",
    "feedback",
    "response_type",
    "complete_trial",
    "event_source",
    "memory_cond",
    "remember_count",
    "ignore_count",
    "remember_letters",
    "ignore_letters",
    "probe_type",
    "probe_letter",
    "behavior_outcome",
    "behavior_label",
    "alignment_status",
}


# =============================================================================
# NUMERIC SCIENTIFIC FEATURES
# =============================================================================

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()

scientific_features = [
    c
    for c in numeric_columns
    if c not in EXCLUDE_COLUMNS
]


print()
print("=" * 80)
print("INITIAL FEATURE INVENTORY")
print("=" * 80)

print(
    f"Numeric columns:       {len(numeric_columns)}"
)

print(
    f"Scientific features:   {len(scientific_features)}"
)

print()
print(scientific_features)


# =============================================================================
# REMOVE CONSTANT FEATURES
# =============================================================================

print()
print("=" * 80)
print("CONSTANT FEATURE REMOVAL")
print("=" * 80)

constant_features = []

for feature in scientific_features:

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    unique_count = values.nunique(
        dropna=True
    )

    if unique_count <= 1:
        constant_features.append(feature)


print(
    f"Constant features found: {len(constant_features)}"
)

if constant_features:
    print(constant_features)


features_after_constant_removal = [
    f
    for f in scientific_features
    if f not in constant_features
]


print(
    f"Remaining features:      "
    f"{len(features_after_constant_removal)}"
)


# =============================================================================
# FEATURE MATRIX
# =============================================================================

X = df[
    features_after_constant_removal
].apply(
    pd.to_numeric,
    errors="coerce"
)


# =============================================================================
# BASIC QC
# =============================================================================

nan_count = int(
    X.isna().sum().sum()
)

inf_count = int(
    np.isinf(
        X.to_numpy(
            dtype=np.float64,
            copy=True
        )
    ).sum()
)

print()
print("=" * 80)
print("FEATURE QC")
print("=" * 80)

print(
    f"NaN values:       {nan_count:,}"
)

print(
    f"Inf values:       {inf_count:,}"
)


if nan_count > 0:
    raise RuntimeError(
        "NaN values detected. STOP."
    )

if inf_count > 0:
    raise RuntimeError(
        "Inf values detected. STOP."
    )


# =============================================================================
# CORRELATION MATRIX
# =============================================================================

print()
print("=" * 80)
print("FEATURE CORRELATION")
print("=" * 80)

corr_df = X.corr(
    method="pearson"
)

# Explicit writable copy.
corr_matrix = np.array(
    corr_df.to_numpy(
        dtype=np.float64,
        copy=True
    ),
    dtype=np.float64,
    copy=True
)

corr_matrix.setflags(
    write=True
)

np.fill_diagonal(
    corr_matrix,
    np.nan
)


# =============================================================================
# SAVE CORRELATION MATRIX
# =============================================================================

corr_output = corr_df.copy()

corr_output.to_csv(
    OUTPUT_CORR
)


# =============================================================================
# REDUNDANCY CONTROL
# =============================================================================

CORRELATION_THRESHOLD = 0.95

print()
print("=" * 80)
print("REDUNDANCY CONTROL")
print("=" * 80)

print(
    f"Correlation threshold: |r| >= "
    f"{CORRELATION_THRESHOLD}"
)


# -------------------------------------------------------------------------
# Important scientific rule:
#
# We do NOT use target_remember or target_correct to decide which feature
# survives.
#
# When two scientific features are highly correlated, keep the first
# feature according to the original scientific feature order and remove
# the later redundant feature.
#
# This keeps the procedure target-independent.
# -------------------------------------------------------------------------

features = features_after_constant_removal

features_to_remove = set()

redundancy_records = []

for i in range(len(features)):

    if features[i] in features_to_remove:
        continue

    for j in range(i + 1, len(features)):

        if features[j] in features_to_remove:
            continue

        r = corr_matrix[i, j]

        if not np.isfinite(r):
            continue

        if abs(r) >= CORRELATION_THRESHOLD:

            keep_feature = features[i]
            remove_feature = features[j]

            features_to_remove.add(
                remove_feature
            )

            redundancy_records.append(
                {
                    "feature_kept": keep_feature,
                    "feature_removed": remove_feature,
                    "correlation": float(r),
                    "abs_correlation": float(abs(r)),
                    "threshold": CORRELATION_THRESHOLD,
                }
            )


redundancy_df = pd.DataFrame(
    redundancy_records
)


selected_features = [
    f
    for f in features
    if f not in features_to_remove
]


print(
    f"Highly correlated pairs: "
    f"{len(redundancy_df):,}"
)

print(
    f"Features removed:        "
    f"{len(features_to_remove):,}"
)

print(
    f"Final selected features: "
    f"{len(selected_features):,}"
)


if len(redundancy_df) > 0:

    print()
    print("REMOVED REDUNDANT FEATURES")
    print("-" * 80)

    print(
        redundancy_df.to_string(
            index=False
        )
    )


# =============================================================================
# FINAL FEATURE QC
# =============================================================================

print()
print("=" * 80)
print("FINAL FEATURE QC")
print("=" * 80)

selected_matrix = df[
    selected_features
].apply(
    pd.to_numeric,
    errors="coerce"
)

selected_nan = int(
    selected_matrix.isna().sum().sum()
)

selected_inf = int(
    np.isinf(
        selected_matrix.to_numpy(
            dtype=np.float64,
            copy=True
        )
    ).sum()
)

duplicate_keys = int(
    df.duplicated(
        subset=KEY_COLUMNS,
        keep=False
    ).sum()
)

print(
    f"Selected features:  {len(selected_features)}"
)

print(
    f"NaN values:         {selected_nan:,}"
)

print(
    f"Inf values:         {selected_inf:,}"
)

print(
    f"Duplicate keys:     {duplicate_keys:,}"
)


# =============================================================================
# TARGET LEAKAGE PROTECTION
# =============================================================================

TARGET_NAMES = {
    "target",
    "target_label",
    "target_remember",
    "target_correct",
    "is_correct",
    "is_remembered",
    "is_ignored",
}


target_leak_features = [
    f
    for f in selected_features
    if f in TARGET_NAMES
]


target_like_names = [
    f
    for f in selected_features
    if any(
        keyword in f.lower()
        for keyword in [
            "target",
            "correct",
            "remember",
            "ignore",
            "behavior",
            "feedback",
            "response",
            "probe",
        ]
    )
]


print()
print("=" * 80)
print("TARGET LEAKAGE PROTECTION")
print("=" * 80)

print(
    f"Exact target-derived features: "
    f"{len(target_leak_features)}"
)

print(
    f"Target-like feature names:      "
    f"{len(target_like_names)}"
)

if target_leak_features:

    print(
        target_leak_features
    )

    raise RuntimeError(
        "Target-derived feature detected "
        "in selected predictors. STOP."
    )

if target_like_names:

    print()
    print(
        "WARNING - target-like names detected:"
    )

    print(
        target_like_names
    )


# =============================================================================
# BUILD SELECTED DATASET
# =============================================================================

metadata_columns = [
    c
    for c in df.columns
    if c not in scientific_features
]

selected_dataset = pd.concat(
    [
        df[metadata_columns].reset_index(drop=True),
        df[selected_features].reset_index(drop=True),
    ],
    axis=1
)


# =============================================================================
# PRESERVE ROW ORDER AND KEYS
# =============================================================================

if len(selected_dataset) != len(df):

    raise RuntimeError(
        "Row count changed unexpectedly. STOP."
    )


for key in KEY_COLUMNS:

    if not selected_dataset[key].equals(
        df[key]
    ):

        raise RuntimeError(
            f"Key column changed: {key}"
        )


# =============================================================================
# FINAL DATASET VALIDATION
# =============================================================================

print()
print("=" * 80)
print("FINAL DATASET VALIDATION")
print("=" * 80)

print(
    f"Rows:                 {len(selected_dataset):,}"
)

print(
    f"Columns:              {len(selected_dataset.columns):,}"
)

print(
    f"Scientific predictors: "
    f"{len(selected_features):,}"
)

print(
    f"Subjects:             "
    f"{selected_dataset['subject'].nunique():,}"
)

print(
    f"Runs:                 "
    f"{selected_dataset['run'].nunique():,}"
)

print(
    f"Duplicate keys:       {duplicate_keys:,}"
)

print(
    f"NaN numeric:          {selected_nan:,}"
)

print(
    f"Inf numeric:          {selected_inf:,}"
)


# =============================================================================
# SAVE SELECTED FEATURES
# =============================================================================

feature_records = []

for feature in selected_features:

    feature_records.append(
        {
            "feature": feature,
            "source": "scientific_eeg",
            "target_independent": True,
            "constant_removed": False,
            "redundancy_removed": False,
        }
    )


for feature in constant_features:

    feature_records.append(
        {
            "feature": feature,
            "source": "scientific_eeg",
            "target_independent": True,
            "constant_removed": True,
            "redundancy_removed": False,
        }
    )


for feature in features_to_remove:

    feature_records.append(
        {
            "feature": feature,
            "source": "scientific_eeg",
            "target_independent": True,
            "constant_removed": False,
            "redundancy_removed": True,
        }
    )


feature_inventory_df = pd.DataFrame(
    feature_records
)

feature_inventory_df.to_csv(
    OUTPUT_FEATURES,
    index=False
)


# =============================================================================
# QC OUTPUT
# =============================================================================

qc_records = [
    {
        "metric": "input_rows",
        "value": len(df),
    },
    {
        "metric": "input_scientific_features",
        "value": len(scientific_features),
    },
    {
        "metric": "constant_features_removed",
        "value": len(constant_features),
    },
    {
        "metric": "features_after_constant_removal",
        "value": len(features_after_constant_removal),
    },
    {
        "metric": "high_correlation_pairs",
        "value": len(redundancy_df),
    },
    {
        "metric": "redundant_features_removed",
        "value": len(features_to_remove),
    },
    {
        "metric": "final_selected_features",
        "value": len(selected_features),
    },
    {
        "metric": "nan_values",
        "value": selected_nan,
    },
    {
        "metric": "inf_values",
        "value": selected_inf,
    },
    {
        "metric": "duplicate_keys",
        "value": duplicate_keys,
    },
    {
        "metric": "target_derived_features",
        "value": len(target_leak_features),
    },
    {
        "metric": "target_like_names",
        "value": len(target_like_names),
    },
]


qc_df = pd.DataFrame(
    qc_records
)

qc_df.to_csv(
    OUTPUT_QC,
    index=False
)


# =============================================================================
# SAVE DATASET
# =============================================================================

selected_dataset.to_csv(
    OUTPUT_DATASET,
    index=False
)


# =============================================================================
# FINAL STATUS
# =============================================================================

print()
print("=" * 80)
print("SCIENTIFIC FEATURE SELECTION V2 COMPLETE")
print("=" * 80)

print(
    f"Original scientific features: "
    f"{len(scientific_features)}"
)

print(
    f"Constant removed:             "
    f"{len(constant_features)}"
)

print(
    f"Redundant removed:            "
    f"{len(features_to_remove)}"
)

print(
    f"Final scientific features:    "
    f"{len(selected_features)}"
)

print(
    f"Rows preserved:               "
    f"{len(selected_dataset):,}"
)

print(
    f"Target-derived predictors:    "
    f"{len(target_leak_features)}"
)

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_DATASET)
print(OUTPUT_FEATURES)
print(OUTPUT_QC)
print(OUTPUT_CORR)

print()
print("=" * 80)
print("STATUS: PASS - TARGET-INDEPENDENT FEATURE SELECTION COMPLETED")
print("=" * 80)