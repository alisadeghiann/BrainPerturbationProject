import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# ML-READY DATASET CONSTRUCTION V2
# Scientific V2 + Deterministic Trial/Epoch Mapping
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

FEATURES = (
    BASE
    / "features"
    / "scientific_v2"
    / "scientific_features_v2.csv"
)

MAPPING = (
    BASE
    / "features"
    / "scientific_v1"
    / "merged"
    / "deterministic_trial_epoch_map.csv"
)

OUTPUT_DIR = BASE / "features" / "ml_ready_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "ml_ready_dataset_v2.csv"
QC_OUTPUT = OUTPUT_DIR / "ml_ready_v2_qc.csv"

print("=" * 80)
print("ML-READY DATASET CONSTRUCTION V2")
print("=" * 80)

# =============================================================================
# LOAD
# =============================================================================

features = pd.read_csv(FEATURES)
mapping = pd.read_csv(MAPPING)

print(f"Scientific V2 rows: {len(features):,}")
print(f"Mapping rows:       {len(mapping):,}")

# =============================================================================
# NORMALIZE KEYS
# =============================================================================

KEYS = ["file", "subject", "run", "epoch"]

for df in [features, mapping]:

    for col in ["file", "subject", "run"]:
        df[col] = df[col].astype(str).str.strip()

    df["epoch"] = pd.to_numeric(
        df["epoch"],
        errors="coerce"
    )

features = features.dropna(subset=KEYS).copy()
mapping = mapping.dropna(subset=KEYS).copy()

features["epoch"] = features["epoch"].astype(int)
mapping["epoch"] = mapping["epoch"].astype(int)

# =============================================================================
# KEY OVERLAP
# =============================================================================

print()
print("=" * 80)
print("CHECKING KEY OVERLAP")
print("=" * 80)

feature_keys = set(
    map(tuple, features[KEYS].to_numpy())
)

mapping_keys = set(
    map(tuple, mapping[KEYS].to_numpy())
)

overlap = feature_keys.intersection(mapping_keys)

print(f"Feature keys:  {len(feature_keys):,}")
print(f"Mapping keys:  {len(mapping_keys):,}")
print(f"Overlap keys:  {len(overlap):,}")

if len(overlap) == 0:
    raise RuntimeError(
        "ZERO OVERLAP between Scientific V2 features and mapping."
    )

print()
print("OVERLAP STATUS: PASS")

# =============================================================================
# DUPLICATE KEY CHECK
# =============================================================================

print()
print("=" * 80)
print("DUPLICATE KEY CHECK")
print("=" * 80)

feature_dups = int(
    features.duplicated(KEYS).sum()
)

mapping_dups = int(
    mapping.duplicated(KEYS).sum()
)

print(f"Feature duplicate keys: {feature_dups}")
print(f"Mapping duplicate keys: {mapping_dups}")

if feature_dups > 0:
    raise RuntimeError(
        "Duplicate feature keys detected."
    )

if mapping_dups > 0:
    raise RuntimeError(
        "Duplicate mapping keys detected."
    )

# =============================================================================
# REQUIRED MAPPING COLUMNS
# =============================================================================

required_mapping = [
    "file",
    "subject",
    "run",
    "epoch",
    "trial",
    "memory_cond",
    "probe_type",
    "probe_letter",
    "behavior_label",
    "is_correct",
    "is_remembered",
    "is_ignored",
]

missing = [
    c for c in required_mapping
    if c not in mapping.columns
]

if missing:
    raise RuntimeError(
        f"Missing mapping columns: {missing}"
    )

mapping_ml = mapping[required_mapping].copy()

# =============================================================================
# MERGE
# =============================================================================

print()
print("=" * 80)
print("MERGING SCIENTIFIC V2 FEATURES + BEHAVIOR")
print("=" * 80)

merged = features.merge(
    mapping_ml,
    on=KEYS,
    how="inner",
    validate="one_to_one"
)

print(f"Merged rows: {len(merged):,}")

if len(merged) == 0:
    raise RuntimeError(
        "Merged dataset contains zero rows."
    )

# =============================================================================
# TARGET VALIDATION
# =============================================================================

print()
print("=" * 80)
print("TARGET VALIDATION")
print("=" * 80)

valid_mask = (
    merged["behavior_label"].notna()
    & merged["is_correct"].notna()
    & merged["is_remembered"].notna()
)

print(
    f"Valid target rows:   {int(valid_mask.sum()):,}"
)

print(
    f"Invalid target rows: {int((~valid_mask).sum()):,}"
)

merged = merged.loc[valid_mask].copy()

# =============================================================================
# CREATE TARGETS
# =============================================================================

merged["target_label"] = (
    merged["behavior_label"].astype(str)
)

merged["target_remember"] = (
    pd.to_numeric(
        merged["is_remembered"],
        errors="raise"
    ).astype(int)
)

merged["target_correct"] = (
    pd.to_numeric(
        merged["is_correct"],
        errors="raise"
    ).astype(int)
)

# =============================================================================
# REMOVE TARGET-DERIVED FEATURES
# =============================================================================

print()
print("=" * 80)
print("REMOVING TARGET-DERIVED FEATURES")
print("=" * 80)

TARGET_DERIVED = [
    "behavior_label",
    "is_correct",
    "is_remembered",
    "is_ignored",
    "target_label",
    "target_remember",
    "target_correct",
]

IDENTIFIERS = [
    "file",
    "subject",
    "run",
    "epoch",
    "trial",
    "memory_cond",
    "probe_type",
    "probe_letter",
]

DROP_COLS = set(
    TARGET_DERIVED + IDENTIFIERS
)

candidate_features = [
    c
    for c in merged.columns
    if c not in DROP_COLS
]

numeric_features = [
    c
    for c in candidate_features
    if pd.api.types.is_numeric_dtype(
        merged[c]
    )
]

print(
    f"Candidate predictor columns: {len(candidate_features)}"
)

print(
    f"Numeric EEG features:         {len(numeric_features)}"
)

if len(numeric_features) == 0:
    raise RuntimeError(
        "No numeric EEG features remain."
    )

print()
print("FEATURES:")
print(numeric_features)

# =============================================================================
# NUMERIC QC
# =============================================================================

print()
print("=" * 80)
print("NUMERIC QC")
print("=" * 80)

numeric_data = merged[numeric_features].apply(
    pd.to_numeric,
    errors="coerce"
)

nan_count = int(
    numeric_data.isna().sum().sum()
)

numeric_array = numeric_data.to_numpy(
    dtype=np.float64
)

inf_count = int(
    np.isinf(numeric_array).sum()
)

print(f"Numeric columns: {len(numeric_features)}")
print(f"NaN values:      {nan_count}")
print(f"Inf values:      {inf_count}")

if nan_count > 0:
    raise RuntimeError(
        "NaN values detected."
    )

if inf_count > 0:
    raise RuntimeError(
        "Inf values detected."
    )

# =============================================================================
# TARGET DISTRIBUTION
# =============================================================================

print()
print("=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

print(
    merged["target_label"].value_counts()
)

print()
print("REMEMBER TARGET")
print("-" * 80)

print(
    merged["target_remember"]
    .value_counts()
    .sort_index()
)

print()
print("CORRECT TARGET")
print("-" * 80)

print(
    merged["target_correct"]
    .value_counts()
    .sort_index()
)

# =============================================================================
# FINAL TARGET LEAKAGE CHECK
# =============================================================================

print()
print("=" * 80)
print("FINAL TARGET LEAKAGE CHECK")
print("=" * 80)

for forbidden in [
    "is_correct",
    "is_remembered",
    "is_ignored",
    "behavior_label",
    "target_label",
    "target_remember",
    "target_correct",
]:

    if forbidden in numeric_features:
        raise RuntimeError(
            f"TARGET LEAKAGE DETECTED: {forbidden}"
        )

print(
    "PASS - No target-derived variables are predictors."
)

# =============================================================================
# BUILD FINAL DATASET
# =============================================================================

final_columns = (
    IDENTIFIERS
    + [
        "target_label",
        "target_remember",
        "target_correct",
    ]
    + numeric_features
)

final_columns = list(
    dict.fromkeys(final_columns)
)

ml = merged[final_columns].copy()

# =============================================================================
# FINAL VALIDATION
# =============================================================================

print()
print("=" * 80)
print("FINAL DATASET VALIDATION")
print("=" * 80)

duplicate_final = int(
    ml.duplicated(KEYS).sum()
)

final_numeric = ml[numeric_features].apply(
    pd.to_numeric,
    errors="coerce"
)

final_nan = int(
    final_numeric.isna().sum().sum()
)

final_inf = int(
    np.isinf(
        final_numeric.to_numpy(
            dtype=np.float64
        )
    ).sum()
)

print(f"Rows:              {len(ml):,}")
print(f"Columns:           {len(ml.columns)}")
print(f"Numeric features:  {len(numeric_features)}")
print(f"Subjects:          {ml['subject'].nunique()}")
print(f"Runs:              {ml['run'].nunique()}")
print(f"Trials:            {ml['trial'].nunique()}")
print(f"Duplicate keys:    {duplicate_final}")
print(f"NaN numeric:       {final_nan}")
print(f"Inf numeric:       {final_inf}")

if duplicate_final > 0:
    raise RuntimeError(
        "Duplicate keys found in final dataset."
    )

if final_nan > 0:
    raise RuntimeError(
        "NaN values found in final dataset."
    )

if final_inf > 0:
    raise RuntimeError(
        "Inf values found in final dataset."
    )

# =============================================================================
# SAVE DATASET
# =============================================================================

ml.to_csv(
    OUTPUT,
    index=False
)

# =============================================================================
# SAVE QC
# =============================================================================

qc_rows = [
    ["scientific_v2_rows", len(features)],
    ["mapping_rows", len(mapping)],
    ["overlap_keys", len(overlap)],
    ["merged_rows", len(merged)],
    ["final_ml_rows", len(ml)],
    ["subjects", ml["subject"].nunique()],
    ["runs", ml["run"].nunique()],
    ["trials", ml["trial"].nunique()],
    ["numeric_features", len(numeric_features)],
    ["duplicate_keys", duplicate_final],
    ["nan_values", final_nan],
    ["inf_values", final_inf],
    [
        "remember_false",
        int(
            (ml["target_remember"] == 0).sum()
        )
    ],
    [
        "remember_true",
        int(
            (ml["target_remember"] == 1).sum()
        )
    ],
    [
        "correct_false",
        int(
            (ml["target_correct"] == 0).sum()
        )
    ],
    [
        "correct_true",
        int(
            (ml["target_correct"] == 1).sum()
        )
    ],
]

qc = pd.DataFrame(
    qc_rows,
    columns=["metric", "value"]
)

qc.to_csv(
    QC_OUTPUT,
    index=False
)

# =============================================================================
# COMPLETE
# =============================================================================

print()
print("=" * 80)
print("ML-READY DATASET V2 COMPLETE")
print("=" * 80)

print(f"Scientific V2 rows: {len(features):,}")
print(f"Mapping rows:       {len(mapping):,}")
print(f"Merged rows:        {len(merged):,}")
print(f"Final ML rows:      {len(ml):,}")
print(f"Numeric features:   {len(numeric_features)}")
print(f"Subjects:           {ml['subject'].nunique()}")
print(f"Runs:               {ml['run'].nunique()}")
print(f"Trials:             {ml['trial'].nunique()}")
print(f"NaN numeric:        {final_nan}")
print(f"Inf numeric:        {final_inf}")

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT)
print(QC_OUTPUT)

print()
print("=" * 80)
print("STATUS: PASS - SCIENTIFIC V2 ML DATASET CREATED")
print("=" * 80)