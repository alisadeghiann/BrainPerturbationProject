from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

FEATURE_FILE = (
    BASE
    / "features"
    / "scientific_v1"
    / "scientific_features_v1.csv"
)

MAPPING_FILE = (
    BASE
    / "features"
    / "scientific_v1"
    / "merged"
    / "deterministic_trial_epoch_map.csv"
)

OUT_DIR = BASE / "features" / "ml_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "ml_ready_dataset.csv"
QC_FILE = OUT_DIR / "ml_ready_qc.csv"

print("=" * 80)
print("ML-READY DATASET CONSTRUCTION V4")
print("=" * 80)

# ================================================================
# 1. LOAD
# ================================================================

features = pd.read_csv(FEATURE_FILE)
mapping = pd.read_csv(MAPPING_FILE)

print(f"Scientific rows: {len(features):,}")
print(f"Mapping rows:    {len(mapping):,}")

# ================================================================
# 2. NORMALIZE IDENTIFIERS
# ================================================================

features["subject"] = features["subject"].astype(str).str.strip()
mapping["subject"] = mapping["subject"].astype(str).str.strip()

features["file"] = features["file"].astype(str).str.strip()
mapping["file"] = mapping["file"].astype(str).str.strip()

features["epoch"] = pd.to_numeric(
    features["epoch"],
    errors="coerce"
)

mapping["epoch"] = pd.to_numeric(
    mapping["epoch"],
    errors="coerce"
)

if features["epoch"].isna().any():
    raise RuntimeError("Invalid epoch values found in feature dataset.")

if mapping["epoch"].isna().any():
    raise RuntimeError("Invalid epoch values found in mapping dataset.")

features["epoch"] = features["epoch"].astype(int)
mapping["epoch"] = mapping["epoch"].astype(int)

# ================================================================
# 3. CREATE DETERMINISTIC MERGE KEY
# ================================================================

# IMPORTANT:
# Do NOT merge using run.
#
# Feature dataset:
#     run = 1, 2, 3...
#
# Mapping:
#     run = run-01, run-02...
#
# Therefore we use:
#     file + epoch
#
# The FIF filename identifies the run and epoch identifies
# the exact EEG epoch.

features["merge_key"] = (
    features["file"]
    + "|"
    + features["epoch"].astype(str)
)

mapping["merge_key"] = (
    mapping["file"]
    + "|"
    + mapping["epoch"].astype(str)
)

print()
print("=" * 80)
print("CHECKING KEY OVERLAP")
print("=" * 80)

feature_keys = set(features["merge_key"])
mapping_keys = set(mapping["merge_key"])

overlap_keys = feature_keys.intersection(mapping_keys)

print(f"Feature keys:  {len(feature_keys):,}")
print(f"Mapping keys:  {len(mapping_keys):,}")
print(f"Overlap keys:  {len(overlap_keys):,}")

if len(overlap_keys) == 0:
    raise RuntimeError(
        "ZERO OVERLAP after file+epoch normalization. STOP."
    )

print()
print("OVERLAP STATUS: PASS")

# ================================================================
# 4. DUPLICATE KEY CHECK
# ================================================================

feature_duplicates = int(
    features["merge_key"].duplicated().sum()
)

mapping_duplicates = int(
    mapping["merge_key"].duplicated().sum()
)

print()
print("=" * 80)
print("DUPLICATE KEY CHECK")
print("=" * 80)

print(f"Feature duplicate keys:  {feature_duplicates}")
print(f"Mapping duplicate keys:  {mapping_duplicates}")

if feature_duplicates > 0:
    raise RuntimeError(
        "Duplicate feature epoch keys detected."
    )

if mapping_duplicates > 0:
    raise RuntimeError(
        "Duplicate mapping epoch keys detected."
    )

# ================================================================
# 5. SELECT BEHAVIOR METADATA
# ================================================================

behavior_columns = [
    "subject",
    "run",
    "trial",
    "n_epochs",
    "memory_cond",
    "remember_count",
    "ignore_count",
    "remember_letters",
    "ignore_letters",
    "probe_type",
    "probe_letter",
    "behavior_outcome",
    "behavior_label",
    "is_correct",
    "is_remembered",
    "is_ignored",
    "complete_trial",
    "alignment_status",
    "event_name",
]

available_behavior_columns = [
    col
    for col in behavior_columns
    if col in mapping.columns
]

mapping_small = mapping[
    ["merge_key"] + available_behavior_columns
].copy()

# ================================================================
# 6. MERGE
# ================================================================

print()
print("=" * 80)
print("MERGING FEATURES + BEHAVIOR")
print("=" * 80)

merged = features.merge(
    mapping_small,
    on="merge_key",
    how="inner",
    suffixes=("", "_mapping")
)

print(f"Merged rows: {len(merged):,}")

if len(merged) == 0:
    raise RuntimeError(
        "MERGE PRODUCED ZERO ROWS. STOP."
    )

# ================================================================
# 7. REMOVE INTERNAL KEY
# ================================================================

merged.drop(
    columns=["merge_key"],
    inplace=True
)

# ================================================================
# 8. TARGET CHECK
# ================================================================

print()
print("=" * 80)
print("TARGET CHECK")
print("=" * 80)

if "behavior_label" not in merged.columns:
    raise RuntimeError(
        "behavior_label is missing after merge."
    )

valid_labels = {
    "remembered_correct",
    "ignored_correct",
    "remembered_incorrect",
    "ignored_incorrect",
}

valid_target = merged["behavior_label"].isin(
    valid_labels
)

valid_count = int(valid_target.sum())
invalid_count = int((~valid_target).sum())

print(f"Valid target rows:   {valid_count:,}")
print(f"Invalid target rows: {invalid_count:,}")

if valid_count == 0:
    raise RuntimeError(
        "ZERO VALID TARGET ROWS. STOP."
    )

# Keep only valid behavioral labels
ml = merged.loc[valid_target].copy()

# ================================================================
# 9. CREATE ML TARGETS
# ================================================================

ml["target_label"] = (
    ml["behavior_label"].astype(str)
)

ml["target_remember"] = pd.to_numeric(
    ml["is_remembered"],
    errors="coerce"
)

ml["target_correct"] = pd.to_numeric(
    ml["is_correct"],
    errors="coerce"
)

# ================================================================
# 10. NUMERIC QC
# ================================================================

numeric_df = ml.select_dtypes(
    include=[np.number]
)

nan_count = int(
    numeric_df.isna().sum().sum()
)

inf_count = 0

for column in numeric_df.columns:

    values = pd.to_numeric(
        numeric_df[column],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    inf_count += int(
        np.isinf(values).sum()
    )

print()
print("=" * 80)
print("NUMERIC QC")
print("=" * 80)

print(f"Numeric columns: {len(numeric_df.columns)}")
print(f"NaN values:      {nan_count:,}")
print(f"Inf values:      {inf_count:,}")

# ================================================================
# 11. TARGET DISTRIBUTION
# ================================================================

print()
print("=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

print(
    ml["target_label"].value_counts(
        dropna=False
    )
)

print()
print("REMEMBER TARGET")
print("-" * 80)

print(
    ml["target_remember"].value_counts(
        dropna=False
    )
)

print()
print("CORRECT TARGET")
print("-" * 80)

print(
    ml["target_correct"].value_counts(
        dropna=False
    )
)

# ================================================================
# 12. DATASET SUMMARY
# ================================================================

print()
print("=" * 80)
print("ML-READY DATASET COMPLETE")
print("=" * 80)

print(
    f"Scientific feature rows: {len(features):,}"
)

print(
    f"Mapping rows:            {len(mapping):,}"
)

print(
    f"Merged rows:             {len(merged):,}"
)

print(
    f"Final ML rows:           {len(ml):,}"
)

print(
    f"Feature columns:         {len(features.columns):,}"
)

print(
    f"Subjects:                {ml['subject'].nunique():,}"
)

print(
    f"Runs:                    {ml['run'].nunique():,}"
)

print(
    f"Trials:                  {ml['trial'].nunique():,}"
)

print(
    f"NaN numeric values:      {nan_count:,}"
)

print(
    f"Inf numeric values:      {inf_count:,}"
)

# ================================================================
# 13. QC TABLE
# ================================================================

qc = pd.DataFrame(
    {
        "metric": [
            "scientific_rows",
            "mapping_rows",
            "feature_unique_keys",
            "mapping_unique_keys",
            "overlap_keys",
            "feature_duplicate_keys",
            "mapping_duplicate_keys",
            "merged_rows",
            "valid_target_rows",
            "invalid_target_rows",
            "final_ml_rows",
            "subjects",
            "runs",
            "trials",
            "nan_numeric_values",
            "inf_numeric_values",
        ],
        "value": [
            len(features),
            len(mapping),
            len(feature_keys),
            len(mapping_keys),
            len(overlap_keys),
            feature_duplicates,
            mapping_duplicates,
            len(merged),
            valid_count,
            invalid_count,
            len(ml),
            ml["subject"].nunique(),
            ml["run"].nunique(),
            ml["trial"].nunique(),
            nan_count,
            inf_count,
        ],
    }
)

# ================================================================
# 14. SAVE
# ================================================================

ml.to_csv(
    OUT_FILE,
    index=False
)

qc.to_csv(
    QC_FILE,
    index=False
)

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUT_FILE)
print(QC_FILE)

print()
print("=" * 80)
print("STATUS: PASS - ML-READY DATASET CREATED")
print("=" * 80)