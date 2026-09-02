# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

FEATURES = (
    BASE
    / "features"
    / "scientific_v1"
    / "scientific_features_v1.csv"
)

BEHAVIOR = (
    BASE
    / "features"
    / "behavior_aligned"
    / "final"
    / "final_behavioral_trials.csv"
)

OUT_DIR = (
    BASE
    / "features"
    / "scientific_v1"
    / "merged"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "scientific_behavior_merged.csv"
QC_OUTPUT = OUT_DIR / "scientific_behavior_merge_qc.csv"


print("=" * 90)
print("SCIENTIFIC FEATURES + BEHAVIOR MERGE QC")
print("=" * 90)

# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

features = pd.read_csv(FEATURES)
behavior = pd.read_csv(BEHAVIOR)

print(f"Feature rows:  {len(features):,}")
print(f"Behavior rows: {len(behavior):,}")

# ---------------------------------------------------------
# NORMALIZE KEYS
# ---------------------------------------------------------

features["subject"] = features["subject"].astype(str)
features["run"] = features["run"].astype(str)

behavior["subject"] = behavior["subject"].astype(str)
behavior["run"] = behavior["run"].astype(str)

# ---------------------------------------------------------
# IMPORTANT:
# One trial corresponds to multiple EEG epochs in the original
# dataset. The final behavioral table contains one row per trial.
#
# Therefore we DO NOT blindly merge epoch -> trial.
#
# We first inspect whether the epoch numbering and behavioral
# trial structure allow a deterministic mapping.
# ---------------------------------------------------------

print()
print("=" * 90)
print("STRUCTURAL INSPECTION")
print("=" * 90)

print("Feature columns:")
print(features.columns.tolist())

print()
print("Behavior columns:")
print(behavior.columns.tolist())

print()
print("Feature epoch range by run:")

feature_ranges = (
    features
    .groupby(["subject", "run"])["epoch"]
    .agg(["min", "max", "count"])
    .reset_index()
)

print(feature_ranges.head(20).to_string(index=False))

print()
print("Behavior trial range by run:")

behavior_ranges = (
    behavior
    .groupby(["subject", "run"])["trial"]
    .agg(["min", "max", "count"])
    .reset_index()
)

print(behavior_ranges.head(20).to_string(index=False))

# ---------------------------------------------------------
# CREATE RUN-LEVEL COVERAGE QC
# ---------------------------------------------------------

run_qc = (
    feature_ranges
    .merge(
        behavior_ranges,
        on=["subject", "run"],
        how="outer",
        suffixes=("_feature", "_behavior")
    )
)

run_qc["feature_run_present"] = run_qc["count_feature"].notna()
run_qc["behavior_run_present"] = run_qc["count_behavior"].notna()

run_qc["coverage_status"] = np.select(
    [
        run_qc["feature_run_present"]
        & run_qc["behavior_run_present"],

        run_qc["feature_run_present"]
        & ~run_qc["behavior_run_present"],

        ~run_qc["feature_run_present"]
        & run_qc["behavior_run_present"],
    ],
    [
        "BOTH_PRESENT",
        "FEATURE_ONLY",
        "BEHAVIOR_ONLY",
    ],
    default="UNKNOWN"
)

# ---------------------------------------------------------
# SAVE QC
# ---------------------------------------------------------

run_qc.to_csv(QC_OUTPUT, index=False)

print()
print("=" * 90)
print("RUN COVERAGE")
print("=" * 90)

print(
    run_qc["coverage_status"]
    .value_counts(dropna=False)
)

print()
print("=" * 90)
print("QC SAVED")
print("=" * 90)

print(QC_OUTPUT)

print()
print("IMPORTANT:")
print(
    "No feature/behavior merge was performed yet because "
    "trial-to-epoch mapping must be deterministic."
)

print()
print("READ-ONLY")
print("No input files modified.")