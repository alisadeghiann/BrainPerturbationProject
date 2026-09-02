# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "behavior_aligned"
    / "trial_level_behavior_full.csv"
)

OUT_DIR = BASE / "features" / "behavior_aligned" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "final_behavioral_trials.csv"
QC_OUTPUT = OUT_DIR / "final_behavioral_qc.csv"
EXCLUDED_OUTPUT = OUT_DIR / "excluded_trials.csv"


print("=" * 90)
print("FINAL BEHAVIORAL DATASET")
print("=" * 90)

df = pd.read_csv(INPUT)

print(f"Input trials: {len(df):,}")

# ---------------------------------------------------------
# 1. DEFINE VALID TRIALS
# ---------------------------------------------------------

valid_alignment = df["alignment_status"].eq("ALIGNED")

valid_probe = df["probe_type"].isin(["target", "not_shown"])

valid_label = df["behavior_label"].isin([
    "remembered_correct",
    "ignored_correct",
    "remembered_incorrect",
    "ignored_incorrect",
])

valid_trial = (
    valid_alignment
    & valid_probe
    & valid_label
)

final_df = df.loc[valid_trial].copy()
excluded_df = df.loc[~valid_trial].copy()

# ---------------------------------------------------------
# 2. FINAL LABEL CATEGORIES
# ---------------------------------------------------------

final_df["is_correct"] = final_df["behavior_label"].isin([
    "remembered_correct",
    "ignored_correct",
])

final_df["is_remembered"] = final_df["behavior_label"].isin([
    "remembered_correct",
    "remembered_incorrect",
])

final_df["is_ignored"] = final_df["behavior_label"].isin([
    "ignored_correct",
    "ignored_incorrect",
])

# ---------------------------------------------------------
# 3. QC
# ---------------------------------------------------------

qc = {
    "input_trials": len(df),
    "final_trials": len(final_df),
    "excluded_trials": len(excluded_df),
    "excluded_percent": round(
        100 * len(excluded_df) / len(df), 2
    ),
    "subjects": final_df["subject"].nunique(),
    "runs": final_df[["subject", "run"]].drop_duplicates().shape[0],
    "target_trials": int(
        (final_df["probe_type"] == "target").sum()
    ),
    "not_shown_trials": int(
        (final_df["probe_type"] == "not_shown").sum()
    ),
    "remembered_correct": int(
        (final_df["behavior_label"] == "remembered_correct").sum()
    ),
    "ignored_correct": int(
        (final_df["behavior_label"] == "ignored_correct").sum()
    ),
    "remembered_incorrect": int(
        (final_df["behavior_label"] == "remembered_incorrect").sum()
    ),
    "ignored_incorrect": int(
        (final_df["behavior_label"] == "ignored_incorrect").sum()
    ),
}

qc_df = pd.DataFrame([qc])

# ---------------------------------------------------------
# 4. SAVE
# ---------------------------------------------------------

final_df.to_csv(OUTPUT, index=False)
excluded_df.to_csv(EXCLUDED_OUTPUT, index=False)
qc_df.to_csv(QC_OUTPUT, index=False)

# ---------------------------------------------------------
# 5. REPORT
# ---------------------------------------------------------

print()
print("=" * 90)
print("FINAL BEHAVIORAL DATASET COMPLETE")
print("=" * 90)

print(f"Input trials:       {len(df):,}")
print(f"Final valid trials: {len(final_df):,}")
print(f"Excluded trials:    {len(excluded_df):,}")
print(f"Excluded percent:   {qc['excluded_percent']}%")
print(f"Subjects:           {qc['subjects']}")
print(f"Runs:               {qc['runs']}")

print()
print("PROBE TYPE")
print(final_df["probe_type"].value_counts(dropna=False))

print()
print("BEHAVIOR LABEL")
print(final_df["behavior_label"].value_counts(dropna=False))

print()
print("SUBJECT COVERAGE")
print(
    final_df.groupby("subject")
    .size()
    .to_string()
)

print()
print("=" * 90)
print("SAVED")
print("=" * 90)

print(OUTPUT)
print(QC_OUTPUT)
print(EXCLUDED_OUTPUT)

print()
print("READ-ONLY INPUT")
print("No previous files modified.")