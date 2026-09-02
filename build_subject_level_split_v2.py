from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================
# SUBJECT-LEVEL SPLIT V2
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "ml_ready_v2"
    / "ml_ready_dataset_v2.csv"
)

OUT_DIR = (
    BASE
    / "features"
    / "ml_ready_v2"
    / "split"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "subject_level_split_v2.csv"
QC_OUTPUT = OUT_DIR / "subject_level_split_v2_qc.csv"

print("=" * 80)
print("SUBJECT-LEVEL SPLIT V2")
print("=" * 80)

df = pd.read_csv(INPUT)

print(f"Input rows: {len(df):,}")
print(f"Subjects:    {df['subject'].nunique()}")

required = {
    "subject",
    "target_remember",
    "target_correct",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )

# ============================================================
# SUBJECT LIST
# ============================================================

subjects = sorted(df["subject"].dropna().unique())

if len(subjects) < 10:
    raise RuntimeError(
        f"Too few subjects for subject-level split: {len(subjects)}"
    )

# ============================================================
# 70 / 15 / 15 SUBJECT SPLIT
# ============================================================

train_subjects, temp_subjects = train_test_split(
    subjects,
    test_size=0.30,
    random_state=42,
)

validation_subjects, test_subjects = train_test_split(
    temp_subjects,
    test_size=0.50,
    random_state=42,
)

train_subjects = sorted(train_subjects)
validation_subjects = sorted(validation_subjects)
test_subjects = sorted(test_subjects)

print()
print("=" * 80)
print("SUBJECT SPLIT")
print("=" * 80)

print(f"Train subjects:      {len(train_subjects)}")
print(f"Validation subjects: {len(validation_subjects)}")
print(f"Test subjects:       {len(test_subjects)}")

print()
print("Train:", train_subjects)
print("Validation:", validation_subjects)
print("Test:", test_subjects)

# ============================================================
# ASSIGN SPLIT
# ============================================================

subject_to_split = {}

for s in train_subjects:
    subject_to_split[s] = "train"

for s in validation_subjects:
    subject_to_split[s] = "validation"

for s in test_subjects:
    subject_to_split[s] = "test"

df["split"] = df["subject"].map(subject_to_split)

if df["split"].isna().any():
    missing_subjects = sorted(
        df.loc[df["split"].isna(), "subject"].unique()
    )

    raise RuntimeError(
        f"Some subjects were not assigned to a split: {missing_subjects}"
    )

# ============================================================
# LEAKAGE CHECK
# ============================================================

train_set = set(train_subjects)
validation_set = set(validation_subjects)
test_set = set(test_subjects)

tv = train_set & validation_set
tt = train_set & test_set
vt = validation_set & test_set

print()
print("=" * 80)
print("SUBJECT-LEVEL LEAKAGE CHECK")
print("=" * 80)

print(f"Train ∩ Validation:  {len(tv)}")
print(f"Train ∩ Test:        {len(tt)}")
print(f"Validation ∩ Test:   {len(vt)}")

if tv or tt or vt:
    raise RuntimeError(
        "SUBJECT LEAKAGE DETECTED. STOP."
    )

print()
print("LEAKAGE STATUS: PASS")

# ============================================================
# ROW DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("ROW DISTRIBUTION")
print("=" * 80)

print(
    df["split"]
    .value_counts()
    .sort_index()
)

# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("REMEMBER TARGET BY SPLIT")
print("=" * 80)

remember_table = pd.crosstab(
    df["split"],
    df["target_remember"]
)

print(remember_table)

print()
print("=" * 80)
print("CORRECT TARGET BY SPLIT")
print("=" * 80)

correct_table = pd.crosstab(
    df["split"],
    df["target_correct"]
)

print(correct_table)

# ============================================================
# SUBJECT SUMMARY
# ============================================================

summary = (
    df.groupby(["subject", "split"])
    .agg(
        rows=("subject", "size"),
        runs=("run", "nunique"),
        trials=("trial", "nunique"),
        remembered=("target_remember", "sum"),
        correct=("target_correct", "sum"),
    )
    .reset_index()
)

print()
print("=" * 80)
print("SUBJECT SUMMARY")
print("=" * 80)

print(summary.to_string(index=False))

# ============================================================
# DUPLICATE CHECK
# ============================================================

key_cols = ["subject", "run", "epoch"]

duplicate_keys = df.duplicated(
    subset=key_cols,
    keep=False
).sum()

print()
print("=" * 80)
print("DUPLICATE KEY CHECK")
print("=" * 80)

print(f"Duplicate subject/run/epoch rows: {duplicate_keys}")

if duplicate_keys != 0:
    raise RuntimeError(
        "Duplicate subject/run/epoch keys detected."
    )

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)

qc_rows = []

qc_rows.append({
    "metric": "total_rows",
    "value": len(df)
})

qc_rows.append({
    "metric": "total_subjects",
    "value": df["subject"].nunique()
})

qc_rows.append({
    "metric": "train_subjects",
    "value": len(train_subjects)
})

qc_rows.append({
    "metric": "validation_subjects",
    "value": len(validation_subjects)
})

qc_rows.append({
    "metric": "test_subjects",
    "value": len(test_subjects)
})

qc_rows.append({
    "metric": "train_validation_overlap",
    "value": len(tv)
})

qc_rows.append({
    "metric": "train_test_overlap",
    "value": len(tt)
})

qc_rows.append({
    "metric": "validation_test_overlap",
    "value": len(vt)
})

qc_rows.append({
    "metric": "duplicate_keys",
    "value": duplicate_keys
})

qc = pd.DataFrame(qc_rows)

qc.to_csv(
    QC_OUTPUT,
    index=False
)

# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 80)
print("SUBJECT-LEVEL SPLIT V2 COMPLETE")
print("=" * 80)

print(f"Total rows:       {len(df):,}")
print(f"Train rows:       {(df['split'] == 'train').sum():,}")
print(f"Validation rows:  {(df['split'] == 'validation').sum():,}")
print(f"Test rows:        {(df['split'] == 'test').sum():,}")

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT)
print(QC_OUTPUT)

print()
print("=" * 80)
print("STATUS: PASS - SUBJECT-LEVEL V2 SPLIT CREATED")
print("=" * 80)