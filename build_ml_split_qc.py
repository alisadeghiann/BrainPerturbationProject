from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT_FILE = (
    BASE
    / "features"
    / "ml_ready"
    / "ml_ready_dataset.csv"
)

OUT_DIR = (
    BASE
    / "features"
    / "ml_ready"
    / "split"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILE = OUT_DIR / "subject_level_split.csv"
QC_FILE = OUT_DIR / "subject_level_split_qc.csv"

print("=" * 80)
print("ML QUALITY CONTROL + SUBJECT-LEVEL SPLIT")
print("=" * 80)

# ================================================================
# 1. LOAD
# ================================================================

df = pd.read_csv(INPUT_FILE)

print(f"Input rows: {len(df):,}")
print(f"Subjects:    {df['subject'].nunique():,}")

# ================================================================
# 2. BASIC VALIDATION
# ================================================================

required_columns = [
    "subject",
    "run",
    "trial",
    "target_label",
    "target_remember",
    "target_correct",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

# ================================================================
# 3. SUBJECT INFORMATION
# ================================================================

subject_summary = (
    df.groupby("subject")
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

print(subject_summary.to_string(index=False))

# ================================================================
# 4. SUBJECT-LEVEL SPLIT
# ================================================================

subjects = sorted(
    df["subject"].dropna().unique()
)

n_subjects = len(subjects)

if n_subjects < 5:
    raise RuntimeError(
        "Too few subjects for reliable subject-level ML."
    )

# Fixed deterministic split.
#
# Approximately:
#   70% TRAIN
#   15% VALIDATION
#   15% TEST
#
# IMPORTANT:
# Subjects, NOT epochs, are split.

rng = np.random.RandomState(42)

shuffled_subjects = subjects.copy()
rng.shuffle(shuffled_subjects)

n_train = int(round(n_subjects * 0.70))
n_valid = int(round(n_subjects * 0.15))

# Ensure at least one subject remains for test
if n_train + n_valid >= n_subjects:
    n_valid = max(1, n_subjects - n_train - 1)

train_subjects = shuffled_subjects[:n_train]

valid_subjects = shuffled_subjects[
    n_train:n_train + n_valid
]

test_subjects = shuffled_subjects[
    n_train + n_valid:
]

# ================================================================
# 5. ASSIGN SPLIT
# ================================================================

split_map = {}

for subject in train_subjects:
    split_map[subject] = "train"

for subject in valid_subjects:
    split_map[subject] = "validation"

for subject in test_subjects:
    split_map[subject] = "test"

df["split"] = df["subject"].map(split_map)

if df["split"].isna().any():
    raise RuntimeError(
        "Some subjects were not assigned to a split."
    )

# ================================================================
# 6. LEAKAGE CHECK
# ================================================================

train_set = set(train_subjects)
valid_set = set(valid_subjects)
test_set = set(test_subjects)

train_valid_overlap = train_set.intersection(valid_set)
train_test_overlap = train_set.intersection(test_set)
valid_test_overlap = valid_set.intersection(test_set)

print()
print("=" * 80)
print("LEAKAGE CHECK")
print("=" * 80)

print(
    f"Train subjects:      {len(train_set)}"
)

print(
    f"Validation subjects: {len(valid_set)}"
)

print(
    f"Test subjects:       {len(test_set)}"
)

print(
    f"Train ∩ Validation:  {len(train_valid_overlap)}"
)

print(
    f"Train ∩ Test:        {len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test:   {len(valid_test_overlap)}"
)

if (
    train_valid_overlap
    or train_test_overlap
    or valid_test_overlap
):
    raise RuntimeError(
        "SUBJECT LEAKAGE DETECTED. STOP."
    )

print()
print("LEAKAGE STATUS: PASS")

# ================================================================
# 7. ROW DISTRIBUTION
# ================================================================

print()
print("=" * 80)
print("ROW DISTRIBUTION")
print("=" * 80)

print(
    df["split"].value_counts()
)

# ================================================================
# 8. TARGET DISTRIBUTION BY SPLIT
# ================================================================

print()
print("=" * 80)
print("TARGET DISTRIBUTION BY SPLIT")
print("=" * 80)

target_table = pd.crosstab(
    df["split"],
    df["target_label"]
)

print(target_table)

# ================================================================
# 9. BINARY TARGET DISTRIBUTION
# ================================================================

print()
print("=" * 80)
print("REMEMBER TARGET BY SPLIT")
print("=" * 80)

print(
    pd.crosstab(
        df["split"],
        df["target_remember"]
    )
)

print()
print("=" * 80)
print("CORRECT TARGET BY SPLIT")
print("=" * 80)

print(
    pd.crosstab(
        df["split"],
        df["target_correct"]
    )
)

# ================================================================
# 10. SUBJECT SPLIT TABLE
# ================================================================

subject_split = pd.DataFrame(
    {
        "subject": subjects,
        "split": [
            split_map[s]
            for s in subjects
        ],
    }
)

subject_split = subject_split.merge(
    subject_summary,
    on="subject",
    how="left"
)

# ================================================================
# 11. FINAL QC
# ================================================================

print()
print("=" * 80)
print("FINAL ML SPLIT QC")
print("=" * 80)

print(
    f"Total rows:     {len(df):,}"
)

print(
    f"Total subjects: {df['subject'].nunique():,}"
)

print(
    f"Train rows:     {(df['split'] == 'train').sum():,}"
)

print(
    f"Validation rows:{(df['split'] == 'validation').sum():,}"
)

print(
    f"Test rows:      {(df['split'] == 'test').sum():,}"
)

# ================================================================
# 12. SAVE
# ================================================================

df.to_csv(
    SPLIT_FILE,
    index=False
)

qc_rows = []

qc_rows.append(
    ["total_rows", len(df)]
)

qc_rows.append(
    ["total_subjects", df["subject"].nunique()]
)

qc_rows.append(
    ["train_subjects", len(train_set)]
)

qc_rows.append(
    ["validation_subjects", len(valid_set)]
)

qc_rows.append(
    ["test_subjects", len(test_set)]
)

qc_rows.append(
    ["train_validation_overlap", len(train_valid_overlap)]
)

qc_rows.append(
    ["train_test_overlap", len(train_test_overlap)]
)

qc_rows.append(
    ["validation_test_overlap", len(valid_test_overlap)]
)

qc_rows.append(
    ["train_rows", int((df["split"] == "train").sum())]
)

qc_rows.append(
    [
        "validation_rows",
        int((df["split"] == "validation").sum()),
    ]
)

qc_rows.append(
    ["test_rows", int((df["split"] == "test").sum())]
)

qc = pd.DataFrame(
    qc_rows,
    columns=["metric", "value"]
)

qc.to_csv(
    QC_FILE,
    index=False
)

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(SPLIT_FILE)
print(QC_FILE)

print()
print("=" * 80)
print("STATUS: PASS - SUBJECT-LEVEL SPLIT CREATED")
print("=" * 80)