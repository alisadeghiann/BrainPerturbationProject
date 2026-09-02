from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# REMOVE TARGET LEAKAGE + BUILD CLEAN ML DATASET
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "ml_ready"
    / "ml_ready_dataset.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "ml_ready"
    / "clean"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DATASET = OUTPUT_DIR / "ml_ready_clean_dataset.csv"
OUTPUT_QC = OUTPUT_DIR / "ml_ready_clean_qc.csv"


print("=" * 80)
print("ML TARGET LEAKAGE REMOVAL + CLEAN DATASET")
print("=" * 80)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

print(f"Input rows:       {len(df):,}")
print(f"Input columns:    {len(df.columns)}")
print(f"Subjects:         {df['subject'].nunique()}")


# ------------------------------------------------------------
# TARGET COLUMNS
# ------------------------------------------------------------

TARGET_COLUMNS = [
    "target_label",
    "target_remember",
    "target_correct",
]


# ------------------------------------------------------------
# CONFIRMED LEAKAGE FEATURES
# ------------------------------------------------------------

LEAKAGE_FEATURES = [
    "is_correct",
    "is_remembered",
    "is_ignored",
]


print()
print("=" * 80)
print("LEAKAGE FEATURES TO REMOVE")
print("=" * 80)

for col in LEAKAGE_FEATURES:
    if col in df.columns:
        print(f"REMOVE: {col}")
    else:
        print(f"NOT FOUND: {col}")


# ------------------------------------------------------------
# REMOVE TARGET-DERIVED FEATURES
# ------------------------------------------------------------

existing_leakage = [
    c for c in LEAKAGE_FEATURES
    if c in df.columns
]

df_clean = df.drop(columns=existing_leakage)


# ------------------------------------------------------------
# VERIFY TARGETS STILL EXIST
# ------------------------------------------------------------

print()
print("=" * 80)
print("TARGET VERIFICATION")
print("=" * 80)

for col in TARGET_COLUMNS:
    if col in df_clean.columns:
        print(f"{col}: PRESENT")
    else:
        print(f"{col}: MISSING")


# ------------------------------------------------------------
# FEATURE INVENTORY
# ------------------------------------------------------------

excluded_metadata = set([
    "file",
    "subject",
    "run",
    "epoch",
    "trial",
    "target_label",
    "target_remember",
    "target_correct",
])

numeric_cols = df_clean.select_dtypes(
    include=[np.number]
).columns.tolist()

feature_cols = [
    c for c in numeric_cols
    if c not in excluded_metadata
]


print()
print("=" * 80)
print("CLEAN FEATURE INVENTORY")
print("=" * 80)

print(f"Numeric columns:  {len(numeric_cols)}")
print(f"ML features:      {len(feature_cols)}")

print()
print("Features:")
for f in feature_cols:
    print(f"  {f}")


# ------------------------------------------------------------
# CHECK FOR TARGET-LIKE FEATURES
# ------------------------------------------------------------

print()
print("=" * 80)
print("TARGET-LIKE FEATURE CHECK")
print("=" * 80)

target_keywords = [
    "target",
    "correct",
    "remember",
    "ignore",
    "feedback",
    "response",
    "probe",
]

target_like = []

for f in feature_cols:
    lower = f.lower()

    if any(keyword in lower for keyword in target_keywords):
        target_like.append(f)

if target_like:
    print("WARNING - target-like feature names found:")
    for f in target_like:
        print(f"  {f}")
else:
    print("PASS - no target-like feature names found")


# ------------------------------------------------------------
# EXACT TARGET EQUALITY CHECK
# ------------------------------------------------------------

print()
print("=" * 80)
print("EXACT TARGET EQUALITY CHECK")
print("=" * 80)

equality_problems = []

if "target_remember" in df_clean.columns:
    for f in feature_cols:
        try:
            direct = (
                df_clean[f].astype(float)
                == df_clean["target_remember"].astype(float)
            ).all()

            inverse = (
                df_clean[f].astype(float)
                == (1 - df_clean["target_remember"].astype(float))
            ).all()

            if direct or inverse:
                equality_problems.append(
                    ("remember", f, direct, inverse)
                )

        except Exception:
            pass


if "target_correct" in df_clean.columns:
    for f in feature_cols:
        try:
            direct = (
                df_clean[f].astype(float)
                == df_clean["target_correct"].astype(float)
            ).all()

            inverse = (
                df_clean[f].astype(float)
                == (1 - df_clean["target_correct"].astype(float))
            ).all()

            if direct or inverse:
                equality_problems.append(
                    ("correct", f, direct, inverse)
                )

        except Exception:
            pass


if equality_problems:
    print("FAIL - exact target-derived feature detected")

    for item in equality_problems:
        print(
            f"Target={item[0]} | "
            f"Feature={item[1]} | "
            f"Direct={item[2]} | "
            f"Inverse={item[3]}"
        )

    raise RuntimeError(
        "Target leakage remains. STOP."
    )

else:
    print("PASS - no exact target equality detected")


# ------------------------------------------------------------
# NUMERIC QC
# ------------------------------------------------------------

print()
print("=" * 80)
print("NUMERIC QC")
print("=" * 80)

numeric_matrix = df_clean[feature_cols].apply(
    pd.to_numeric,
    errors="coerce"
)

nan_count = int(numeric_matrix.isna().sum().sum())

inf_count = int(
    np.isinf(
        numeric_matrix.to_numpy(dtype=np.float64)
    ).sum()
)

print(f"Feature NaN values:  {nan_count}")
print(f"Feature Inf values:  {inf_count}")

if nan_count > 0 or inf_count > 0:
    raise RuntimeError(
        "NaN or Inf detected in clean feature matrix."
    )

print("PASS - numeric QC")


# ------------------------------------------------------------
# TARGET DISTRIBUTIONS
# ------------------------------------------------------------

print()
print("=" * 80)
print("TARGET DISTRIBUTIONS")
print("=" * 80)

if "target_remember" in df_clean.columns:
    print()
    print("REMEMBER")
    print(
        df_clean["target_remember"]
        .value_counts(dropna=False)
        .sort_index()
    )

if "target_correct" in df_clean.columns:
    print()
    print("CORRECT")
    print(
        df_clean["target_correct"]
        .value_counts(dropna=False)
        .sort_index()
    )

if "target_label" in df_clean.columns:
    print()
    print("TARGET LABEL")
    print(
        df_clean["target_label"]
        .value_counts(dropna=False)
    )


# ------------------------------------------------------------
# SUBJECT COVERAGE
# ------------------------------------------------------------

print()
print("=" * 80)
print("SUBJECT COVERAGE")
print("=" * 80)

subject_summary = (
    df_clean
    .groupby("subject")
    .size()
    .reset_index(name="rows")
)

print(subject_summary.to_string(index=False))


# ------------------------------------------------------------
# RUN COVERAGE
# ------------------------------------------------------------

print()
print("=" * 80)
print("RUN COVERAGE")
print("=" * 80)

print(
    df_clean
    .groupby(["subject", "run"])
    .size()
    .reset_index(name="rows")
    .to_string(index=False)
)


# ------------------------------------------------------------
# FINAL COLUMN ORDER
# ------------------------------------------------------------

metadata_cols = [
    c for c in [
        "file",
        "subject",
        "run",
        "epoch",
        "trial",
    ]
    if c in df_clean.columns
]

target_cols = [
    c for c in [
        "target_label",
        "target_remember",
        "target_correct",
    ]
    if c in df_clean.columns
]

final_columns = (
    metadata_cols
    + feature_cols
    + target_cols
)

df_clean = df_clean[final_columns]


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

print()
print("=" * 80)
print("FINAL CLEAN DATASET VALIDATION")
print("=" * 80)

print(f"Rows:              {len(df_clean):,}")
print(f"Columns:           {len(df_clean.columns)}")
print(f"Features:          {len(feature_cols)}")
print(f"Subjects:          {df_clean['subject'].nunique()}")

if "run" in df_clean.columns:
    print(f"Runs:              {df_clean['run'].nunique()}")

if "trial" in df_clean.columns:
    print(f"Trials:            {df_clean['trial'].nunique()}")


# ------------------------------------------------------------
# QC TABLE
# ------------------------------------------------------------

qc = pd.DataFrame({
    "metric": [
        "input_rows",
        "output_rows",
        "input_columns",
        "output_columns",
        "removed_leakage_features",
        "feature_count",
        "subjects",
        "nan_values",
        "inf_values",
        "exact_target_leakage",
    ],
    "value": [
        len(df),
        len(df_clean),
        len(df.columns),
        len(df_clean.columns),
        len(existing_leakage),
        len(feature_cols),
        df_clean["subject"].nunique(),
        nan_count,
        inf_count,
        len(equality_problems),
    ]
})


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df_clean.to_csv(
    OUTPUT_DATASET,
    index=False
)

qc.to_csv(
    OUTPUT_QC,
    index=False
)


print()
print("=" * 80)
print("CLEAN ML DATASET CREATED")
print("=" * 80)

print()
print("SAVED:")
print(OUTPUT_DATASET)
print(OUTPUT_QC)

print()
print("=" * 80)
print("STATUS: PASS - TARGET-DERIVED FEATURES REMOVED")
print("=" * 80)