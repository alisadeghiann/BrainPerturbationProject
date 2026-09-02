# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# SUBJECT-LEVEL PERTURBATION / WHAT-IF ROBUSTNESS V6
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "robustness_v6"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "subject_level_perturbation_robustness_v6.csv"
)

QC_FILE = (
    OUTPUT_DIR
    / "subject_level_perturbation_robustness_v6_qc.csv"
)

print("=" * 80)
print("SUBJECT-LEVEL PERTURBATION / WHAT-IF ROBUSTNESS V6")
print("=" * 80)
print(f"Project root: {BASE}")


# =============================================================================
# 1. FIND TRUE SUBJECT-LEVEL PERTURBATION DATA
# =============================================================================

print("\n" + "=" * 80)
print("SEARCHING FOR SUBJECT-LEVEL PERTURBATION DATA")
print("=" * 80)

required_columns = {
    "subject",
    "feature",
    "remember_effect_size",
    "correct_effect_size",
}

candidate_files = []

for path in BASE.rglob("*.csv"):

    # Never use our own output as an input
    if OUTPUT_DIR in path.parents:
        continue

    try:
        header = pd.read_csv(path, nrows=0)
        cols = set(header.columns)

        if required_columns.issubset(cols):
            candidate_files.append(path)

    except Exception:
        continue


if len(candidate_files) == 0:

    print("\nNO VALID SUBJECT-LEVEL PERTURBATION FILE FOUND.")

    print("\nThe file must contain these columns:")
    for c in sorted(required_columns):
        print("  ", c)

    print("\nCSV files inspected:")
    all_csvs = list(BASE.rglob("*.csv"))

    for p in all_csvs[:100]:
        print("  ", p)

    raise FileNotFoundError(
        "No subject-level perturbation CSV containing the required columns was found."
    )


# Prefer files whose name explicitly indicates subject-level perturbation
preferred = [
    p for p in candidate_files
    if "subject" in p.name.lower()
    and "perturb" in p.name.lower()
]

if preferred:
    INPUT_FILE = sorted(preferred)[0]
else:
    INPUT_FILE = sorted(candidate_files)[0]


print("\nINPUT FILE FOUND:")
print(INPUT_FILE)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

df = pd.read_csv(INPUT_FILE)

print("\n" + "=" * 80)
print("INPUT VALIDATION")
print("=" * 80)

print(f"Rows:       {len(df):,}")
print(f"Columns:    {len(df.columns):,}")
print(f"Subjects:   {df['subject'].nunique():,}")
print(f"Features:   {df['feature'].nunique():,}")


# =============================================================================
# 3. REQUIRED COLUMN CHECK
# =============================================================================

missing = required_columns - set(df.columns)

if missing:
    raise RuntimeError(
        f"Required columns missing: {sorted(missing)}"
    )

print("\nRequired columns: PASS")


# =============================================================================
# 4. CLEAN NUMERIC DATA
# =============================================================================

effect_columns = [
    "remember_effect_size",
    "correct_effect_size",
]

for col in effect_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# =============================================================================
# 5. DUPLICATE CHECK
# =============================================================================

duplicate_count = df.duplicated(
    subset=["subject", "feature"]
).sum()

print("\n" + "=" * 80)
print("DUPLICATE CHECK")
print("=" * 80)

print(f"Duplicate subject-feature rows: {duplicate_count}")

if duplicate_count > 0:
    raise RuntimeError(
        "Duplicate subject-feature rows detected. "
        "Do not continue until they are resolved."
    )


# =============================================================================
# 6. SUBJECT-LEVEL ROBUSTNESS CALCULATION
# =============================================================================

print("\n" + "=" * 80)
print("SUBJECT-LEVEL ROBUSTNESS CALCULATION")
print("=" * 80)

results = []

for feature, g in df.groupby("feature"):

    remember = g["remember_effect_size"].dropna()
    correct = g["correct_effect_size"].dropna()

    # -------------------------------------------------------------------------
    # REMEMBER
    # -------------------------------------------------------------------------

    if len(remember) > 0:

        remember_mean = remember.mean()
        remember_median = remember.median()
        remember_std = remember.std(ddof=1)

        remember_positive_fraction = (
            (remember > 0).mean()
        )

        remember_negative_fraction = (
            (remember < 0).mean()
        )

        remember_direction_consistency = max(
            remember_positive_fraction,
            remember_negative_fraction,
        )

        if remember_positive_fraction > 0.5:
            remember_direction = "positive"
        elif remember_negative_fraction > 0.5:
            remember_direction = "negative"
        else:
            remember_direction = "mixed"

    else:

        remember_mean = np.nan
        remember_median = np.nan
        remember_std = np.nan
        remember_positive_fraction = np.nan
        remember_negative_fraction = np.nan
        remember_direction_consistency = np.nan
        remember_direction = "not_available"


    # -------------------------------------------------------------------------
    # CORRECT
    # -------------------------------------------------------------------------

    if len(correct) > 0:

        correct_mean = correct.mean()
        correct_median = correct.median()
        correct_std = correct.std(ddof=1)

        correct_positive_fraction = (
            (correct > 0).mean()
        )

        correct_negative_fraction = (
            (correct < 0).mean()
        )

        correct_direction_consistency = max(
            correct_positive_fraction,
            correct_negative_fraction,
        )

        if correct_positive_fraction > 0.5:
            correct_direction = "positive"
        elif correct_negative_fraction > 0.5:
            correct_direction = "negative"
        else:
            correct_direction = "mixed"

    else:

        correct_mean = np.nan
        correct_median = np.nan
        correct_std = np.nan
        correct_positive_fraction = np.nan
        correct_negative_fraction = np.nan
        correct_direction_consistency = np.nan
        correct_direction = "not_available"


    # -------------------------------------------------------------------------
    # ROBUSTNESS CLASSIFICATION
    # -------------------------------------------------------------------------

    def classify(consistency):

        if pd.isna(consistency):
            return "not_available"

        if consistency >= 0.80:
            return "strong"

        if consistency >= 0.65:
            return "moderate"

        if consistency >= 0.55:
            return "weak"

        return "mixed"


    results.append({
        "feature": feature,

        "remember_subject_count": len(remember),
        "remember_subject_effect_mean": remember_mean,
        "remember_subject_effect_median": remember_median,
        "remember_subject_effect_std": remember_std,
        "remember_positive_fraction": remember_positive_fraction,
        "remember_negative_fraction": remember_negative_fraction,
        "remember_direction_consistency": remember_direction_consistency,
        "remember_direction": remember_direction,
        "remember_robustness_class": classify(
            remember_direction_consistency
        ),

        "correct_subject_count": len(correct),
        "correct_subject_effect_mean": correct_mean,
        "correct_subject_effect_median": correct_median,
        "correct_subject_effect_std": correct_std,
        "correct_positive_fraction": correct_positive_fraction,
        "correct_negative_fraction": correct_negative_fraction,
        "correct_direction_consistency": correct_direction_consistency,
        "correct_direction": correct_direction,
        "correct_robustness_class": classify(
            correct_direction_consistency
        ),
    })


result_df = pd.DataFrame(results)


# =============================================================================
# 7. ADD GLOBAL ROBUSTNESS SUMMARY
# =============================================================================

def combined_class(row):

    classes = [
        row["remember_robustness_class"],
        row["correct_robustness_class"],
    ]

    if "strong" in classes:
        return "strong"

    if classes.count("moderate") == 2:
        return "moderate"

    if "moderate" in classes:
        return "moderate"

    if classes.count("weak") == 2:
        return "weak"

    if "weak" in classes:
        return "weak"

    if "mixed" in classes:
        return "mixed_weak"

    return "not_available"


result_df["overall_robustness_class"] = result_df.apply(
    combined_class,
    axis=1
)


# =============================================================================
# 8. FINAL QC
# =============================================================================

numeric_columns = result_df.select_dtypes(
    include=[np.number]
).columns

nan_count = int(
    result_df[numeric_columns].isna().sum().sum()
)

inf_count = int(
    np.isinf(
        result_df[numeric_columns].to_numpy(
            dtype=float,
            na_value=np.nan
        )
    ).sum()
)

duplicate_features = result_df["feature"].duplicated().sum()


print("\n" + "=" * 80)
print("FINAL ROBUSTNESS QC")
print("=" * 80)

print(f"Subjects:                  {df['subject'].nunique():,}")
print(f"Features analyzed:         {df['feature'].nunique():,}")
print(f"Result rows:               {len(result_df):,}")
print(f"NaN numeric values:        {nan_count:,}")
print(f"Inf numeric values:        {inf_count:,}")
print(f"Duplicate features:        {duplicate_features:,}")


# =============================================================================
# 9. SAVE
# =============================================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


qc = pd.DataFrame({
    "metric": [
        "input_rows",
        "input_columns",
        "subjects",
        "features",
        "result_rows",
        "nan_numeric_values",
        "inf_numeric_values",
        "duplicate_features",
    ],
    "value": [
        len(df),
        len(df.columns),
        df["subject"].nunique(),
        df["feature"].nunique(),
        len(result_df),
        nan_count,
        inf_count,
        duplicate_features,
    ]
})

qc.to_csv(
    QC_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =============================================================================
# 10. FINAL STATUS
# =============================================================================

print("\n" + "=" * 80)
print("SUBJECT-LEVEL PERTURBATION / WHAT-IF ROBUSTNESS V6 COMPLETE")
print("=" * 80)

print("\nSaved:")
print(OUTPUT_FILE)
print(QC_FILE)

if nan_count == 0 and inf_count == 0 and duplicate_features == 0:
    print("\nSTATUS: PASS - SUBJECT-LEVEL ROBUSTNESS V6 CREATED")
else:
    print("\nSTATUS: REVIEW_REQUIRED")