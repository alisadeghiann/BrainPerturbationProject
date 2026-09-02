from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PERTURBATION / WHAT-IF ML LEAKAGE AUDIT V1
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "perturbation_subject_effects.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "ml_leakage_audit_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("PERTURBATION / WHAT-IF ML LEAKAGE AUDIT V1")
print("=" * 90)
print("Project root:", BASE)

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 90)
print("SEARCHING FOR SUBJECT-LEVEL PERTURBATION DATA")
print("=" * 90)

if not INPUT.exists():
    raise FileNotFoundError(
        "Input file was not found:\n" + str(INPUT)
    )

print("INPUT FOUND:")
print(INPUT)

df = pd.read_csv(INPUT)

print("\n" + "=" * 90)
print("DATASET SUMMARY")
print("=" * 90)

print("Rows:", len(df))
print("Columns:", len(df.columns))

if "subject" in df.columns:
    print("Subjects:", df["subject"].nunique())
else:
    raise ValueError(
        "Required column 'subject' was not found."
    )

print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# TARGET AUDIT
# ============================================================

print("\n" + "=" * 90)
print("TARGET AUDIT")
print("=" * 90)

TARGET_COLUMNS = []

if "target_remember" in df.columns:
    TARGET_COLUMNS.append("target_remember")

if "target_correct" in df.columns:
    TARGET_COLUMNS.append("target_correct")

if len(TARGET_COLUMNS) == 0:
    raise ValueError(
        "Neither target_remember nor target_correct was found."
    )

for target in TARGET_COLUMNS:
    print(
        target,
        "| unique =",
        df[target].nunique(dropna=True),
        "| missing =",
        df[target].isna().sum()
    )

# ============================================================
# TARGET-LIKE COLUMN AUDIT
# ============================================================

print("\n" + "=" * 90)
print("TARGET-LIKE COLUMN AUDIT")
print("=" * 90)

TARGET_KEYWORDS = [
    "target",
    "correct",
    "remember",
    "memory",
    "accuracy",
    "response",
    "label",
    "condition"
]

target_like_columns = []

for column in df.columns:

    column_lower = column.lower()

    for keyword in TARGET_KEYWORDS:

        if keyword in column_lower:
            target_like_columns.append(column)
            break

target_like_columns = sorted(
    set(target_like_columns)
)

for column in target_like_columns:
    print(column)

pd.DataFrame(
    {
        "target_like_column": target_like_columns
    }
).to_csv(
    OUTPUT_DIR / "target_like_columns.csv",
    index=False
)

# ============================================================
# META COLUMNS
# ============================================================

META_COLUMNS = {
    "file",
    "subject",
    "run",
    "epoch",
    "trial",
    "memory_cond",
    "probe_type",
    "probe_letter",
    "target_label",
    "target_remember",
    "target_correct"
}

# ============================================================
# NUMERIC FEATURE DETECTION
# ============================================================

print("\n" + "=" * 90)
print("NUMERIC FEATURE AUDIT")
print("=" * 90)

NUMERIC_FEATURES = []

for column in df.columns:

    if column in META_COLUMNS:
        continue

    if not pd.api.types.is_numeric_dtype(df[column]):
        continue

    if df[column].nunique(dropna=True) <= 1:
        continue

    NUMERIC_FEATURES.append(column)

print(
    "Numeric candidate features:",
    len(NUMERIC_FEATURES)
)

# ============================================================
# FEATURE / TARGET CORRELATION
# ============================================================

print("\n" + "=" * 90)
print("FEATURE-TARGET CORRELATION AUDIT")
print("=" * 90)

correlation_rows = []

for target in TARGET_COLUMNS:

    y = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    for feature in NUMERIC_FEATURES:

        x = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        valid = x.notna() & y.notna()

        if valid.sum() < 10:
            correlation = np.nan
        else:
            correlation = x[valid].corr(
                y[valid]
            )

        correlation_rows.append(
            {
                "target": target,
                "feature": feature,
                "pearson_correlation": correlation,
                "absolute_correlation": (
                    abs(correlation)
                    if pd.notna(correlation)
                    else np.nan
                )
            }
        )

correlation_df = pd.DataFrame(
    correlation_rows
)

correlation_df = correlation_df.sort_values(
    [
        "target",
        "absolute_correlation"
    ],
    ascending=[True, False]
)

correlation_file = (
    OUTPUT_DIR
    / "feature_target_correlations.csv"
)

correlation_df.to_csv(
    correlation_file,
    index=False
)

for target in TARGET_COLUMNS:

    print("\nTARGET:", target)

    subset = correlation_df[
        correlation_df["target"] == target
    ].head(20)

    print(
        subset.to_string(index=False)
    )

# ============================================================
# NEAR-PERFECT CORRELATION AUDIT
# ============================================================

print("\n" + "=" * 90)
print("NEAR-PERFECT TARGET CORRELATION AUDIT")
print("=" * 90)

near_perfect = correlation_df[
    correlation_df["absolute_correlation"] >= 0.95
].copy()

near_perfect_file = (
    OUTPUT_DIR
    / "near_perfect_target_features.csv"
)

near_perfect.to_csv(
    near_perfect_file,
    index=False
)

print(
    "Features with absolute correlation >= 0.95:",
    len(near_perfect)
)

if len(near_perfect) > 0:
    print(
        near_perfect.to_string(index=False)
    )
else:
    print("NONE")

# ============================================================
# WITHIN-SUBJECT TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)
print("WITHIN-SUBJECT TARGET DISTRIBUTION")
print("=" * 90)

subject_rows = []

for subject, group in df.groupby("subject"):

    row = {
        "subject": subject,
        "n_rows": len(group)
    }

    for target in TARGET_COLUMNS:

        values = pd.to_numeric(
            group[target],
            errors="coerce"
        ).dropna()

        row[target + "_unique"] = (
            values.nunique()
        )

        row[target + "_mean"] = (
            values.mean()
        )

        row[target + "_class_0"] = (
            values == 0
        ).sum()

        row[target + "_class_1"] = (
            values == 1
        ).sum()

    subject_rows.append(row)

subject_df = pd.DataFrame(
    subject_rows
)

subject_file = (
    OUTPUT_DIR
    / "within_subject_target_distribution.csv"
)

subject_df.to_csv(
    subject_file,
    index=False
)

print(
    subject_df.to_string(index=False)
)

# ============================================================
# DUPLICATE AUDIT
# ============================================================

print("\n" + "=" * 90)
print("DUPLICATE AUDIT")
print("=" * 90)

exact_duplicates = int(
    df.duplicated().sum()
)

print(
    "Exact duplicate rows:",
    exact_duplicates
)

identifier_columns = [
    "subject",
    "run",
    "epoch",
    "trial"
]

if all(
    column in df.columns
    for column in identifier_columns
):

    duplicate_identifiers = int(
        df.duplicated(
            subset=identifier_columns
        ).sum()
    )

    print(
        "Duplicate subject/run/epoch/trial rows:",
        duplicate_identifiers
    )

else:

    duplicate_identifiers = np.nan

    print(
        "Subject/run/epoch/trial identifier:",
        "NOT AVAILABLE"
    )

# ============================================================
# FEATURE UNIQUENESS
# ============================================================

print("\n" + "=" * 90)
print("FEATURE UNIQUENESS AUDIT")
print("=" * 90)

uniqueness_rows = []

for feature in NUMERIC_FEATURES:

    unique_values = df[feature].nunique(
        dropna=True
    )

    uniqueness_rows.append(
        {
            "feature": feature,
            "unique_values": unique_values,
            "rows": len(df),
            "unique_fraction": (
                unique_values / len(df)
            )
        }
    )

uniqueness_df = pd.DataFrame(
    uniqueness_rows
)

uniqueness_df = uniqueness_df.sort_values(
    "unique_fraction"
)

uniqueness_file = (
    OUTPUT_DIR
    / "feature_uniqueness_audit.csv"
)

uniqueness_df.to_csv(
    uniqueness_file,
    index=False
)

print(
    uniqueness_df.head(20).to_string(
        index=False
    )
)

# ============================================================
# MEMORY CONDITION AUDIT
# ============================================================

if "memory_cond" in df.columns:

    print("\n" + "=" * 90)
    print("MEMORY CONDITION AUDIT")
    print("=" * 90)

    if "target_remember" in df.columns:

        memory_table = pd.crosstab(
            df["memory_cond"],
            df["target_remember"]
        )

        print(
            memory_table.to_string()
        )

        memory_table.to_csv(
            OUTPUT_DIR
            / "memory_condition_vs_remember.csv"
        )

# ============================================================
# TARGET LABEL AUDIT
# ============================================================

if "target_label" in df.columns:

    print("\n" + "=" * 90)
    print("TARGET LABEL AUDIT")
    print("=" * 90)

    if "target_remember" in df.columns:

        label_table = pd.crosstab(
            df["target_label"],
            df["target_remember"]
        )

        print(
            label_table.to_string()
        )

        label_table.to_csv(
            OUTPUT_DIR
            / "target_label_vs_remember.csv"
        )

# ============================================================
# DIRECT TARGET ENCODING CHECK
# ============================================================

print("\n" + "=" * 90)
print("DIRECT TARGET ENCODING CHECK")
print("=" * 90)

encoding_rows = []

for feature in NUMERIC_FEATURES:

    feature_values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    for target in TARGET_COLUMNS:

        target_values = pd.to_numeric(
            df[target],
            errors="coerce"
        )

        valid = (
            feature_values.notna()
            & target_values.notna()
        )

        if valid.sum() < 10:
            continue

        grouped = pd.DataFrame(
            {
                "feature": feature_values[valid],
                "target": target_values[valid]
            }
        ).groupby("target")["feature"].agg(
            ["mean", "std", "min", "max"]
        )

        encoding_rows.append(
            {
                "feature": feature,
                "target": target,
                "target_groups": len(grouped),
                "group_mean_difference": (
                    grouped["mean"].max()
                    - grouped["mean"].min()
                )
            }
        )

encoding_df = pd.DataFrame(
    encoding_rows
)

encoding_df.to_csv(
    OUTPUT_DIR
    / "direct_target_encoding_audit.csv",
    index=False
)

# ============================================================
# FINAL QC
# ============================================================

print("\n" + "=" * 90)
print("FINAL LEAKAGE AUDIT QC")
print("=" * 90)

print(
    "Rows:",
    len(df)
)

print(
    "Subjects:",
    df["subject"].nunique()
)

print(
    "Numeric features:",
    len(NUMERIC_FEATURES)
)

print(
    "Targets:",
    len(TARGET_COLUMNS)
)

print(
    "Near-perfect |r| >= 0.95:",
    len(near_perfect)
)

print(
    "Exact duplicate rows:",
    exact_duplicates
)

# ============================================================
# SCIENTIFIC STATUS
# ============================================================

print("\n" + "=" * 90)
print("SCIENTIFIC STATUS")
print("=" * 90)

if len(near_perfect) > 0:

    print(
        "STATUS: REVIEW_REQUIRED"
    )

    print(
        "Potential target leakage detected."
    )

    print(
        "Do NOT interpret 100 percent ML performance "
        "as valid scientific predictive evidence yet."
    )

else:

    print(
        "STATUS: PRELIMINARY_PASS"
    )

    print(
        "No obvious near-perfect feature-target "
        "correlation was detected."
    )

    print(
        "Further subject-held-out validation is still required."
    )

# ============================================================
# SAVED FILES
# ============================================================

print("\n" + "=" * 90)
print("SAVED")
print("=" * 90)

print(
    OUTPUT_DIR
    / "target_like_columns.csv"
)

print(
    correlation_file
)

print(
    near_perfect_file
)

print(
    subject_file
)

print(
    uniqueness_file
)

print(
    OUTPUT_DIR
    / "direct_target_encoding_audit.csv"
)

print("\n" + "=" * 90)
print("PERTURBATION / WHAT-IF ML LEAKAGE AUDIT V1 COMPLETE")
print("=" * 90)