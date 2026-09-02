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
    / "ml_results"
    / "feature_sanity_v2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_REPORT = OUTPUT_DIR / "scientific_feature_sanity_v2.csv"
OUTPUT_QC = OUTPUT_DIR / "scientific_feature_sanity_v2_qc.csv"


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("SCIENTIFIC FEATURE SANITY CHECK V2")
print("=" * 80)

df = pd.read_csv(INPUT)

print(f"Input rows:      {len(df):,}")
print(f"Input columns:   {len(df.columns):,}")


# =============================================================================
# IDENTIFY KEYS
# =============================================================================

KEY_COLUMNS = [
    "subject",
    "run",
    "epoch",
]

missing_keys = [c for c in KEY_COLUMNS if c not in df.columns]

if missing_keys:
    raise RuntimeError(
        f"Missing required key columns: {missing_keys}"
    )


# =============================================================================
# SCIENTIFIC FEATURE INVENTORY
# =============================================================================

EXCLUDE_COLUMNS = {
    # identifiers
    "subject",
    "run",
    "epoch",
    "trial",
    "file",

    # split / metadata
    "split",

    # behavioral targets
    "target",
    "target_label",
    "target_remember",
    "target_correct",

    # behavior-derived variables
    "is_correct",
    "is_remembered",
    "is_ignored",

    # trial metadata
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


numeric_columns_all = df.select_dtypes(
    include=[np.number]
).columns.tolist()

scientific_features = [
    c for c in numeric_columns_all
    if c not in EXCLUDE_COLUMNS
]


print()
print("=" * 80)
print("FEATURE INVENTORY")
print("=" * 80)

print(f"Numeric columns:   {len(numeric_columns_all)}")
print(f"Scientific features: {len(scientific_features)}")
print()

print(scientific_features)


if len(scientific_features) == 0:
    raise RuntimeError(
        "No scientific numeric features were found."
    )


# =============================================================================
# BASIC FEATURE QC
# =============================================================================

print()
print("=" * 80)
print("BASIC FEATURE QC")
print("=" * 80)

feature_matrix = df[scientific_features].apply(
    pd.to_numeric,
    errors="coerce"
)

nan_count = int(feature_matrix.isna().sum().sum())

inf_count = int(
    np.isinf(
        feature_matrix.to_numpy(dtype=np.float64, copy=True)
    ).sum()
)

duplicate_key_count = int(
    df.duplicated(
        subset=KEY_COLUMNS,
        keep=False
    ).sum()
)

print(f"NaN values:          {nan_count:,}")
print(f"Inf values:          {inf_count:,}")
print(f"Duplicate keys:      {duplicate_key_count:,}")


# =============================================================================
# FEATURE DISTRIBUTION
# =============================================================================

print()
print("=" * 80)
print("FEATURE DISTRIBUTION STATISTICS")
print("=" * 80)

constant_records = []
low_variance_records = []

for feature in scientific_features:

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64,
        copy=True
    )

    finite_values = values[np.isfinite(values)]

    if len(finite_values) == 0:
        unique_count = 0
        std_value = np.nan

    else:
        unique_count = int(
            np.unique(finite_values).size
        )

        std_value = float(
            np.std(finite_values)
        )

    if unique_count <= 1:
        constant_records.append(
            {
                "feature": feature,
                "unique_values": unique_count,
            }
        )

    if np.isfinite(std_value) and std_value == 0:
        low_variance_records.append(
            {
                "feature": feature,
                "std": std_value,
            }
        )


constant_df = pd.DataFrame(
    constant_records
)

low_variance_df = pd.DataFrame(
    low_variance_records
)


print()
print("Constant features:")

if len(constant_df) > 0:
    print(
        constant_df.to_string(index=False)
    )
else:
    print("None")


print()
print("Low variance features:")

if len(low_variance_df) > 0:
    print(
        low_variance_df.to_string(index=False)
    )
else:
    print("None")


# =============================================================================
# SUBJECT-LEVEL FEATURE VARIABILITY
# =============================================================================

print()
print("=" * 80)
print("SUBJECT-LEVEL FEATURE VARIABILITY")
print("=" * 80)

subject_variability_records = []

for feature in scientific_features:

    subject_std = (
        df.groupby("subject")[feature]
        .std()
        .replace([np.inf, -np.inf], np.nan)
    )

    valid_subject_std = subject_std.dropna()

    if len(valid_subject_std) == 0:
        mean_subject_std = np.nan
        median_subject_std = np.nan
        zero_subject_std_count = np.nan

    else:
        mean_subject_std = float(
            valid_subject_std.mean()
        )

        median_subject_std = float(
            valid_subject_std.median()
        )

        zero_subject_std_count = int(
            (valid_subject_std == 0).sum()
        )

    subject_variability_records.append(
        {
            "feature": feature,
            "mean_subject_std": mean_subject_std,
            "median_subject_std": median_subject_std,
            "zero_subject_std_subjects": zero_subject_std_count,
        }
    )


subject_variability_df = pd.DataFrame(
    subject_variability_records
)


# =============================================================================
# FEATURE CORRELATION
# =============================================================================

print()
print("=" * 80)
print("FEATURE CORRELATION")
print("=" * 80)

# IMPORTANT:
# pandas may return a read-only NumPy view in some versions.
# Therefore ALWAYS make an explicit writable copy before
# np.fill_diagonal().

correlation_df = feature_matrix.corr(
    method="pearson"
)

correlation_matrix = np.array(
    correlation_df.to_numpy(
        dtype=np.float64,
        copy=True
    ),
    dtype=np.float64,
    copy=True
)

# Make absolutely sure the array is writable.
correlation_matrix.setflags(write=True)

np.fill_diagonal(
    correlation_matrix,
    np.nan
)


# =============================================================================
# HIGH CORRELATION PAIRS
# =============================================================================

HIGH_CORR_THRESHOLD = 0.95

high_corr_records = []

n_features = len(scientific_features)

for i in range(n_features):

    for j in range(i + 1, n_features):

        corr_value = correlation_matrix[i, j]

        if not np.isfinite(corr_value):
            continue

        abs_corr = abs(corr_value)

        if abs_corr >= HIGH_CORR_THRESHOLD:

            high_corr_records.append(
                {
                    "feature_1": scientific_features[i],
                    "feature_2": scientific_features[j],
                    "correlation": float(corr_value),
                    "abs_correlation": float(abs_corr),
                }
            )


high_corr_df = pd.DataFrame(
    high_corr_records
)

if len(high_corr_df) > 0:

    high_corr_df = high_corr_df.sort_values(
        "abs_correlation",
        ascending=False
    ).reset_index(drop=True)


print()
print(
    f"High-correlation pairs (|r| >= {HIGH_CORR_THRESHOLD}): "
    f"{len(high_corr_df):,}"
)

if len(high_corr_df) > 0:
    print(
        high_corr_df.head(30).to_string(
            index=False
        )
    )
else:
    print("None")


# =============================================================================
# FEATURE-LEVEL SUMMARY
# =============================================================================

feature_summary_records = []

for feature in scientific_features:

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64,
        copy=True
    )

    finite_values = values[np.isfinite(values)]

    if len(finite_values) == 0:

        mean_value = np.nan
        std_value = np.nan
        min_value = np.nan
        max_value = np.nan

    else:

        mean_value = float(
            np.mean(finite_values)
        )

        std_value = float(
            np.std(finite_values)
        )

        min_value = float(
            np.min(finite_values)
        )

        max_value = float(
            np.max(finite_values)
        )

    subject_info = subject_variability_df[
        subject_variability_df["feature"] == feature
    ]

    if len(subject_info) > 0:

        mean_subject_std = float(
            subject_info.iloc[0]["mean_subject_std"]
        )

        median_subject_std = float(
            subject_info.iloc[0]["median_subject_std"]
        )

        zero_subject_std = int(
            subject_info.iloc[0][
                "zero_subject_std_subjects"
            ]
        )

    else:

        mean_subject_std = np.nan
        median_subject_std = np.nan
        zero_subject_std = np.nan

    feature_summary_records.append(
        {
            "feature": feature,
            "mean": mean_value,
            "std": std_value,
            "min": min_value,
            "max": max_value,
            "mean_subject_std": mean_subject_std,
            "median_subject_std": median_subject_std,
            "zero_subject_std_subjects": zero_subject_std,
        }
    )


feature_summary_df = pd.DataFrame(
    feature_summary_records
)


# =============================================================================
# TARGET-DERIVED FEATURE CHECK
# =============================================================================

TARGET_KEYWORDS = [
    "target",
    "correct",
    "remember",
    "ignore",
    "behavior",
    "response",
    "feedback",
    "probe",
]


target_like_features = []

for feature in scientific_features:

    feature_lower = feature.lower()

    matched_keywords = [
        keyword
        for keyword in TARGET_KEYWORDS
        if keyword in feature_lower
    ]

    if matched_keywords:

        target_like_features.append(
            {
                "feature": feature,
                "keywords": ",".join(
                    matched_keywords
                ),
            }
        )


target_like_df = pd.DataFrame(
    target_like_features
)


print()
print("=" * 80)
print("TARGET-LIKE SCIENTIFIC FEATURE NAME CHECK")
print("=" * 80)

if len(target_like_df) > 0:
    print(
        target_like_df.to_string(
            index=False
        )
    )
else:
    print("None")


# =============================================================================
# QC SUMMARY
# =============================================================================

qc_records = [
    {
        "metric": "input_rows",
        "value": int(len(df)),
    },
    {
        "metric": "input_columns",
        "value": int(len(df.columns)),
    },
    {
        "metric": "numeric_columns",
        "value": int(len(numeric_columns_all)),
    },
    {
        "metric": "scientific_features",
        "value": int(len(scientific_features)),
    },
    {
        "metric": "nan_values",
        "value": int(nan_count),
    },
    {
        "metric": "inf_values",
        "value": int(inf_count),
    },
    {
        "metric": "duplicate_keys",
        "value": int(duplicate_key_count),
    },
    {
        "metric": "constant_features",
        "value": int(len(constant_df)),
    },
    {
        "metric": "low_variance_features",
        "value": int(len(low_variance_df)),
    },
    {
        "metric": "high_correlation_pairs",
        "value": int(len(high_corr_df)),
    },
    {
        "metric": "target_like_scientific_features",
        "value": int(len(target_like_df)),
    },
]


qc_df = pd.DataFrame(
    qc_records
)


# =============================================================================
# SAVE FEATURE REPORT
# =============================================================================

feature_summary_df.to_csv(
    OUTPUT_REPORT,
    index=False
)

qc_df.to_csv(
    OUTPUT_QC,
    index=False
)


# =============================================================================
# FINAL STATUS
# =============================================================================

print()
print("=" * 80)
print("SCIENTIFIC FEATURE SANITY CHECK COMPLETE")
print("=" * 80)

print()
print(f"Scientific features:       {len(scientific_features)}")
print(f"NaN values:                {nan_count:,}")
print(f"Inf values:                {inf_count:,}")
print(f"Duplicate keys:            {duplicate_key_count:,}")
print(f"Constant features:         {len(constant_df):,}")
print(f"High-correlation pairs:    {len(high_corr_df):,}")
print(f"Target-like features:      {len(target_like_df):,}")

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_REPORT)
print(OUTPUT_QC)

print()
print("=" * 80)
print("STATUS: PASS - SCIENTIFIC FEATURE SANITY CHECK COMPLETED")
print("=" * 80)