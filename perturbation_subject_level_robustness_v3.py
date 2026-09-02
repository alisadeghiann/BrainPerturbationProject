# perturbation_subject_level_robustness_v3.py

from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT_DATA = (
    BASE
    / "features"
    / "ml_ready_v2"
    / "feature_selection"
    / "ml_ready_dataset_v2_selected.csv"
)

INPUT_STATS = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "statistical_v2"
    / "perturbation_statistical_results_v2.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "robustness_v3"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_RESULTS = (
    OUTPUT_DIR
    / "subject_level_perturbation_robustness_v3.csv"
)

OUTPUT_QC = (
    OUTPUT_DIR
    / "subject_level_perturbation_robustness_qc_v3.csv"
)

# =============================================================================
# HEADER
# =============================================================================

print("=" * 90)
print("SUBJECT-LEVEL PERTURBATION / WHAT-IF ROBUSTNESS V3")
print("=" * 90)

# =============================================================================
# LOAD DATA
# =============================================================================

if not INPUT_DATA.exists():
    raise FileNotFoundError(f"Input dataset not found:\n{INPUT_DATA}")

if not INPUT_STATS.exists():
    raise FileNotFoundError(f"Statistical results not found:\n{INPUT_STATS}")

df = pd.read_csv(INPUT_DATA)
stats = pd.read_csv(INPUT_STATS)

print(f"Dataset rows:        {len(df):,}")
print(f"Dataset columns:     {len(df.columns):,}")
print(f"Statistical rows:    {len(stats):,}")

# =============================================================================
# IDENTIFY SUBJECT COLUMN
# =============================================================================

subject_candidates = [
    "subject",
    "sub",
    "participant",
    "participant_id",
    "subject_id"
]

subject_col = None

for col in subject_candidates:
    if col in df.columns:
        subject_col = col
        break

if subject_col is None:
    raise RuntimeError(
        "Could not identify subject column. "
        f"Available columns:\n{list(df.columns)}"
    )

print(f"Subject column:      {subject_col}")

# =============================================================================
# IDENTIFY TARGETS
# =============================================================================

target_candidates = {
    "remember": [
        "target_remember",
        "remember",
        "is_remembered"
    ],
    "correct": [
        "target_correct",
        "correct",
        "is_correct"
    ]
}

targets = {}

for target_name, candidates in target_candidates.items():

    found = None

    for col in candidates:
        if col in df.columns:
            found = col
            break

    if found is not None:
        targets[target_name] = found

print()
print("=" * 90)
print("TARGETS")
print("=" * 90)

for target_name, target_col in targets.items():
    print(f"{target_name:12s}: {target_col}")

if not targets:
    raise RuntimeError("No valid target columns found.")

# =============================================================================
# IDENTIFY SCIENTIFIC FEATURES
# =============================================================================

excluded = {
    subject_col,
    "run",
    "trial",
    "epoch",
    "split",
    "target_label",
    "target_remember",
    "target_correct",
    "remember",
    "correct",
    "is_remembered",
    "is_correct",
    "is_ignored"
}

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

scientific_features = [
    col for col in numeric_cols
    if col not in excluded
]

print()
print("=" * 90)
print("SCIENTIFIC FEATURES")
print("=" * 90)

print(f"Scientific numeric features: {len(scientific_features)}")

# =============================================================================
# BASIC DATA QC
# =============================================================================

print()
print("=" * 90)
print("BASIC DATA QC")
print("=" * 90)

nan_count = int(
    df[scientific_features]
    .isna()
    .sum()
    .sum()
)

inf_count = int(
    np.isinf(
        df[scientific_features]
        .to_numpy(dtype=float)
    ).sum()
)

print(f"NaN values:  {nan_count:,}")
print(f"Inf values:  {inf_count:,}")

if nan_count > 0 or inf_count > 0:
    raise RuntimeError(
        "Scientific feature data contain NaN or Inf values."
    )

# =============================================================================
# SUBJECT COUNTS
# =============================================================================

subjects = sorted(
    df[subject_col]
    .dropna()
    .astype(str)
    .unique()
)

print()
print("=" * 90)
print("SUBJECT COVERAGE")
print("=" * 90)

print(f"Subjects: {len(subjects)}")

# =============================================================================
# SUBJECT-LEVEL EFFECTS
#
# For every scientific feature and target:
#
#   subject mean in target=1
#   subject mean in target=0
#   within-subject difference
#
# This gives an actual subject-level robustness distribution.
# =============================================================================

results = []

for target_name, target_col in targets.items():

    print()
    print("-" * 90)
    print(f"TARGET: {target_name.upper()}")
    print("-" * 90)

    target_values = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    if target_values.isna().any():
        raise RuntimeError(
            f"Target {target_col} contains invalid values."
        )

    target_values = target_values.astype(int)

    print(
        f"Group 0: {(target_values == 0).sum():,} rows"
    )

    print(
        f"Group 1: {(target_values == 1).sum():,} rows"
    )

    for feature in scientific_features:

        subject_effects = []

        for subject in subjects:

            mask_subject = (
                df[subject_col].astype(str) == str(subject)
            )

            x = df.loc[mask_subject, feature]
            y = target_values.loc[mask_subject]

            group0 = x[y == 0]
            group1 = x[y == 1]

            if len(group0) == 0 or len(group1) == 0:
                continue

            mean0 = group0.mean()
            mean1 = group1.mean()

            effect = mean1 - mean0

            if not np.isfinite(effect):
                continue

            subject_effects.append(effect)

        if len(subject_effects) == 0:
            continue

        effects = np.asarray(
            subject_effects,
            dtype=float
        )

        median_effect = float(
            np.median(effects)
        )

        mean_effect = float(
            np.mean(effects)
        )

        std_effect = float(
            np.std(effects, ddof=1)
        ) if len(effects) > 1 else np.nan

        q25 = float(
            np.percentile(effects, 25)
        )

        q75 = float(
            np.percentile(effects, 75)
        )

        positive_subjects = int(
            np.sum(effects > 0)
        )

        negative_subjects = int(
            np.sum(effects < 0)
        )

        zero_subjects = int(
            np.sum(effects == 0)
        )

        n_subjects = len(effects)

        positive_fraction = (
            positive_subjects / n_subjects
        )

        negative_fraction = (
            negative_subjects / n_subjects
        )

        # ---------------------------------------------------------------------
        # ROBUSTNESS CLASSIFICATION
        # ---------------------------------------------------------------------

        if positive_fraction >= 0.70:
            direction_consistency = "positive_robust"

        elif negative_fraction >= 0.70:
            direction_consistency = "negative_robust"

        elif (
            positive_fraction >= 0.55
            or negative_fraction >= 0.55
        ):
            direction_consistency = "mixed_weak"

        else:
            direction_consistency = "unstable"

        # ---------------------------------------------------------------------
        # MEDIAN DIRECTION
        # ---------------------------------------------------------------------

        if median_effect > 0:
            median_direction = "positive"

        elif median_effect < 0:
            median_direction = "negative"

        else:
            median_direction = "zero"

        results.append({
            "target": target_name,
            "feature": feature,
            "n_subjects": n_subjects,
            "mean_subject_effect": mean_effect,
            "median_subject_effect": median_effect,
            "std_subject_effect": std_effect,
            "q25_subject_effect": q25,
            "q75_subject_effect": q75,
            "positive_subjects": positive_subjects,
            "negative_subjects": negative_subjects,
            "zero_subjects": zero_subjects,
            "positive_fraction": positive_fraction,
            "negative_fraction": negative_fraction,
            "median_direction": median_direction,
            "direction_consistency": direction_consistency
        })

# =============================================================================
# RESULTS DATAFRAME
# =============================================================================

robustness = pd.DataFrame(results)

if robustness.empty:
    raise RuntimeError(
        "No subject-level robustness results were generated."
    )

# =============================================================================
# SORT
# =============================================================================

robustness["consistency_score"] = np.maximum(
    robustness["positive_fraction"],
    robustness["negative_fraction"]
)

robustness = robustness.sort_values(
    [
        "target",
        "consistency_score",
        "n_subjects"
    ],
    ascending=[True, False, False]
)

robustness = robustness.drop(
    columns=["consistency_score"]
)

# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 90)
print("ROBUSTNESS SUMMARY")
print("=" * 90)

for target_name in robustness["target"].unique():

    sub = robustness[
        robustness["target"] == target_name
    ]

    print()
    print(f"TARGET: {target_name.upper()}")

    print(
        f"Features analyzed: "
        f"{len(sub):,}"
    )

    print(
        f"Positive robust: "
        f"{(sub['direction_consistency'] == 'positive_robust').sum():,}"
    )

    print(
        f"Negative robust: "
        f"{(sub['direction_consistency'] == 'negative_robust').sum():,}"
    )

    print(
        f"Mixed weak: "
        f"{(sub['direction_consistency'] == 'mixed_weak').sum():,}"
    )

    print(
        f"Unstable: "
        f"{(sub['direction_consistency'] == 'unstable').sum():,}"
    )

# =============================================================================
# TOP ROBUST FEATURES
# =============================================================================

print()
print("=" * 90)
print("TOP SUBJECT-ROBUST FEATURES")
print("=" * 90)

display_cols = [
    "target",
    "feature",
    "n_subjects",
    "median_subject_effect",
    "positive_fraction",
    "negative_fraction",
    "direction_consistency"
]

print(
    robustness[display_cols]
    .head(30)
    .to_string(index=False)
)

# =============================================================================
# FINAL QC
# =============================================================================

numeric_result_cols = robustness.select_dtypes(
    include=[np.number]
).columns

result_nan = int(
    robustness[numeric_result_cols]
    .isna()
    .sum()
    .sum()
)

result_inf = int(
    np.isinf(
        robustness[numeric_result_cols]
        .to_numpy(dtype=float)
    ).sum()
)

print()
print("=" * 90)
print("FINAL ROBUSTNESS QC")
print("=" * 90)

print(
    f"Subjects:              {len(subjects)}"
)

print(
    f"Features analyzed:     {len(scientific_features)}"
)

print(
    f"Targets analyzed:      {len(targets)}"
)

print(
    f"Result rows:            {len(robustness):,}"
)

print(
    f"NaN numeric values:     {result_nan:,}"
)

print(
    f"Inf numeric values:     {result_inf:,}"
)

# =============================================================================
# SAVE
# =============================================================================

robustness.to_csv(
    OUTPUT_RESULTS,
    index=False
)

qc = pd.DataFrame({
    "metric": [
        "input_rows",
        "input_columns",
        "subjects",
        "scientific_features",
        "targets",
        "result_rows",
        "input_nan_values",
        "input_inf_values",
        "result_nan_values",
        "result_inf_values"
    ],
    "value": [
        len(df),
        len(df.columns),
        len(subjects),
        len(scientific_features),
        len(targets),
        len(robustness),
        nan_count,
        inf_count,
        result_nan,
        result_inf
    ]
})

qc.to_csv(
    OUTPUT_QC,
    index=False
)

# =============================================================================
# FINAL STATUS
# =============================================================================

print()
print("=" * 90)
print("SUBJECT-LEVEL PERTURBATION / WHAT-IF ROBUSTNESS V3 COMPLETE")
print("=" * 90)

print()
print("Results:")
print(OUTPUT_RESULTS)

print()
print("QC:")
print(OUTPUT_QC)

print()
print("=" * 90)

if (
    result_nan == 0
    and result_inf == 0
    and not robustness.empty
):
    print(
        "STATUS: PASS - SUBJECT-LEVEL ROBUSTNESS ANALYSIS CREATED"
    )
else:
    print(
        "STATUS: REVIEW_REQUIRED"
    )