from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# =============================================================================
# PERTURBATION STATISTICAL ANALYSIS V2
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "ml_ready_v2"
    / "feature_selection"
    / "ml_ready_dataset_v2_selected.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "statistical_v2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_STATS = (
    OUTPUT_DIR
    / "perturbation_statistical_results_v2.csv"
)

OUTPUT_SUBJECT = (
    OUTPUT_DIR
    / "perturbation_subject_robustness_v2.csv"
)

OUTPUT_QC = (
    OUTPUT_DIR
    / "perturbation_statistical_qc_v2.csv"
)

# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_COLUMNS = {
    "remember": "target_remember",
    "correct": "target_correct",
}

NON_FEATURE_COLUMNS = {
    "subject",
    "run",
    "trial",
    "epoch",
    "file",
    "target",
    "target_label",
    "target_remember",
    "target_correct",
    "split",
    "behavior_label",
    "behavior_outcome",
    "response_type",
    "feedback",
    "event_name",
    "event_code",
    "alignment_status",
    "is_correct",
    "is_remembered",
    "is_ignored",
}

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 90)
print("PERTURBATION STATISTICAL ANALYSIS V2")
print("=" * 90)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

print(f"Input rows:       {len(df):,}")
print(f"Input columns:    {len(df.columns):,}")

# =============================================================================
# VALIDATE TARGETS
# =============================================================================

print("\n" + "=" * 90)
print("TARGET VALIDATION")
print("=" * 90)

for name, col in TARGET_COLUMNS.items():

    if col not in df.columns:
        raise RuntimeError(
            f"Required target column missing: {col}"
        )

    print(
        f"{name}: {df[col].value_counts(dropna=False).to_dict()}"
    )

# =============================================================================
# IDENTIFY SCIENTIFIC FEATURES
# =============================================================================

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()

scientific_features = [
    col
    for col in numeric_columns
    if col not in NON_FEATURE_COLUMNS
]

# Remove obvious metadata
scientific_features = [
    col
    for col in scientific_features
    if col not in {
        "sfreq",
        "n_channels",
        "n_timepoints",
    }
]

print("\n" + "=" * 90)
print("SCIENTIFIC FEATURE INVENTORY")
print("=" * 90)

print(
    f"Scientific features: {len(scientific_features)}"
)

print(scientific_features)

if len(scientific_features) == 0:
    raise RuntimeError(
        "No scientific numeric features found."
    )

# =============================================================================
# BASIC FEATURE QC
# =============================================================================

print("\n" + "=" * 90)
print("FEATURE QC")
print("=" * 90)

feature_matrix = df[scientific_features].apply(
    pd.to_numeric,
    errors="coerce"
)

nan_count = int(
    feature_matrix.isna().sum().sum()
)

inf_count = int(
    np.isinf(
        feature_matrix.to_numpy(
            dtype=float,
            copy=True
        )
    ).sum()
)

print(f"NaN values: {nan_count}")
print(f"Inf values: {inf_count}")

if nan_count > 0:
    raise RuntimeError(
        "Scientific feature matrix contains NaN values."
    )

if inf_count > 0:
    raise RuntimeError(
        "Scientific feature matrix contains Inf values."
    )

# =============================================================================
# FEATURE-LEVEL STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "=" * 90)
print("FEATURE-LEVEL PERTURBATION ANALYSIS")
print("=" * 90)

results = []

for target_name, target_col in TARGET_COLUMNS.items():

    print(f"\nTARGET: {target_name.upper()}")

    y = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    valid_mask = y.notna()

    target_df = df.loc[valid_mask].copy()

    y_valid = y.loc[valid_mask]

    group_0_mask = y_valid == 0
    group_1_mask = y_valid == 1

    n0 = int(group_0_mask.sum())
    n1 = int(group_1_mask.sum())

    print(
        f"Group 0: {n0:,} rows"
    )

    print(
        f"Group 1: {n1:,} rows"
    )

    if n0 < 2 or n1 < 2:
        raise RuntimeError(
            f"Insufficient target groups for {target_name}."
        )

    for feature in scientific_features:

        x = pd.to_numeric(
            target_df[feature],
            errors="coerce"
        )

        x0 = x.loc[group_0_mask].dropna()
        x1 = x.loc[group_1_mask].dropna()

        if len(x0) < 2 or len(x1) < 2:
            continue

        mean0 = float(x0.mean())
        mean1 = float(x1.mean())

        median0 = float(x0.median())
        median1 = float(x1.median())

        std0 = float(x0.std(ddof=1))
        std1 = float(x1.std(ddof=1))

        # ---------------------------------------------------------------------
        # Difference: group 1 - group 0
        # ---------------------------------------------------------------------

        mean_difference = mean1 - mean0

        median_difference = median1 - median0

        # ---------------------------------------------------------------------
        # Welch's t-test
        # ---------------------------------------------------------------------

        t_stat, p_value = stats.ttest_ind(
            x1,
            x0,
            equal_var=False,
            nan_policy="omit"
        )

        # ---------------------------------------------------------------------
        # Cohen's d
        # ---------------------------------------------------------------------

        pooled_variance = (
            (
                (len(x1) - 1) * (std1 ** 2)
                +
                (len(x0) - 1) * (std0 ** 2)
            )
            /
            (
                len(x1)
                +
                len(x0)
                -
                2
            )
        )

        pooled_sd = np.sqrt(
            pooled_variance
        )

        if pooled_sd > 0:

            cohen_d = (
                mean_difference
                /
                pooled_sd
            )

        else:
            cohen_d = np.nan

        # ---------------------------------------------------------------------
        # Effect magnitude
        # ---------------------------------------------------------------------

        abs_d = (
            abs(cohen_d)
            if not pd.isna(cohen_d)
            else np.nan
        )

        if pd.isna(abs_d):
            effect_magnitude = "not_available"
        elif abs_d < 0.2:
            effect_magnitude = "negligible"
        elif abs_d < 0.5:
            effect_magnitude = "small"
        elif abs_d < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"

        direction = (
            "positive"
            if mean_difference > 0
            else
            "negative"
            if mean_difference < 0
            else
            "zero"
        )

        results.append(
            {
                "target": target_name,
                "feature": feature,
                "n_group_0": len(x0),
                "n_group_1": len(x1),
                "mean_group_0": mean0,
                "mean_group_1": mean1,
                "median_group_0": median0,
                "median_group_1": median1,
                "std_group_0": std0,
                "std_group_1": std1,
                "mean_difference": mean_difference,
                "median_difference": median_difference,
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "cohen_d": (
                    float(cohen_d)
                    if not pd.isna(cohen_d)
                    else np.nan
                ),
                "effect_magnitude": effect_magnitude,
                "direction": direction,
            }
        )

results_df = pd.DataFrame(results)

# =============================================================================
# FDR CORRECTION
# =============================================================================

print("\n" + "=" * 90)
print("FDR CORRECTION")
print("=" * 90)

results_df["p_fdr"] = np.nan
results_df["significant_fdr"] = False

for target_name in results_df["target"].unique():

    mask = (
        results_df["target"]
        == target_name
    )

    pvals = (
        results_df.loc[mask, "p_value"]
        .to_numpy(dtype=float)
    )

    m = len(pvals)

    if m == 0:
        continue

    order = np.argsort(pvals)

    ranked = pvals[order]

    adjusted = np.empty(m)

    for i in range(m):

        rank = i + 1

        adjusted[i] = (
            ranked[i]
            * m
            / rank
        )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1
    )

    restored = np.empty(m)

    for i, original_index in enumerate(order):

        restored[original_index] = (
            adjusted[i]
        )

    results_df.loc[
        mask,
        "p_fdr"
    ] = restored

    results_df.loc[
        mask,
        "significant_fdr"
    ] = restored < 0.05

# =============================================================================
# SUBJECT-LEVEL ROBUSTNESS
# =============================================================================

print("\n" + "=" * 90)
print("SUBJECT-LEVEL ROBUSTNESS")
print("=" * 90)

if "subject" not in df.columns:
    raise RuntimeError(
        "Subject column not found."
    )

subject_results = []

for target_name, target_col in TARGET_COLUMNS.items():

    for feature in scientific_features:

        per_subject = []

        for subject, subject_data in df.groupby(
            "subject"
        ):

            y_subject = pd.to_numeric(
                subject_data[target_col],
                errors="coerce"
            )

            x_subject = pd.to_numeric(
                subject_data[feature],
                errors="coerce"
            )

            valid = (
                y_subject.notna()
                &
                x_subject.notna()
            )

            y_s = y_subject.loc[valid]
            x_s = x_subject.loc[valid]

            group0 = x_s.loc[y_s == 0]
            group1 = x_s.loc[y_s == 1]

            if len(group0) < 2 or len(group1) < 2:
                continue

            diff = (
                group1.mean()
                -
                group0.mean()
            )

            per_subject.append(
                diff
            )

        if len(per_subject) == 0:
            continue

        per_subject = np.asarray(
            per_subject,
            dtype=float
        )

        positive_fraction = float(
            np.mean(per_subject > 0)
        )

        negative_fraction = float(
            np.mean(per_subject < 0)
        )

        consistency = max(
            positive_fraction,
            negative_fraction
        )

        # Subject-level one-sample test
        if (
            len(per_subject) >= 2
            and np.std(
                per_subject,
                ddof=1
            ) > 0
        ):

            subject_t, subject_p = (
                stats.ttest_1samp(
                    per_subject,
                    0
                )
            )

        else:

            subject_t = np.nan
            subject_p = np.nan

        subject_results.append(
            {
                "target": target_name,
                "feature": feature,
                "n_subjects": len(per_subject),
                "mean_subject_effect":
                    float(np.mean(per_subject)),
                "median_subject_effect":
                    float(np.median(per_subject)),
                "subject_std":
                    float(
                        np.std(
                            per_subject,
                            ddof=1
                        )
                    )
                    if len(per_subject) > 1
                    else np.nan,
                "positive_fraction":
                    positive_fraction,
                "negative_fraction":
                    negative_fraction,
                "directional_consistency":
                    consistency,
                "subject_t_stat":
                    float(subject_t)
                    if not pd.isna(subject_t)
                    else np.nan,
                "subject_p_value":
                    float(subject_p)
                    if not pd.isna(subject_p)
                    else np.nan,
            }
        )

subject_df = pd.DataFrame(
    subject_results
)

# =============================================================================
# SUBJECT FDR
# =============================================================================

if not subject_df.empty:

    subject_df["subject_p_fdr"] = np.nan

    for target_name in subject_df["target"].unique():

        mask = (
            subject_df["target"]
            == target_name
        )

        pvals = (
            subject_df.loc[
                mask,
                "subject_p_value"
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(pvals) == 0:
            continue

        order = np.argsort(pvals)

        ranked = pvals[order]

        m = len(ranked)

        adjusted = np.empty(m)

        for i in range(m):

            adjusted[i] = (
                ranked[i]
                * m
                / (i + 1)
            )

        adjusted = np.minimum.accumulate(
            adjusted[::-1]
        )[::-1]

        adjusted = np.clip(
            adjusted,
            0,
            1
        )

        # Assign only to valid rows
        valid_indices = (
            subject_df.loc[
                mask,
                "subject_p_value"
            ]
            .dropna()
            .index
        )

        for i, idx in enumerate(
            valid_indices[
                np.argsort(
                    subject_df.loc[
                        valid_indices,
                        "subject_p_value"
                    ].to_numpy()
                )
            ]
        ):
            subject_df.loc[
                idx,
                "subject_p_fdr"
            ] = adjusted[i]

# =============================================================================
# SORT
# =============================================================================

results_df = results_df.sort_values(
    by=[
        "target",
        "p_fdr",
        "cohen_d"
    ],
    ascending=[
        True,
        True,
        False
    ]
)

# =============================================================================
# FINAL QC
# =============================================================================

print("\n" + "=" * 90)
print("FINAL STATISTICAL QC")
print("=" * 90)

numeric_results = results_df.select_dtypes(
    include=[np.number]
)

nan_results = int(
    numeric_results.isna()
    .sum()
    .sum()
)

inf_results = int(
    np.isinf(
        numeric_results.to_numpy(
            dtype=float,
            copy=True
        )
    ).sum()
)

print(
    f"Features analyzed:       {len(scientific_features)}"
)

print(
    f"Statistical result rows:  {len(results_df):,}"
)

print(
    f"NaN numeric values:       {nan_results}"
)

print(
    f"Inf numeric values:       {inf_results}"
)

for target_name in results_df["target"].unique():

    target_results = results_df[
        results_df["target"]
        == target_name
    ]

    significant = int(
        target_results[
            "significant_fdr"
        ].sum()
    )

    print(
        f"{target_name.upper()} "
        f"FDR-significant features: "
        f"{significant}"
    )

# =============================================================================
# CRITICAL VALIDATION
# =============================================================================

if len(results_df) != (
    len(scientific_features)
    *
    len(TARGET_COLUMNS)
):

    raise RuntimeError(
        "Unexpected number of statistical result rows."
    )

if results_df["p_value"].notna().sum() == 0:

    raise RuntimeError(
        "All p-values are NaN. "
        "Statistical analysis failed."
    )

if results_df["cohen_d"].notna().sum() == 0:

    raise RuntimeError(
        "All Cohen's d values are NaN. "
        "Effect-size calculation failed."
    )

if inf_results > 0:

    raise RuntimeError(
        "Infinite statistical values detected."
    )

# =============================================================================
# SAVE
# =============================================================================

results_df.to_csv(
    OUTPUT_STATS,
    index=False
)

subject_df.to_csv(
    OUTPUT_SUBJECT,
    index=False
)

qc = pd.DataFrame(
    [
        {
            "input_rows":
                len(df),
            "scientific_features":
                len(scientific_features),
            "targets_analyzed":
                len(TARGET_COLUMNS),
            "statistical_rows":
                len(results_df),
            "p_values_valid":
                int(
                    results_df[
                        "p_value"
                    ].notna().sum()
                ),
            "cohen_d_valid":
                int(
                    results_df[
                        "cohen_d"
                    ].notna().sum()
                ),
            "fdr_significant_total":
                int(
                    results_df[
                        "significant_fdr"
                    ].sum()
                ),
            "subject_robustness_rows":
                len(subject_df),
            "nan_numeric_values":
                nan_results,
            "inf_numeric_values":
                inf_results,
            "status":
                "PASS",
        }
    ]
)

qc.to_csv(
    OUTPUT_QC,
    index=False
)

# =============================================================================
# DISPLAY TOP RESULTS
# =============================================================================

print("\n" + "=" * 90)
print("TOP PERTURBATION EFFECTS")
print("=" * 90)

display_cols = [
    "target",
    "feature",
    "mean_difference",
    "cohen_d",
    "p_value",
    "p_fdr",
    "effect_magnitude",
    "direction",
]

print(
    results_df[
        display_cols
    ]
    .head(20)
    .to_string(index=False)
)

# =============================================================================
# COMPLETE
# =============================================================================

print("\n" + "=" * 90)
print("PERTURBATION STATISTICAL ANALYSIS V2 COMPLETE")
print("=" * 90)

print(
    f"Scientific features: {len(scientific_features)}"
)

print(
    f"Rows analyzed:       {len(df):,}"
)

print(
    f"Statistical rows:    {len(results_df):,}"
)

print("\nSAVED:")

print(OUTPUT_STATS)
print(OUTPUT_SUBJECT)
print(OUTPUT_QC)

print("\n" + "=" * 90)
print("STATUS: PASS - PERTURBATION STATISTICS V2 CREATED")
print("=" * 90)