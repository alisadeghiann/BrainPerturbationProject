from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# PERTURBATION EFFECT + WHAT-IF SCIENTIFIC ANALYSIS
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
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EFFECT = (
    OUTPUT_DIR
    / "perturbation_effect_analysis.csv"
)

OUTPUT_SUBJECT = (
    OUTPUT_DIR
    / "perturbation_subject_effects.csv"
)

OUTPUT_QC = (
    OUTPUT_DIR
    / "perturbation_effect_qc.csv"
)


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("PERTURBATION EFFECT + WHAT-IF SCIENTIFIC ANALYSIS")
print("=" * 80)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

print(f"Input rows:       {len(df):,}")
print(f"Input columns:    {len(df.columns):,}")


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

required_columns = [
    "subject",
    "run",
    "trial",
    "epoch",
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


# =============================================================================
# SCIENTIFIC FEATURE IDENTIFICATION
# =============================================================================

TARGET_DERIVED = {
    "target_label",
    "target_remember",
    "target_correct",
    "is_correct",
    "is_remembered",
    "is_ignored",
    "behavior_label",
    "behavior_outcome",
    "feedback",
    "response_type",
    "probe_type",
    "probe_letter",
    "memory_cond",
    "remember_count",
    "ignore_count",
    "remember_letters",
    "ignore_letters",
    "complete_trial",
    "alignment_status",
    "event_source",
    "subject",
    "run",
    "trial",
    "epoch",
    "file",
}

numeric_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

scientific_features = [
    c for c in numeric_cols
    if c not in TARGET_DERIVED
]


print()
print("=" * 80)
print("SCIENTIFIC FEATURE INVENTORY")
print("=" * 80)

print(f"Scientific features: {len(scientific_features)}")


# =============================================================================
# BASIC QC
# =============================================================================

print()
print("=" * 80)
print("BASIC QC")
print("=" * 80)

numeric_data = (
    df[scientific_features]
    .apply(pd.to_numeric, errors="coerce")
)

nan_count = int(
    numeric_data.isna().sum().sum()
)

inf_count = int(
    np.isinf(
        numeric_data.to_numpy(dtype=float)
    ).sum()
)

duplicate_keys = int(
    df.duplicated(
        ["subject", "run", "epoch"]
    ).sum()
)

print(f"NaN values:       {nan_count}")
print(f"Inf values:       {inf_count}")
print(f"Duplicate keys:   {duplicate_keys}")


# =============================================================================
# CONDITION COUNTS
# =============================================================================

print()
print("=" * 80)
print("CONDITION COUNTS")
print("=" * 80)

print()
print("REMEMBER CONDITION")
print("-" * 80)

print(
    df["target_remember"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print("CORRECT CONDITION")
print("-" * 80)

print(
    df["target_correct"]
    .value_counts()
    .sort_index()
    .to_string()
)


# =============================================================================
# GLOBAL EFFECT ANALYSIS
# =============================================================================

print()
print("=" * 80)
print("GLOBAL PERTURBATION / CONDITION EFFECT")
print("=" * 80)

effect_rows = []


for feature in scientific_features:

    x = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    row = {
        "feature": feature
    }

    # -------------------------------------------------------------------------
    # REMEMBER EFFECT
    # -------------------------------------------------------------------------

    remember_0 = x[
        df["target_remember"] == 0
    ].dropna()

    remember_1 = x[
        df["target_remember"] == 1
    ].dropna()

    if len(remember_0) > 1 and len(remember_1) > 1:

        mean_0 = remember_0.mean()
        mean_1 = remember_1.mean()

        std_0 = remember_0.std(ddof=1)
        std_1 = remember_1.std(ddof=1)

        pooled_sd = np.sqrt(
            (
                std_0 ** 2
                +
                std_1 ** 2
            ) / 2
        )

        difference = mean_1 - mean_0

        if pooled_sd > 0:
            effect_size = difference / pooled_sd
        else:
            effect_size = np.nan

        row["remember_mean_0"] = mean_0
        row["remember_mean_1"] = mean_1
        row["remember_difference"] = difference
        row["remember_effect_size"] = effect_size


    # -------------------------------------------------------------------------
    # CORRECT EFFECT
    # -------------------------------------------------------------------------

    correct_0 = x[
        df["target_correct"] == 0
    ].dropna()

    correct_1 = x[
        df["target_correct"] == 1
    ].dropna()

    if len(correct_0) > 1 and len(correct_1) > 1:

        mean_0 = correct_0.mean()
        mean_1 = correct_1.mean()

        std_0 = correct_0.std(ddof=1)
        std_1 = correct_1.std(ddof=1)

        pooled_sd = np.sqrt(
            (
                std_0 ** 2
                +
                std_1 ** 2
            ) / 2
        )

        difference = mean_1 - mean_0

        if pooled_sd > 0:
            effect_size = difference / pooled_sd
        else:
            effect_size = np.nan

        row["correct_mean_0"] = mean_0
        row["correct_mean_1"] = mean_1
        row["correct_difference"] = difference
        row["correct_effect_size"] = effect_size


    effect_rows.append(row)


effects = pd.DataFrame(effect_rows)


# =============================================================================
# EFFECT MAGNITUDE
# =============================================================================

effects["abs_remember_effect"] = (
    effects["remember_effect_size"]
    .abs()
)

effects["abs_correct_effect"] = (
    effects["correct_effect_size"]
    .abs()
)


def classify_effect(x):

    if pd.isna(x):
        return "NA"

    x = abs(x)

    if x < 0.20:
        return "negligible"

    if x < 0.50:
        return "small"

    if x < 0.80:
        return "medium"

    return "large"


effects["remember_effect_magnitude"] = (
    effects["remember_effect_size"]
    .apply(classify_effect)
)

effects["correct_effect_magnitude"] = (
    effects["correct_effect_size"]
    .apply(classify_effect)
)


# =============================================================================
# RANK FEATURES
# =============================================================================

effects = effects.sort_values(
    "abs_remember_effect",
    ascending=False
)


# =============================================================================
# SAVE GLOBAL EFFECTS
# =============================================================================

effects.to_csv(
    OUTPUT_EFFECT,
    index=False
)


# =============================================================================
# SUBJECT-LEVEL EFFECT ANALYSIS
# =============================================================================

print()
print("=" * 80)
print("SUBJECT-LEVEL PERTURBATION EFFECT")
print("=" * 80)

subject_rows = []


for subject, sdf in df.groupby("subject"):

    for feature in scientific_features:

        x = pd.to_numeric(
            sdf[feature],
            errors="coerce"
        )

        # -------------------------------------------------------------
        # REMEMBER
        # -------------------------------------------------------------

        r0 = x[
            sdf["target_remember"] == 0
        ].dropna()

        r1 = x[
            sdf["target_remember"] == 1
        ].dropna()

        if len(r0) > 1 and len(r1) > 1:

            m0 = r0.mean()
            m1 = r1.mean()

            pooled = np.sqrt(
                (
                    r0.var(ddof=1)
                    +
                    r1.var(ddof=1)
                ) / 2
            )

            if pooled > 0:
                remember_effect = (
                    (m1 - m0) / pooled
                )
            else:
                remember_effect = np.nan

        else:
            remember_effect = np.nan


        # -------------------------------------------------------------
        # CORRECT
        # -------------------------------------------------------------

        c0 = x[
            sdf["target_correct"] == 0
        ].dropna()

        c1 = x[
            sdf["target_correct"] == 1
        ].dropna()

        if len(c0) > 1 and len(c1) > 1:

            m0 = c0.mean()
            m1 = c1.mean()

            pooled = np.sqrt(
                (
                    c0.var(ddof=1)
                    +
                    c1.var(ddof=1)
                ) / 2
            )

            if pooled > 0:
                correct_effect = (
                    (m1 - m0) / pooled
                )
            else:
                correct_effect = np.nan

        else:
            correct_effect = np.nan


        subject_rows.append({
            "subject": subject,
            "feature": feature,
            "remember_effect_size": remember_effect,
            "correct_effect_size": correct_effect,
            "remember_abs_effect": (
                abs(remember_effect)
                if not pd.isna(remember_effect)
                else np.nan
            ),
            "correct_abs_effect": (
                abs(correct_effect)
                if not pd.isna(correct_effect)
                else np.nan
            ),
        })


subject_effects = pd.DataFrame(
    subject_rows
)

subject_effects.to_csv(
    OUTPUT_SUBJECT,
    index=False
)


# =============================================================================
# SUBJECT STABILITY
# =============================================================================

print()
print("=" * 80)
print("SUBJECT-LEVEL EFFECT STABILITY")
print("=" * 80)

stability_rows = []


for feature in scientific_features:

    sdf = subject_effects[
        subject_effects["feature"] == feature
    ]

    remember_effects = (
        sdf["remember_effect_size"]
        .dropna()
    )

    correct_effects = (
        sdf["correct_effect_size"]
        .dropna()
    )

    row = {
        "feature": feature,
        "subjects_with_remember_effect": len(
            remember_effects
        ),
        "subjects_with_correct_effect": len(
            correct_effects
        ),
    }


    if len(remember_effects) > 0:

        row["remember_subject_mean"] = (
            remember_effects.mean()
        )

        row["remember_subject_median"] = (
            remember_effects.median()
        )

        row["remember_subject_std"] = (
            remember_effects.std(ddof=1)
            if len(remember_effects) > 1
            else np.nan
        )

        row["remember_positive_subject_fraction"] = (
            (remember_effects > 0).mean()
        )


    if len(correct_effects) > 0:

        row["correct_subject_mean"] = (
            correct_effects.mean()
        )

        row["correct_subject_median"] = (
            correct_effects.median()
        )

        row["correct_subject_std"] = (
            correct_effects.std(ddof=1)
            if len(correct_effects) > 1
            else np.nan
        )

        row["correct_positive_subject_fraction"] = (
            (correct_effects > 0).mean()
        )


    stability_rows.append(row)


stability = pd.DataFrame(
    stability_rows
)


# =============================================================================
# ADD STABILITY TO GLOBAL RESULTS
# =============================================================================

effects = effects.merge(
    stability,
    on="feature",
    how="left"
)

effects = effects.sort_values(
    "abs_remember_effect",
    ascending=False
)

effects.to_csv(
    OUTPUT_EFFECT,
    index=False
)


# =============================================================================
# TOP FEATURES
# =============================================================================

print()
print("=" * 80)
print("TOP FEATURES FOR REMEMBER CONDITION")
print("=" * 80)

top_remember_cols = [
    "feature",
    "remember_difference",
    "remember_effect_size",
    "remember_effect_magnitude",
    "remember_subject_mean",
    "remember_subject_std",
    "remember_positive_subject_fraction",
]

top_remember_cols = [
    c for c in top_remember_cols
    if c in effects.columns
]

print(
    effects[
        top_remember_cols
    ]
    .head(20)
    .to_string(index=False)
)


print()
print("=" * 80)
print("TOP FEATURES FOR CORRECT CONDITION")
print("=" * 80)

top_correct = effects.sort_values(
    "abs_correct_effect",
    ascending=False
)

top_correct_cols = [
    "feature",
    "correct_difference",
    "correct_effect_size",
    "correct_effect_magnitude",
    "correct_subject_mean",
    "correct_subject_std",
    "correct_positive_subject_fraction",
]

top_correct_cols = [
    c for c in top_correct_cols
    if c in top_correct.columns
]

print(
    top_correct[
        top_correct_cols
    ]
    .head(20)
    .to_string(index=False)
)


# =============================================================================
# QC
# =============================================================================

qc = pd.DataFrame({
    "metric": [
        "input_rows",
        "input_columns",
        "scientific_features",
        "subjects",
        "runs",
        "NaN_values",
        "Inf_values",
        "duplicate_subject_run_epoch_keys",
        "remember_rows_0",
        "remember_rows_1",
        "correct_rows_0",
        "correct_rows_1",
        "features_with_remember_effect",
        "features_with_correct_effect",
    ],
    "value": [
        len(df),
        len(df.columns),
        len(scientific_features),
        df["subject"].nunique(),
        df["run"].nunique(),
        nan_count,
        inf_count,
        duplicate_keys,
        int((df["target_remember"] == 0).sum()),
        int((df["target_remember"] == 1).sum()),
        int((df["target_correct"] == 0).sum()),
        int((df["target_correct"] == 1).sum()),
        int(
            effects["remember_effect_size"]
            .notna()
            .sum()
        ),
        int(
            effects["correct_effect_size"]
            .notna()
            .sum()
        ),
    ]
})

qc.to_csv(
    OUTPUT_QC,
    index=False
)


# =============================================================================
# FINAL VALIDATION
# =============================================================================

print()
print("=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

print(f"Rows:                    {len(df):,}")
print(f"Scientific features:     {len(scientific_features)}")
print(f"Subjects:                {df['subject'].nunique()}")
print(f"NaN values:              {nan_count}")
print(f"Inf values:              {inf_count}")
print(f"Duplicate keys:          {duplicate_keys}")

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_EFFECT)
print(OUTPUT_SUBJECT)
print(OUTPUT_QC)

print()
print("=" * 80)

if (
    nan_count == 0
    and inf_count == 0
    and duplicate_keys == 0
    and len(scientific_features) > 0
):
    print(
        "STATUS: PASS - PERTURBATION EFFECT ANALYSIS COMPLETED"
    )
else:
    print(
        "STATUS: REVIEW REQUIRED - QC ISSUE DETECTED"
    )

print("=" * 80)