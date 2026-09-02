from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# PERTURBATION / WHAT-IF SCIENTIFIC FEATURE ANALYSIS
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

OUTPUT_SUMMARY = OUTPUT_DIR / "perturbation_feature_summary.csv"
OUTPUT_QC = OUTPUT_DIR / "perturbation_feature_analysis_qc.csv"


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("PERTURBATION / WHAT-IF SCIENTIFIC FEATURE ANALYSIS")
print("=" * 80)

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT}")

df = pd.read_csv(INPUT)

print(f"Input rows:       {len(df):,}")
print(f"Input columns:    {len(df.columns):,}")


# =============================================================================
# IDENTIFY SCIENTIFIC FEATURES
# =============================================================================

TARGET_COLS = {
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


numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

scientific_features = [
    c for c in numeric_cols
    if c not in TARGET_COLS
]

print()
print("=" * 80)
print("SCIENTIFIC FEATURE INVENTORY")
print("=" * 80)

print(f"Scientific features: {len(scientific_features)}")
print()

print(scientific_features)


# =============================================================================
# BASIC QC
# =============================================================================

print()
print("=" * 80)
print("FEATURE QC")
print("=" * 80)

nan_count = int(df[scientific_features].isna().sum().sum())

inf_count = int(
    np.isinf(
        df[scientific_features]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    ).sum()
)

duplicate_keys = 0

key_cols = ["subject", "run", "epoch"]

if all(c in df.columns for c in key_cols):
    duplicate_keys = int(df.duplicated(key_cols).sum())

print(f"NaN values:       {nan_count}")
print(f"Inf values:       {inf_count}")
print(f"Duplicate keys:   {duplicate_keys}")


# =============================================================================
# SUBJECT-LEVEL FEATURE SUMMARY
# =============================================================================

print()
print("=" * 80)
print("SUBJECT-LEVEL SCIENTIFIC FEATURE SUMMARY")
print("=" * 80)

subject_summary = (
    df.groupby("subject")[scientific_features]
    .agg(["mean", "std"])
)

subject_summary_path = OUTPUT_DIR / "subject_feature_summary.csv"
subject_summary.to_csv(subject_summary_path)

print(f"Subjects: {df['subject'].nunique()}")
print(f"Saved: {subject_summary_path}")


# =============================================================================
# TARGET-CONDITION FEATURE DIFFERENCES
# =============================================================================

summary_rows = []

for feature in scientific_features:

    row = {
        "feature": feature
    }

    # -------------------------------------------------------------------------
    # REMEMBER CONDITION
    # -------------------------------------------------------------------------

    if "target_remember" in df.columns:

        remember_0 = pd.to_numeric(
            df.loc[df["target_remember"] == 0, feature],
            errors="coerce"
        ).dropna()

        remember_1 = pd.to_numeric(
            df.loc[df["target_remember"] == 1, feature],
            errors="coerce"
        ).dropna()

        if len(remember_0) > 0 and len(remember_1) > 0:

            mean_0 = remember_0.mean()
            mean_1 = remember_1.mean()

            pooled_std = np.sqrt(
                (
                    remember_0.var(ddof=1)
                    +
                    remember_1.var(ddof=1)
                ) / 2
            )

            effect = (
                (mean_1 - mean_0) / pooled_std
                if pooled_std > 0
                else np.nan
            )

            row["remember_mean_0"] = mean_0
            row["remember_mean_1"] = mean_1
            row["remember_difference"] = mean_1 - mean_0
            row["remember_effect_size"] = effect

    # -------------------------------------------------------------------------
    # CORRECT CONDITION
    # -------------------------------------------------------------------------

    if "target_correct" in df.columns:

        correct_0 = pd.to_numeric(
            df.loc[df["target_correct"] == 0, feature],
            errors="coerce"
        ).dropna()

        correct_1 = pd.to_numeric(
            df.loc[df["target_correct"] == 1, feature],
            errors="coerce"
        ).dropna()

        if len(correct_0) > 0 and len(correct_1) > 0:

            mean_0 = correct_0.mean()
            mean_1 = correct_1.mean()

            pooled_std = np.sqrt(
                (
                    correct_0.var(ddof=1)
                    +
                    correct_1.var(ddof=1)
                ) / 2
            )

            effect = (
                (mean_1 - mean_0) / pooled_std
                if pooled_std > 0
                else np.nan
            )

            row["correct_mean_0"] = mean_0
            row["correct_mean_1"] = mean_1
            row["correct_difference"] = mean_1 - mean_0
            row["correct_effect_size"] = effect

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)


# =============================================================================
# SORT BY EFFECT SIZE
# =============================================================================

if "remember_effect_size" in summary.columns:

    summary["abs_remember_effect"] = (
        summary["remember_effect_size"].abs()
    )

    summary = summary.sort_values(
        "abs_remember_effect",
        ascending=False
    )


# =============================================================================
# SAVE SUMMARY
# =============================================================================

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# =============================================================================
# QC REPORT
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
    ],
    "value": [
        len(df),
        len(df.columns),
        len(scientific_features),
        df["subject"].nunique() if "subject" in df.columns else np.nan,
        df["run"].nunique() if "run" in df.columns else np.nan,
        nan_count,
        inf_count,
        duplicate_keys,
    ]
})

qc.to_csv(
    OUTPUT_QC,
    index=False
)


# =============================================================================
# DISPLAY TOP FEATURES
# =============================================================================

print()
print("=" * 80)
print("TOP FEATURES BY REMEMBER EFFECT SIZE")
print("=" * 80)

display_cols = [
    "feature",
    "remember_mean_0",
    "remember_mean_1",
    "remember_difference",
    "remember_effect_size",
]

display_cols = [
    c for c in display_cols
    if c in summary.columns
]

print(
    summary[display_cols]
    .head(20)
    .to_string(index=False)
)


# =============================================================================
# FINAL STATUS
# =============================================================================

print()
print("=" * 80)
print("PERTURBATION / WHAT-IF FEATURE ANALYSIS COMPLETE")
print("=" * 80)

print(f"Scientific features: {len(scientific_features)}")
print(f"Rows analyzed:       {len(df):,}")
print(f"Subjects:            {df['subject'].nunique()}")

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_SUMMARY)
print(OUTPUT_QC)
print(subject_summary_path)

print()
print("=" * 80)
print("STATUS: PASS - SCIENTIFIC PERTURBATION ANALYSIS CREATED")
print("=" * 80)