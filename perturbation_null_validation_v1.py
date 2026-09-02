from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# PERTURBATION / WHAT-IF NULL VALIDATION V2
# SUBJECT-LEVEL SIGN-FLIP EMPIRICAL NULL
# =============================================================================

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
    / "null_validation_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_RESULTS = OUTPUT_DIR / "perturbation_null_validation_v1.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "perturbation_null_validation_summary_v1.csv"
OUTPUT_QC = OUTPUT_DIR / "perturbation_null_validation_qc_v1.csv"

N_PERMUTATIONS = 2000
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# =============================================================================
# LOAD
# =============================================================================

print("=" * 90)
print("PERTURBATION / WHAT-IF NULL VALIDATION V2")
print("SUBJECT-LEVEL SIGN-FLIP EMPIRICAL NULL")
print("=" * 90)

if not INPUT.exists():
    raise FileNotFoundError(f"Input not found:\n{INPUT}")

df = pd.read_csv(INPUT)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print("Columns:")
print(df.columns.tolist())
print()


# =============================================================================
# IDENTIFY CORE COLUMNS
# =============================================================================

subject_col = None
feature_col = None

for c in df.columns:
    cl = c.lower()

    if subject_col is None and cl == "subject":
        subject_col = c

    if feature_col is None and cl in {
        "feature",
        "feature_std",
        "feature_name",
    }:
        feature_col = c

if subject_col is None:
    raise ValueError("Could not identify subject column.")

if feature_col is None:
    raise ValueError("Could not identify feature column.")


# =============================================================================
# IDENTIFY EFFECT COLUMNS
# =============================================================================

effect_candidates = []

for c in df.columns:
    cl = c.lower()

    if (
        ("effect" in cl or "cohen" in cl or "d" == cl)
        and (
            "remember" in cl
            or "correct" in cl
        )
    ):
        effect_candidates.append(c)

print("Detected effect columns:")
print(effect_candidates)
print()

if len(effect_candidates) == 0:
    raise ValueError(
        "No remember/correct effect columns were detected."
    )


remember_col = None
correct_col = None

for c in effect_candidates:
    cl = c.lower()

    if "remember" in cl and remember_col is None:
        remember_col = c

    if "correct" in cl and correct_col is None:
        correct_col = c


if remember_col is None:
    raise ValueError(
        "Remember effect column could not be identified."
    )

if correct_col is None:
    raise ValueError(
        "Correct effect column could not be identified."
    )

print(f"Remember effect column: {remember_col}")
print(f"Correct effect column:  {correct_col}")
print()


# =============================================================================
# BUILD LONG FORMAT
# =============================================================================

base_cols = [
    subject_col,
    feature_col,
]

remember = df[
    base_cols + [remember_col]
].copy()

remember["target"] = "remember"
remember["effect_size"] = pd.to_numeric(
    remember[remember_col],
    errors="coerce"
)

remember = remember[
    [
        subject_col,
        feature_col,
        "target",
        "effect_size",
    ]
]

correct = df[
    base_cols + [correct_col]
].copy()

correct["target"] = "correct"
correct["effect_size"] = pd.to_numeric(
    correct[correct_col],
    errors="coerce"
)

correct = correct[
    [
        subject_col,
        feature_col,
        "target",
        "effect_size",
    ]
]

work = pd.concat(
    [remember, correct],
    ignore_index=True
)

work = work.rename(
    columns={
        subject_col: "subject",
        feature_col: "feature",
    }
)

work["subject"] = work["subject"].astype(str)
work["feature"] = work["feature"].astype(str)

work = work.replace(
    [np.inf, -np.inf],
    np.nan
)

work = work.dropna(
    subset=[
        "subject",
        "feature",
        "target",
        "effect_size",
    ]
).copy()


# =============================================================================
# DUPLICATE CHECK
# =============================================================================

duplicates = int(
    work.duplicated(
        subset=["subject", "feature", "target"]
    ).sum()
)

if duplicates != 0:
    raise ValueError(
        f"Duplicate subject-feature-target rows: {duplicates}"
    )


# =============================================================================
# NULL VALIDATION
# =============================================================================

print("=" * 90)
print("BUILDING SUBJECT-LEVEL SIGN-FLIP NULL")
print("=" * 90)

print(f"Permutations: {N_PERMUTATIONS}")
print(f"Random seed: {RANDOM_SEED}")
print()

results = []

for target in ["correct", "remember"]:

    target_df = work[
        work["target"] == target
    ]

    for feature, feature_df in target_df.groupby("feature"):

        values = feature_df[
            "effect_size"
        ].to_numpy(dtype=float)

        values = values[np.isfinite(values)]

        n_subjects = len(values)

        if n_subjects < 3:
            continue

        observed_mean = float(
            np.mean(values)
        )

        observed_abs_mean = abs(
            observed_mean
        )

        observed_median = float(
            np.median(values)
        )

        signs = rng.choice(
            [-1.0, 1.0],
            size=(N_PERMUTATIONS, n_subjects)
        )

        permuted = (
            signs * values[None, :]
        )

        null_mean = np.mean(
            permuted,
            axis=1
        )

        null_abs_mean = np.abs(
            null_mean
        )

        empirical_p = (
            np.sum(
                null_abs_mean >= observed_abs_mean
            ) + 1
        ) / (
            N_PERMUTATIONS + 1
        )

        percentile = float(
            np.mean(
                null_abs_mean < observed_abs_mean
            )
        )

        results.append(
            {
                "target": target,
                "feature": feature,
                "n_subjects": n_subjects,
                "observed_mean_effect": observed_mean,
                "observed_abs_mean_effect": observed_abs_mean,
                "observed_median_effect": observed_median,
                "null_mean_abs_median": float(
                    np.median(null_abs_mean)
                ),
                "null_mean_abs_95th_percentile": float(
                    np.percentile(
                        null_abs_mean,
                        95
                    )
                ),
                "null_empirical_p": empirical_p,
                "null_percentile": percentile,
                "null_validation_class": (
                    "above_null_distribution"
                    if empirical_p < 0.05
                    else "within_null_distribution"
                ),
            }
        )


results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError(
        "No valid feature-target combinations were produced."
    )


# =============================================================================
# SUMMARY
# =============================================================================

summary_rows = []

for target in ["correct", "remember"]:

    sub = results_df[
        results_df["target"] == target
    ]

    summary_rows.append(
        {
            "target": target,
            "features_analyzed": len(sub),
            "features_above_null": int(
                (
                    sub["null_empirical_p"] < 0.05
                ).sum()
            ),
            "median_abs_mean_effect": float(
                sub[
                    "observed_abs_mean_effect"
                ].median()
            ),
            "max_abs_mean_effect": float(
                sub[
                    "observed_abs_mean_effect"
                ].max()
            ),
            "median_null_empirical_p": float(
                sub[
                    "null_empirical_p"
                ].median()
            ),
        }
    )

summary_df = pd.DataFrame(
    summary_rows
)


# =============================================================================
# QC
# =============================================================================

numeric_cols = results_df.select_dtypes(
    include=[np.number]
).columns

nan_numeric = int(
    results_df[numeric_cols]
    .isna()
    .sum()
    .sum()
)

inf_numeric = int(
    np.isinf(
        results_df[numeric_cols]
        .to_numpy(dtype=float)
    ).sum()
)

duplicate_target_feature = int(
    results_df.duplicated(
        subset=["target", "feature"]
    ).sum()
)

qc = pd.DataFrame(
    [
        {
            "input_file": str(INPUT),
            "rows": len(results_df),
            "targets": results_df["target"].nunique(),
            "features": results_df["feature"].nunique(),
            "permutations": N_PERMUTATIONS,
            "random_seed": RANDOM_SEED,
            "null_significant_rows": int(
                (
                    results_df[
                        "null_empirical_p"
                    ] < 0.05
                ).sum()
            ),
            "NaN_numeric_cells": nan_numeric,
            "Inf_numeric_cells": inf_numeric,
            "duplicate_target_feature": duplicate_target_feature,
        }
    ]
)

print("=" * 90)
print("NULL VALIDATION QC")
print("=" * 90)

print(f"Rows: {len(results_df)}")
print(f"Targets: {results_df['target'].nunique()}")
print(f"Features: {results_df['feature'].nunique()}")
print(
    "Null-significant rows:",
    int(
        (
            results_df["null_empirical_p"] < 0.05
        ).sum()
    )
)
print("NaN numeric cells:", nan_numeric)
print("Inf numeric cells:", inf_numeric)
print(
    "Duplicate target-feature:",
    duplicate_target_feature
)

if nan_numeric != 0:
    raise ValueError("NaN numeric values detected.")

if inf_numeric != 0:
    raise ValueError("Inf numeric values detected.")

if duplicate_target_feature != 0:
    raise ValueError(
        "Duplicate target-feature rows detected."
    )


# =============================================================================
# SAVE
# =============================================================================

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

qc.to_csv(
    OUTPUT_QC,
    index=False
)

print()
print("=" * 90)
print("PERTURBATION / WHAT-IF NULL VALIDATION V2 COMPLETE")
print("=" * 90)

print("Saved:")
print(OUTPUT_RESULTS)
print(OUTPUT_SUMMARY)
print(OUTPUT_QC)

print()
print("STATUS: PASS - SUBJECT-LEVEL EMPIRICAL NULL VALIDATION CREATED")