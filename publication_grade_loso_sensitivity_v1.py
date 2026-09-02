from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# PUBLICATION-GRADE LEAVE-ONE-SUBJECT-OUT SENSITIVITY V1
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "perturbation_subject_effects.csv"
)

OUTDIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "publication_grade_loso_v1"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTDIR / "publication_grade_loso_sensitivity_v1.csv"
SUMMARY = OUTDIR / "publication_grade_loso_summary_v1.csv"
QC = OUTDIR / "publication_grade_loso_qc_v1.csv"

print("=" * 90)
print("PUBLICATION-GRADE LEAVE-ONE-SUBJECT-OUT SENSITIVITY V1")
print("=" * 90)

# =============================================================================
# LOAD
# =============================================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Subject-level perturbation file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

required = {
    "subject",
    "feature",
    "remember_effect_size",
    "correct_effect_size",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

for col in [
    "remember_effect_size",
    "correct_effect_size",
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# =============================================================================
# BASIC QC
# =============================================================================

subjects = sorted(
    df["subject"]
    .dropna()
    .unique()
)

features = sorted(
    df["feature"]
    .dropna()
    .unique()
)

print("\nDATASET SUMMARY")
print("-" * 90)
print(f"Rows:             {len(df)}")
print(f"Subjects:         {len(subjects)}")
print(f"Features:         {len(features)}")

duplicates = int(
    df.duplicated(
        subset=["subject", "feature"]
    ).sum()
)

print(f"Duplicate rows:   {duplicates}")

if duplicates > 0:
    raise ValueError(
        "Duplicate subject-feature rows detected."
    )

# =============================================================================
# FULL-SAMPLE EFFECTS
# =============================================================================

full_results = []

for target, effect_col in [
    ("remember", "remember_effect_size"),
    ("correct", "correct_effect_size"),
]:

    for feature in features:

        x = df.loc[
            df["feature"] == feature,
            ["subject", effect_col]
        ].dropna()

        if len(x) == 0:
            continue

        values = x[effect_col].to_numpy(dtype=float)

        full_mean = float(np.mean(values))
        full_median = float(np.median(values))

        full_sign = (
            np.sign(full_mean)
            if full_mean != 0
            else 0
        )

        full_results.append({
            "target": target,
            "feature": feature,
            "full_n_subjects": len(values),
            "full_mean_effect": full_mean,
            "full_median_effect": full_median,
            "full_direction": full_sign,
        })

full = pd.DataFrame(full_results)

# =============================================================================
# LEAVE-ONE-SUBJECT-OUT
# =============================================================================

rows = []

print("\nRUNNING LEAVE-ONE-SUBJECT-OUT ANALYSIS")
print("-" * 90)

for subject_out in subjects:

    print(f"Leaving out: {subject_out}")

    train = df[df["subject"] != subject_out]

    for target, effect_col in [
        ("remember", "remember_effect_size"),
        ("correct", "correct_effect_size"),
    ]:

        for feature in features:

            x = train.loc[
                train["feature"] == feature,
                effect_col
            ].dropna()

            if len(x) == 0:
                continue

            values = x.to_numpy(dtype=float)

            mean_effect = float(
                np.mean(values)
            )

            median_effect = float(
                np.median(values)
            )

            sign = (
                np.sign(mean_effect)
                if mean_effect != 0
                else 0
            )

            rows.append({
                "subject_left_out": subject_out,
                "target": target,
                "feature": feature,
                "n_subjects": len(values),
                "mean_effect": mean_effect,
                "median_effect": median_effect,
                "direction": sign,
            })

loso = pd.DataFrame(rows)

# =============================================================================
# MERGE FULL RESULTS
# =============================================================================

result = loso.merge(
    full,
    on=["target", "feature"],
    how="left",
    validate="many_to_one"
)

# =============================================================================
# SENSITIVITY METRICS
# =============================================================================

result["direction_agrees_with_full"] = (
    result["direction"]
    == result["full_direction"]
)

result["absolute_mean_change"] = (
    result["mean_effect"]
    - result["full_mean_effect"]
).abs()

result["relative_mean_change"] = np.where(
    result["full_mean_effect"].abs() > 1e-12,
    result["absolute_mean_change"]
    / result["full_mean_effect"].abs(),
    np.nan
)

# =============================================================================
# FEATURE-LEVEL SUMMARY
# =============================================================================

summary_rows = []

for (target, feature), group in result.groupby(
    ["target", "feature"]
):

    direction_stability = float(
        group["direction_agrees_with_full"].mean()
    )

    mean_change = float(
        group["absolute_mean_change"].mean()
    )

    max_change = float(
        group["absolute_mean_change"].max()
    )

    relative_change = float(
        group["relative_mean_change"]
        .replace([np.inf, -np.inf], np.nan)
        .mean()
    ) if group["relative_mean_change"].notna().any() else np.nan

    full_mean = float(
        group["full_mean_effect"].iloc[0]
    )

    full_median = float(
        group["full_median_effect"].iloc[0]
    )

    if direction_stability >= 0.95:
        stability_class = "high_LOSO_stability"
    elif direction_stability >= 0.80:
        stability_class = "moderate_LOSO_stability"
    elif direction_stability >= 0.60:
        stability_class = "weak_LOSO_stability"
    else:
        stability_class = "direction_sensitive"

    summary_rows.append({
        "target": target,
        "feature": feature,
        "full_mean_effect": full_mean,
        "full_median_effect": full_median,
        "direction_stability": direction_stability,
        "mean_absolute_change": mean_change,
        "max_absolute_change": max_change,
        "mean_relative_change": relative_change,
        "LOSO_stability_class": stability_class,
    })

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    by=[
        "direction_stability",
        "full_mean_effect"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)

# =============================================================================
# QC
# =============================================================================

numeric_cols = result.select_dtypes(
    include=[np.number]
).columns

nan_numeric = int(
    result[numeric_cols]
    .isna()
    .sum()
    .sum()
)

inf_numeric = int(
    np.isinf(
        result[numeric_cols]
        .to_numpy(dtype=float)
    ).sum()
)

duplicate_loso = int(
    result.duplicated(
        subset=[
            "subject_left_out",
            "target",
            "feature"
        ]
    ).sum()
)

high_stability = int(
    (
        summary["LOSO_stability_class"]
        == "high_LOSO_stability"
    ).sum()
)

moderate_stability = int(
    (
        summary["LOSO_stability_class"]
        == "moderate_LOSO_stability"
    ).sum()
)

direction_sensitive = int(
    (
        summary["LOSO_stability_class"]
        == "direction_sensitive"
    ).sum()
)

qc = pd.DataFrame([{
    "subjects": len(subjects),
    "features": len(features),
    "targets": 2,
    "LOSO_iterations": len(subjects),
    "result_rows": len(result),
    "summary_rows": len(summary),
    "high_LOSO_stability": high_stability,
    "moderate_LOSO_stability": moderate_stability,
    "direction_sensitive_features": direction_sensitive,
    "NaN_numeric_cells": nan_numeric,
    "Inf_numeric_cells": inf_numeric,
    "duplicate_LOSO_rows": duplicate_loso,
}])

# =============================================================================
# PRINT
# =============================================================================

print("\n" + "=" * 90)
print("FINAL LOSO SENSITIVITY QC")
print("=" * 90)

print(f"Subjects:                    {len(subjects)}")
print(f"Features:                    {len(features)}")
print(f"Targets:                     2")
print(f"LOSO iterations:             {len(subjects)}")
print(f"Result rows:                 {len(result)}")
print(f"High LOSO stability:         {high_stability}")
print(f"Moderate LOSO stability:     {moderate_stability}")
print(f"Direction-sensitive:         {direction_sensitive}")
print(f"NaN numeric cells:           {nan_numeric}")
print(f"Inf numeric cells:           {inf_numeric}")
print(f"Duplicate LOSO rows:         {duplicate_loso}")

print("\n" + "=" * 90)
print("TOP LOSO-STABLE FEATURES")
print("=" * 90)

print(
    summary[
        [
            "target",
            "feature",
            "full_mean_effect",
            "direction_stability",
            "mean_absolute_change",
            "max_absolute_change",
            "LOSO_stability_class",
        ]
    ]
    .head(40)
    .to_string(index=False)
)

# =============================================================================
# SAVE
# =============================================================================

result.to_csv(
    OUTPUT,
    index=False
)

summary.to_csv(
    SUMMARY,
    index=False
)

qc.to_csv(
    QC,
    index=False
)

print("\n" + "=" * 90)
print("SAVED")
print("=" * 90)

print(OUTPUT)
print(SUMMARY)
print(QC)

print("\n" + "=" * 90)
print("PUBLICATION-GRADE LOSO SENSITIVITY V1 COMPLETE")
print("=" * 90)

if (
    nan_numeric == 0
    and inf_numeric == 0
    and duplicate_loso == 0
):
    print(
        "STATUS: PASS - LOSO SENSITIVITY VALIDATION CREATED"
    )
else:
    print("STATUS: REVIEW_REQUIRED")