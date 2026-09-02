from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# =============================================================================
# PERTURBATION STATISTICAL ANALYSIS V1
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "perturbation_effect_analysis.csv"
)

SUBJECT_INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "perturbation_subject_effects.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "statistical"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_STATS = OUTPUT_DIR / "perturbation_statistical_results.csv"
OUTPUT_QC = OUTPUT_DIR / "perturbation_statistical_qc.csv"
OUTPUT_SUBJECT = OUTPUT_DIR / "perturbation_subject_robustness.csv"

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 90)
print("PERTURBATION STATISTICAL ANALYSIS V1")
print("=" * 90)

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT}")

if not SUBJECT_INPUT.exists():
    raise FileNotFoundError(f"Subject input not found:\n{SUBJECT_INPUT}")

df = pd.read_csv(INPUT)
subject_df = pd.read_csv(SUBJECT_INPUT)

print(f"Input rows:       {len(df):,}")
print(f"Input columns:    {len(df.columns)}")
print(f"Subject rows:     {len(subject_df):,}")

# =============================================================================
# BASIC VALIDATION
# =============================================================================

print("\n" + "=" * 90)
print("BASIC VALIDATION")
print("=" * 90)

if "feature" not in df.columns:
    raise RuntimeError("Column 'feature' not found.")

print("Columns:")
print(df.columns.tolist())

# =============================================================================
# IDENTIFY NUMERIC EFFECT COLUMNS
# =============================================================================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\nNumeric columns:")
print(numeric_cols)

# Exclude obvious metadata columns
exclude = {
    "subject",
    "run",
    "trial",
    "epoch",
    "n_epochs",
}

effect_cols = [
    c for c in numeric_cols
    if c not in exclude
]

print(f"\nCandidate numeric effect columns: {len(effect_cols)}")

# =============================================================================
# STATISTICAL TESTS
# =============================================================================

print("\n" + "=" * 90)
print("FEATURE-LEVEL STATISTICAL TESTS")
print("=" * 90)

results = []

for feature_name, group in df.groupby("feature", dropna=False):

    row = {
        "feature": feature_name,
        "n_rows": len(group),
        "n_subjects": (
            group["subject"].nunique()
            if "subject" in group.columns
            else np.nan
        ),
    }

    # -------------------------------------------------------------------------
    # Find likely effect column
    # -------------------------------------------------------------------------

    preferred_effects = [
        "effect",
        "difference",
        "delta",
        "mean_difference",
        "effect_size",
        "perturbation_effect",
    ]

    effect_col = None

    for candidate in preferred_effects:
        if candidate in group.columns:
            effect_col = candidate
            break

    # If no standard effect column exists, use first suitable numeric column
    if effect_col is None:
        candidates = [
            c for c in effect_cols
            if c in group.columns
        ]

        if len(candidates) > 0:
            effect_col = candidates[0]

    row["effect_column"] = effect_col

    if effect_col is None:
        row["mean_effect"] = np.nan
        row["median_effect"] = np.nan
        row["std_effect"] = np.nan
        row["t_stat"] = np.nan
        row["p_value"] = np.nan
        row["cohen_d"] = np.nan
        row["significant_fdr"] = False

        results.append(row)
        continue

    values = pd.to_numeric(
        group[effect_col],
        errors="coerce"
    ).dropna()

    row["n_valid"] = len(values)

    if len(values) == 0:
        results.append(row)
        continue

    row["mean_effect"] = values.mean()
    row["median_effect"] = values.median()
    row["std_effect"] = values.std(ddof=1)

    # -------------------------------------------------------------------------
    # One-sample test against zero
    # -------------------------------------------------------------------------

    if len(values) >= 2 and values.std(ddof=1) > 0:

        t_stat, p_value = stats.ttest_1samp(
            values,
            0.0
        )

        row["t_stat"] = t_stat
        row["p_value"] = p_value

        # Cohen's d
        row["cohen_d"] = (
            values.mean() / values.std(ddof=1)
        )

    else:
        row["t_stat"] = np.nan
        row["p_value"] = np.nan
        row["cohen_d"] = np.nan

    results.append(row)

stats_df = pd.DataFrame(results)

# =============================================================================
# FDR CORRECTION
# =============================================================================

print("\n" + "=" * 90)
print("MULTIPLE-COMPARISON CORRECTION")
print("=" * 90)

valid_p = stats_df["p_value"].notna()

stats_df["p_fdr"] = np.nan
stats_df["significant_fdr"] = False

if valid_p.sum() > 0:

    pvals = stats_df.loc[valid_p, "p_value"].to_numpy()

    order = np.argsort(pvals)
    ranked = pvals[order]

    m = len(ranked)

    adjusted = np.empty(m)

    for i in range(m):
        rank = i + 1
        adjusted[i] = ranked[i] * m / rank

    # enforce monotonicity
    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1
    )

    restored = np.empty(m)

    for i, idx in enumerate(order):
        restored[idx] = adjusted[i]

    stats_df.loc[valid_p, "p_fdr"] = restored

    stats_df.loc[
        valid_p,
        "significant_fdr"
    ] = restored < 0.05

# =============================================================================
# EFFECT INTERPRETATION
# =============================================================================

def classify_effect(d):

    if pd.isna(d):
        return "not_available"

    d_abs = abs(d)

    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


stats_df["effect_magnitude"] = (
    stats_df["cohen_d"]
    .apply(classify_effect)
)

stats_df["direction"] = np.where(
    stats_df["mean_effect"] > 0,
    "positive",
    np.where(
        stats_df["mean_effect"] < 0,
        "negative",
        "zero"
    )
)

# =============================================================================
# SORT RESULTS
# =============================================================================

stats_df = stats_df.sort_values(
    by=["significant_fdr", "p_fdr"],
    ascending=[False, True],
    na_position="last"
)

# =============================================================================
# SUBJECT-LEVEL ROBUSTNESS
# =============================================================================

print("\n" + "=" * 90)
print("SUBJECT-LEVEL ROBUSTNESS")
print("=" * 90)

subject_results = []

if "subject" in df.columns:

    for feature_name, group in df.groupby(
        "feature",
        dropna=False
    ):

        effect_col = None

        for candidate in [
            "effect",
            "difference",
            "delta",
            "mean_difference",
            "effect_size",
            "perturbation_effect",
        ]:

            if candidate in group.columns:
                effect_col = candidate
                break

        if effect_col is None:
            candidates = [
                c for c in effect_cols
                if c in group.columns
            ]

            if candidates:
                effect_col = candidates[0]

        if effect_col is None:
            continue

        subject_effects = (
            group
            .groupby("subject")[effect_col]
            .mean()
            .dropna()
        )

        if len(subject_effects) == 0:
            continue

        positive_fraction = (
            (subject_effects > 0).mean()
        )

        negative_fraction = (
            (subject_effects < 0).mean()
        )

        consistency = max(
            positive_fraction,
            negative_fraction
        )

        subject_results.append({
            "feature": feature_name,
            "n_subjects": len(subject_effects),
            "mean_subject_effect":
                subject_effects.mean(),
            "median_subject_effect":
                subject_effects.median(),
            "subject_std":
                subject_effects.std(ddof=1),
            "positive_fraction":
                positive_fraction,
            "negative_fraction":
                negative_fraction,
            "directional_consistency":
                consistency,
        })

subject_df_out = pd.DataFrame(subject_results)

if not subject_df_out.empty:

    subject_df_out = subject_df_out.sort_values(
        "directional_consistency",
        ascending=False
    )

# =============================================================================
# QC
# =============================================================================

print("\n" + "=" * 90)
print("FINAL QC")
print("=" * 90)

nan_values = int(
    stats_df.select_dtypes(include=[np.number])
    .isna()
    .sum()
    .sum()
)

inf_values = int(
    np.isinf(
        stats_df.select_dtypes(include=[np.number])
        .to_numpy(dtype=float)
    ).sum()
)

print(f"Features analyzed:       {len(stats_df)}")
print(f"Significant FDR < 0.05:  {int(stats_df['significant_fdr'].sum())}")
print(f"NaN numeric values:      {nan_values}")
print(f"Inf numeric values:      {inf_values}")

# =============================================================================
# SAVE
# =============================================================================

stats_df.to_csv(
    OUTPUT_STATS,
    index=False
)

subject_df_out.to_csv(
    OUTPUT_SUBJECT,
    index=False
)

qc = pd.DataFrame([{
    "input_rows": len(df),
    "input_columns": len(df.columns),
    "features_analyzed": len(stats_df),
    "significant_fdr_features":
        int(stats_df["significant_fdr"].sum()),
    "subject_robustness_features":
        len(subject_df_out),
    "nan_numeric_values": nan_values,
    "inf_numeric_values": inf_values,
}])

qc.to_csv(
    OUTPUT_QC,
    index=False
)

print("\n" + "=" * 90)
print("PERTURBATION STATISTICAL ANALYSIS COMPLETE")
print("=" * 90)

print(f"Results:")
print(OUTPUT_STATS)

print(f"\nSubject robustness:")
print(OUTPUT_SUBJECT)

print(f"\nQC:")
print(OUTPUT_QC)

print("\n" + "=" * 90)
print("STATUS: PASS - PERTURBATION STATISTICAL ANALYSIS CREATED")
print("=" * 90)