# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# FINAL SCIENTIFIC EEG PERTURBATION / WHAT-IF EVIDENCE SYNTHESIS V1
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_master_v1"
    / "final_perturbation_master_v1.csv"
)

OUTDIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_scientific_synthesis_v1"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FEATURE = OUTDIR / "final_scientific_feature_synthesis_v1.csv"
OUTPUT_TARGET = OUTDIR / "final_scientific_target_synthesis_v1.csv"
OUTPUT_FREQUENCY = OUTDIR / "final_scientific_frequency_synthesis_v1.csv"
OUTPUT_REGION = OUTDIR / "final_scientific_region_synthesis_v1.csv"
OUTPUT_QC = OUTDIR / "final_scientific_synthesis_qc_v1.csv"


# =============================================================================
# HELPERS
# =============================================================================

def find_column(df, candidates):
    """
    Return the first matching column from candidates.
    Matching is case-insensitive.
    """
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def numeric_or_nan(series):
    return pd.to_numeric(series, errors="coerce")


def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.mean() if len(s) else np.nan


def safe_median(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.median() if len(s) else np.nan


# =============================================================================
# LOAD
# =============================================================================

print("=" * 90)
print("FINAL SCIENTIFIC EEG PERTURBATION / WHAT-IF EVIDENCE SYNTHESIS V1")
print("=" * 90)

print(f"Project root:\n{BASE}")

if not INPUT.exists():
    raise FileNotFoundError(
        f"\nFinal perturbation master file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

print("\n" + "=" * 90)
print("INPUT VALIDATION")
print("=" * 90)

print(f"Rows:       {len(df):,}")
print(f"Columns:    {len(df.columns):,}")


# =============================================================================
# IDENTIFY CORE COLUMNS
# =============================================================================

target_col = find_column(
    df,
    ["target"]
)

feature_col = find_column(
    df,
    ["feature"]
)

mean_diff_col = find_column(
    df,
    ["mean_difference", "mean_diff"]
)

cohen_col = find_column(
    df,
    ["cohen_d", "effect_size"]
)

p_fdr_col = find_column(
    df,
    ["p_fdr", "fdr", "adjusted_p_value"]
)

direction_col = find_column(
    df,
    ["direction", "statistical_direction", "target_direction"]
)

robustness_col = find_column(
    df,
    [
        "target_robustness_class",
        "robustness_class",
        "subject_robustness_class"
    ]
)

consistency_col = find_column(
    df,
    [
        "target_direction_consistency",
        "direction_consistency"
    ]
)

cross_target_col = find_column(
    df,
    [
        "cross_target_pattern",
        "cross_target_class"
    ]
)

interpretation_col = find_column(
    df,
    [
        "scientific_interpretation",
        "interpretation",
        "pattern_class"
    ]
)

priority_col = find_column(
    df,
    [
        "final_scientific_priority_score",
        "scientific_priority_score",
        "final_priority_score"
    ]
)

frequency_col = find_column(
    df,
    ["frequency", "band"]
)

region_col = find_column(
    df,
    ["region", "brain_region"]
)


required = {
    "target": target_col,
    "feature": feature_col,
    "mean_difference": mean_diff_col,
    "cohen_d": cohen_col,
    "p_fdr": p_fdr_col,
}

missing = [
    name for name, value in required.items()
    if value is None
]

if missing:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(f"- {x}" for x in missing)
    )

print("\nCore columns:")
for name, value in required.items():
    print(f"{name:20s}: {value}")

print("\nOptional columns:")
for name, value in {
    "direction": direction_col,
    "robustness": robustness_col,
    "consistency": consistency_col,
    "cross_target": cross_target_col,
    "interpretation": interpretation_col,
    "priority": priority_col,
    "frequency": frequency_col,
    "region": region_col,
}.items():
    print(f"{name:20s}: {value}")


# =============================================================================
# BASIC VALIDATION
# =============================================================================

df[target_col] = df[target_col].astype(str)
df[feature_col] = df[feature_col].astype(str)

df[mean_diff_col] = numeric_or_nan(df[mean_diff_col])
df[cohen_col] = numeric_or_nan(df[cohen_col])
df[p_fdr_col] = numeric_or_nan(df[p_fdr_col])

if priority_col:
    df[priority_col] = numeric_or_nan(df[priority_col])

if consistency_col:
    df[consistency_col] = numeric_or_nan(df[consistency_col])


# =============================================================================
# TARGET VALIDATION
# =============================================================================

print("\n" + "=" * 90)
print("TARGET VALIDATION")
print("=" * 90)

print(df[target_col].value_counts(dropna=False))

targets = sorted(df[target_col].dropna().unique())

if "remember" in targets and "correct" in targets:
    print("\nExpected targets found: remember + correct")
else:
    print("\nWARNING: Expected remember/correct targets were not both detected.")


# =============================================================================
# DUPLICATE VALIDATION
# =============================================================================

duplicate_count = df.duplicated(
    subset=[target_col, feature_col]
).sum()

print("\n" + "=" * 90)
print("TARGET / FEATURE DUPLICATE CHECK")
print("=" * 90)

print(f"Duplicate target-feature rows: {duplicate_count}")

if duplicate_count != 0:
    raise ValueError(
        "Duplicate target-feature rows detected."
    )


# =============================================================================
# SCIENTIFIC EVIDENCE CLASSIFICATION
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING SCIENTIFIC EVIDENCE CLASSIFICATION")
print("=" * 90)

work = df.copy()

work["abs_cohen_d"] = work[cohen_col].abs()

work["fdr_significant"] = (
    work[p_fdr_col] < 0.05
)

# -------------------------------------------------------------------------
# Effect magnitude
# -------------------------------------------------------------------------

def classify_effect(d):
    if pd.isna(d):
        return "not_available"

    d = abs(float(d))

    if d >= 0.80:
        return "large"
    elif d >= 0.50:
        return "moderate"
    elif d >= 0.20:
        return "small"
    else:
        return "negligible"


work["effect_magnitude_final"] = (
    work[cohen_col]
    .apply(classify_effect)
)


# -------------------------------------------------------------------------
# Evidence class
# -------------------------------------------------------------------------

def evidence_class(row):

    sig = bool(row["fdr_significant"])
    d = row["abs_cohen_d"]

    robustness = (
        str(row[robustness_col]).lower()
        if robustness_col
        else ""
    )

    consistency = (
        row[consistency_col]
        if consistency_col
        else np.nan
    )

    if not sig:
        return "non_significant"

    if pd.notna(d) and d >= 0.50:
        return "strong_statistical_effect"

    if (
        pd.notna(d)
        and d >= 0.20
        and (
            "moderate" in robustness
            or (
                pd.notna(consistency)
                and consistency >= 0.65
            )
        )
    ):
        return "moderate_replicated_effect"

    if (
        pd.notna(d)
        and d >= 0.20
    ):
        return "small_to_moderate_effect"

    if (
        "moderate" in robustness
        or (
            pd.notna(consistency)
            and consistency >= 0.60
        )
    ):
        return "replicated_directional_effect"

    return "significant_but_weak"


work["final_evidence_class"] = work.apply(
    evidence_class,
    axis=1
)


# =============================================================================
# SCIENTIFIC PRIORITY
# =============================================================================

print("\n" + "=" * 90)
print("CALCULATING FINAL SCIENTIFIC PRIORITY")
print("=" * 90)


def priority_score(row):

    score = 0.0

    # FDR
    if row["fdr_significant"]:
        score += 3.0

    # Effect size
    d = row["abs_cohen_d"]

    if pd.notna(d):
        if d >= 0.80:
            score += 4.0
        elif d >= 0.50:
            score += 3.0
        elif d >= 0.20:
            score += 2.0
        elif d >= 0.10:
            score += 1.0

    # Subject-level directional consistency
    if pd.notna(row["target_consistency_internal"]):
        c = row["target_consistency_internal"]

        if c >= 0.75:
            score += 2.0
        elif c >= 0.65:
            score += 1.5
        elif c >= 0.55:
            score += 1.0

    # Robustness
    robustness = str(row["robustness_internal"]).lower()

    if "moderate" in robustness:
        score += 2.0
    elif "weak" in robustness:
        score += 1.0

    return score


if consistency_col:
    work["target_consistency_internal"] = work[
        consistency_col
    ]
else:
    work["target_consistency_internal"] = np.nan


if robustness_col:
    work["robustness_internal"] = work[
        robustness_col
    ].fillna("")
else:
    work["robustness_internal"] = ""


work["final_priority_score_v1"] = work.apply(
    priority_score,
    axis=1
)


# =============================================================================
# FEATURE-LEVEL FINAL SYNTHESIS
# =============================================================================

print("\n" + "=" * 90)
print("FEATURE-LEVEL SYNTHESIS")
print("=" * 90)

feature_rows = []

for feature, g in work.groupby(feature_col):

    remember = g[
        g[target_col].str.lower() == "remember"
    ]

    correct = g[
        g[target_col].str.lower() == "correct"
    ]

    row = {
        "feature": feature,

        "targets_present": len(g),

        "remember_fdr_significant":
            bool(
                len(remember)
                and remember["fdr_significant"].iloc[0]
            ),

        "correct_fdr_significant":
            bool(
                len(correct)
                and correct["fdr_significant"].iloc[0]
            ),

        "remember_cohen_d":
            remember[cohen_col].iloc[0]
            if len(remember) else np.nan,

        "correct_cohen_d":
            correct[cohen_col].iloc[0]
            if len(correct) else np.nan,

        "remember_mean_difference":
            remember[mean_diff_col].iloc[0]
            if len(remember) else np.nan,

        "correct_mean_difference":
            correct[mean_diff_col].iloc[0]
            if len(correct) else np.nan,

        "remember_p_fdr":
            remember[p_fdr_col].iloc[0]
            if len(remember) else np.nan,

        "correct_p_fdr":
            correct[p_fdr_col].iloc[0]
            if len(correct) else np.nan,
    }

    row["remember_abs_cohen_d"] = abs(
        row["remember_cohen_d"]
    ) if pd.notna(row["remember_cohen_d"]) else np.nan

    row["correct_abs_cohen_d"] = abs(
        row["correct_cohen_d"]
    ) if pd.notna(row["correct_cohen_d"]) else np.nan

    # Dominance
    r = row["remember_abs_cohen_d"]
    c = row["correct_abs_cohen_d"]

    if pd.notna(r) and pd.notna(c):

        if r > c * 1.25:
            dominance = "remember_dominant"

        elif c > r * 1.25:
            dominance = "correct_dominant"

        else:
            dominance = "balanced_effect"

    else:
        dominance = "not_available"

    row["effect_dominance"] = dominance

    # Direction relationship
    rd = row["remember_mean_difference"]
    cd = row["correct_mean_difference"]

    if pd.notna(rd) and pd.notna(cd):

        if rd == 0 or cd == 0:
            direction_relationship = "zero_or_undefined"

        elif np.sign(rd) == np.sign(cd):
            direction_relationship = "same_direction"

        else:
            direction_relationship = "opposite_direction"

    else:
        direction_relationship = "not_available"

    row["direction_relationship"] = direction_relationship

    # Overall evidence
    sig_count = int(
        row["remember_fdr_significant"]
    ) + int(
        row["correct_fdr_significant"]
    )

    if sig_count == 2:
        overall = "both_targets_significant"

    elif sig_count == 1:
        if row["remember_fdr_significant"]:
            overall = "remember_only_significant"
        else:
            overall = "correct_only_significant"

    else:
        overall = "neither_target_significant"

    row["overall_cross_target_evidence"] = overall

    feature_rows.append(row)


feature_summary = pd.DataFrame(feature_rows)

feature_summary = feature_summary.sort_values(
    by=[
        "correct_abs_cohen_d",
        "remember_abs_cohen_d"
    ],
    ascending=False,
    na_position="last"
)


# =============================================================================
# TARGET-LEVEL SYNTHESIS
# =============================================================================

print("\n" + "=" * 90)
print("TARGET-LEVEL SYNTHESIS")
print("=" * 90)

target_rows = []

for target, g in work.groupby(target_col):

    target_name = str(target)

    sig = g["fdr_significant"]

    abs_d = g["abs_cohen_d"]

    row = {
        "target": target_name,
        "features_analyzed": len(g),
        "fdr_significant_features": int(sig.sum()),
        "fdr_significant_fraction":
            float(sig.mean()) if len(g) else np.nan,

        "mean_abs_cohen_d":
            safe_mean(abs_d),

        "median_abs_cohen_d":
            safe_median(abs_d),

        "max_abs_cohen_d":
            abs_d.max() if len(abs_d) else np.nan,

        "large_effect_features":
            int((abs_d >= 0.80).sum()),

        "moderate_effect_features":
            int(
                (
                    (abs_d >= 0.50)
                    & (abs_d < 0.80)
                ).sum()
            ),

        "small_effect_features":
            int(
                (
                    (abs_d >= 0.20)
                    & (abs_d < 0.50)
                ).sum()
            ),
    }

    if consistency_col:
        row["mean_direction_consistency"] = safe_mean(
            g[consistency_col]
        )
    else:
        row["mean_direction_consistency"] = np.nan

    target_rows.append(row)


target_summary = pd.DataFrame(target_rows)


# =============================================================================
# FREQUENCY SYNTHESIS
# =============================================================================

print("\n" + "=" * 90)
print("FREQUENCY-LEVEL SYNTHESIS")
print("=" * 90)

if frequency_col:

    frequency_summary = (
        work.groupby(
            [target_col, frequency_col],
            dropna=False
        )
        .agg(
            features=(feature_col, "count"),
            fdr_significant=("fdr_significant", "sum"),
            mean_abs_cohen_d=("abs_cohen_d", "mean"),
            median_abs_cohen_d=("abs_cohen_d", "median"),
            max_abs_cohen_d=("abs_cohen_d", "max")
        )
        .reset_index()
    )

else:

    frequency_summary = pd.DataFrame(
        columns=[
            target_col,
            "frequency",
            "features",
            "fdr_significant",
            "mean_abs_cohen_d",
            "median_abs_cohen_d",
            "max_abs_cohen_d"
        ]
    )


# =============================================================================
# REGION SYNTHESIS
# =============================================================================

print("\n" + "=" * 90)
print("REGION-LEVEL SYNTHESIS")
print("=" * 90)

if region_col:

    region_summary = (
        work.groupby(
            [target_col, region_col],
            dropna=False
        )
        .agg(
            features=(feature_col, "count"),
            fdr_significant=("fdr_significant", "sum"),
            mean_abs_cohen_d=("abs_cohen_d", "mean"),
            median_abs_cohen_d=("abs_cohen_d", "median"),
            max_abs_cohen_d=("abs_cohen_d", "max")
        )
        .reset_index()
    )

else:

    region_summary = pd.DataFrame(
        columns=[
            target_col,
            "region",
            "features",
            "fdr_significant",
            "mean_abs_cohen_d",
            "median_abs_cohen_d",
            "max_abs_cohen_d"
        ]
    )


# =============================================================================
# FINAL MASTER SCIENTIFIC TABLE
# =============================================================================

final_columns = [
    target_col,
    feature_col,
    mean_diff_col,
    cohen_col,
    "abs_cohen_d",
    p_fdr_col,
    "fdr_significant",
    "effect_magnitude_final",
    "final_evidence_class",
    "final_priority_score_v1",
]

for optional in [
    direction_col,
    robustness_col,
    consistency_col,
    cross_target_col,
    interpretation_col,
    frequency_col,
    region_col,
]:

    if optional and optional not in final_columns:
        final_columns.append(optional)


final_scientific = work[
    [c for c in final_columns if c in work.columns]
].copy()

final_scientific = final_scientific.sort_values(
    by=[
        "final_priority_score_v1",
        "abs_cohen_d"
    ],
    ascending=False,
    na_position="last"
)


# =============================================================================
# SAVE
# =============================================================================

print("\n" + "=" * 90)
print("SAVING FINAL SCIENTIFIC SYNTHESIS")
print("=" * 90)

final_scientific.to_csv(
    OUTPUT_FEATURE,
    index=False
)

target_summary.to_csv(
    OUTPUT_TARGET,
    index=False
)

frequency_summary.to_csv(
    OUTPUT_FREQUENCY,
    index=False
)

region_summary.to_csv(
    OUTPUT_REGION,
    index=False
)


# =============================================================================
# FINAL QC
# =============================================================================

numeric_cols = final_scientific.select_dtypes(
    include=[np.number]
).columns

nan_numeric = int(
    final_scientific[numeric_cols]
    .isna()
    .sum()
    .sum()
)

inf_numeric = int(
    np.isinf(
        final_scientific[numeric_cols]
        .to_numpy(dtype=float)
    ).sum()
)

duplicate_final = int(
    final_scientific.duplicated(
        subset=[target_col, feature_col]
    ).sum()
)

fdr_significant = int(
    final_scientific["fdr_significant"].sum()
)

qc = pd.DataFrame(
    [
        {
            "rows": len(final_scientific),
            "unique_features": final_scientific[
                feature_col
            ].nunique(),

            "targets": final_scientific[
                target_col
            ].nunique(),

            "fdr_significant_rows":
                fdr_significant,

            "nan_numeric_cells":
                nan_numeric,

            "inf_numeric_cells":
                inf_numeric,

            "duplicate_target_feature":
                duplicate_final,

            "input_rows":
                len(df),

            "input_columns":
                len(df.columns),
        }
    ]
)

qc.to_csv(
    OUTPUT_QC,
    index=False
)


# =============================================================================
# REPORT
# =============================================================================

print("\n" + "=" * 90)
print("FINAL SCIENTIFIC SYNTHESIS QC")
print("=" * 90)

print(f"Rows:                         {len(final_scientific)}")
print(
    f"Unique features:              "
    f"{final_scientific[feature_col].nunique()}"
)
print(
    f"Targets:                      "
    f"{final_scientific[target_col].nunique()}"
)
print(
    f"FDR-significant rows:         "
    f"{fdr_significant}"
)
print(
    f"NaN numeric cells:            "
    f"{nan_numeric}"
)
print(
    f"Inf numeric cells:            "
    f"{inf_numeric}"
)
print(
    f"Duplicate target-feature:     "
    f"{duplicate_final}"
)


# =============================================================================
# TOP RESULTS
# =============================================================================

print("\n" + "=" * 90)
print("TOP SCIENTIFIC EVIDENCE")
print("=" * 90)

display_cols = [
    target_col,
    feature_col,
    cohen_col,
    p_fdr_col,
    "effect_magnitude_final",
    "final_evidence_class",
    "final_priority_score_v1"
]

display_cols = [
    c for c in display_cols
    if c in final_scientific.columns
]

print(
    final_scientific[
        display_cols
    ].head(30).to_string(index=False)
)


# =============================================================================
# STATUS
# =============================================================================

if (
    nan_numeric == 0
    and inf_numeric == 0
    and duplicate_final == 0
):
    status = "PASS - FINAL SCIENTIFIC EVIDENCE SYNTHESIS CREATED"
else:
    status = "REVIEW_REQUIRED - FINAL SCIENTIFIC SYNTHESIS QC FAILED"


print("\n" + "=" * 90)
print("FINAL SCIENTIFIC EEG PERTURBATION / WHAT-IF EVIDENCE SYNTHESIS V1 COMPLETE")
print("=" * 90)

print("\nSaved:")
print(OUTPUT_FEATURE)
print(OUTPUT_TARGET)
print(OUTPUT_FREQUENCY)
print(OUTPUT_REGION)
print(OUTPUT_QC)

print("\n" + "=" * 90)
print(f"STATUS: {status}")
print("=" * 90)