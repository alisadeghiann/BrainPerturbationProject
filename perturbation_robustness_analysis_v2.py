from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# =============================================================================
# PERTURBATION / WHAT-IF ROBUSTNESS ANALYSIS V2
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
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
    / "robustness_v2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_OUT = (
    OUTPUT_DIR
    / "perturbation_robustness_results_v2.csv"
)

SUBJECT_OUT = (
    OUTPUT_DIR
    / "perturbation_subject_robustness_v2.csv"
)

QC_OUT = (
    OUTPUT_DIR
    / "perturbation_robustness_qc_v2.csv"
)


# =============================================================================
# HELPERS
# =============================================================================

def banner(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def cohen_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx = len(x)
    ny = len(y)

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled = np.sqrt(
        ((nx - 1) * vx + (ny - 1) * vy)
        / (nx + ny - 2)
    )

    if pooled == 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled


# =============================================================================
# LOAD
# =============================================================================

banner("PERTURBATION / WHAT-IF ROBUSTNESS ANALYSIS V2")

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

print(f"Input rows:       {len(df):,}")
print(f"Input columns:    {len(df.columns):,}")


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

required = {
    "target",
    "feature",
    "mean_difference",
    "cohen_d",
    "p_value",
    "p_fdr",
}

missing = sorted(required - set(df.columns))

if missing:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# =============================================================================
# CLEAN TYPES
# =============================================================================

numeric_cols = [
    "mean_difference",
    "cohen_d",
    "p_value",
    "p_fdr",
]

for col in numeric_cols:
    df[col] = safe_numeric(df[col])


# =============================================================================
# BASIC QC
# =============================================================================

banner("BASIC QC")

nan_count = int(df[numeric_cols].isna().sum().sum())

inf_count = int(
    np.isinf(
        df[numeric_cols]
        .to_numpy(dtype=float)
    ).sum()
)

print(f"NaN numeric values: {nan_count}")
print(f"Inf numeric values: {inf_count}")

if inf_count > 0:
    raise RuntimeError(
        "Infinite values detected. STOP."
    )


# =============================================================================
# FEATURE LIST
# =============================================================================

features = (
    df["feature"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

targets = (
    df["target"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

print(f"Features: {len(features)}")
print(f"Targets:  {targets}")


# =============================================================================
# EFFECT ROBUSTNESS
# =============================================================================

banner("EFFECT ROBUSTNESS")

robust_rows = []

for target in targets:

    sub = df[df["target"] == target].copy()

    for feature in features:

        row = sub[sub["feature"] == feature]

        if row.empty:
            continue

        r = row.iloc[0]

        effect = r["cohen_d"]
        p_fdr = r["p_fdr"]
        mean_diff = r["mean_difference"]

        if not np.isfinite(effect):
            robustness = "not_estimable"

        elif abs(effect) >= 0.8:
            robustness = "large"

        elif abs(effect) >= 0.5:
            robustness = "medium"

        elif abs(effect) >= 0.2:
            robustness = "small"

        else:
            robustness = "negligible"

        if np.isfinite(p_fdr):
            significant = bool(p_fdr < 0.05)
        else:
            significant = False

        robust_rows.append(
            {
                "target": target,
                "feature": feature,
                "mean_difference": mean_diff,
                "cohen_d": effect,
                "p_fdr": p_fdr,
                "fdr_significant": significant,
                "effect_category": robustness,
            }
        )


robustness_df = pd.DataFrame(robust_rows)


# =============================================================================
# DIRECTION CONSISTENCY
# =============================================================================

banner("DIRECTION CONSISTENCY")

direction_rows = []

for target in targets:

    sub = robustness_df[
        robustness_df["target"] == target
    ].copy()

    for feature in sub["feature"].unique():

        r = sub[sub["feature"] == feature]

        if r.empty:
            continue

        effect = r["cohen_d"].iloc[0]

        if not np.isfinite(effect):
            direction = "undefined"
        elif effect > 0:
            direction = "positive"
        elif effect < 0:
            direction = "negative"
        else:
            direction = "zero"

        direction_rows.append(
            {
                "target": target,
                "feature": feature,
                "direction": direction,
                "cohen_d": effect,
            }
        )

direction_df = pd.DataFrame(direction_rows)


# =============================================================================
# FDR SIGNIFICANCE SUMMARY
# =============================================================================

banner("FDR SIGNIFICANCE SUMMARY")

summary_rows = []

for target in targets:

    sub = robustness_df[
        robustness_df["target"] == target
    ]

    n_features = len(sub)

    n_sig = int(
        sub["fdr_significant"].sum()
    )

    n_small_or_larger = int(
        (
            sub["cohen_d"].abs() >= 0.2
        ).sum()
    )

    n_medium_or_larger = int(
        (
            sub["cohen_d"].abs() >= 0.5
        ).sum()
    )

    summary_rows.append(
        {
            "target": target,
            "features": n_features,
            "fdr_significant": n_sig,
            "effect_ge_0.20": n_small_or_larger,
            "effect_ge_0.50": n_medium_or_larger,
        }
    )

summary_df = pd.DataFrame(summary_rows)

print(summary_df.to_string(index=False))


# =============================================================================
# TOP EFFECTS
# =============================================================================

banner("TOP EFFECTS")

top = (
    robustness_df
    .sort_values(
        by=["target", "cohen_d"],
        key=lambda x: x.abs()
        if x.name == "cohen_d"
        else x,
        ascending=[True, False],
    )
)

print(
    top[
        [
            "target",
            "feature",
            "mean_difference",
            "cohen_d",
            "p_fdr",
            "fdr_significant",
            "effect_category",
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# =============================================================================
# SUBJECT ROBUSTNESS
# =============================================================================
#
# The statistical V2 file may not contain subject-level raw observations.
# Therefore we explicitly inspect available subject columns instead of
# fabricating subject-level statistics.
# =============================================================================

banner("SUBJECT-LEVEL ROBUSTNESS CHECK")

subject_candidates = [
    "subject",
    "sub",
    "participant",
]

subject_col = None

for col in subject_candidates:
    if col in df.columns:
        subject_col = col
        break

if subject_col is None:

    print(
        "No subject-level raw observation column exists "
        "in the statistical result file."
    )

    subject_df = pd.DataFrame(
        columns=[
            "target",
            "feature",
            "subject_n",
            "subject_effect_mean",
            "subject_effect_median",
            "subject_effect_std",
            "positive_subject_fraction",
        ]
    )

else:

    print(f"Subject column detected: {subject_col}")

    subject_rows = []

    temp = df.copy()

    temp["effect"] = safe_numeric(
        temp["cohen_d"]
    )

    for target in targets:

        t = temp[
            temp["target"] == target
        ]

        for feature in features:

            x = t[
                t["feature"] == feature
            ]

            effects = x["effect"].dropna()

            if len(effects) == 0:
                continue

            positive_fraction = float(
                (effects > 0).mean()
            )

            subject_rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "subject_n": len(effects),
                    "subject_effect_mean": effects.mean(),
                    "subject_effect_median": effects.median(),
                    "subject_effect_std": effects.std(),
                    "positive_subject_fraction": positive_fraction,
                }
            )

    subject_df = pd.DataFrame(subject_rows)


# =============================================================================
# FINAL ROBUSTNESS CLASSIFICATION
# =============================================================================

banner("FINAL ROBUSTNESS CLASSIFICATION")

classification_rows = []

for _, r in robustness_df.iterrows():

    effect = r["cohen_d"]
    p_fdr = r["p_fdr"]

    if not np.isfinite(effect):
        classification = "not_estimable"

    elif not np.isfinite(p_fdr):
        classification = "effect_only"

    elif p_fdr < 0.05 and abs(effect) >= 0.5:
        classification = "strong"

    elif p_fdr < 0.05 and abs(effect) >= 0.2:
        classification = "moderate"

    elif p_fdr < 0.05:
        classification = "statistically_significant_small"

    else:
        classification = "not_fdr_significant"

    classification_rows.append(
        {
            "target": r["target"],
            "feature": r["feature"],
            "mean_difference": r["mean_difference"],
            "cohen_d": effect,
            "p_fdr": p_fdr,
            "classification": classification,
        }
    )

classification_df = pd.DataFrame(
    classification_rows
)


# =============================================================================
# QC
# =============================================================================

banner("FINAL QC")

final_nan = int(
    classification_df[
        [
            "mean_difference",
            "cohen_d",
            "p_fdr",
        ]
    ]
    .isna()
    .sum()
    .sum()
)

final_inf = int(
    np.isinf(
        classification_df[
            [
                "mean_difference",
                "cohen_d",
                "p_fdr",
            ]
        ]
        .to_numpy(dtype=float)
    ).sum()
)

print(
    f"Features analyzed:        {len(features)}"
)

print(
    f"Targets analyzed:         {len(targets)}"
)

print(
    f"Statistical rows:         {len(classification_df)}"
)

print(
    f"NaN numeric values:       {final_nan}"
)

print(
    f"Inf numeric values:       {final_inf}"
)


# =============================================================================
# SAVE
# =============================================================================

classification_df.to_csv(
    RESULTS_OUT,
    index=False,
)

subject_df.to_csv(
    SUBJECT_OUT,
    index=False,
)

qc = pd.DataFrame(
    [
        {
            "input_rows": len(df),
            "input_columns": len(df.columns),
            "features": len(features),
            "targets": len(targets),
            "statistical_rows": len(classification_df),
            "nan_numeric_values": final_nan,
            "inf_numeric_values": final_inf,
            "fdr_significant_total": int(
                robustness_df["fdr_significant"].sum()
            ),
            "status": "PASS",
        }
    ]
)

qc.to_csv(
    QC_OUT,
    index=False,
)


# =============================================================================
# FINAL
# =============================================================================

banner(
    "PERTURBATION / WHAT-IF ROBUSTNESS ANALYSIS V2 COMPLETE"
)

print("Results:")
print(RESULTS_OUT)

print("\nSubject robustness:")
print(SUBJECT_OUT)

print("\nQC:")
print(QC_OUT)

print(
    "\nSTATUS: PASS - ROBUSTNESS ANALYSIS CREATED"
)