from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# FINAL EVIDENCE RECONCILIATION V1
# EEG PERTURBATION / WHAT-IF PROJECT
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

PERTURBATION = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_evidence_v2"
    / "final_scientific_perturbation_evidence_v2.csv"
)

CROSS_TARGET = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "cross_target_v1"
    / "cross_target_perturbation_comparison_v1.csv"
)

ML_STABILITY = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "ml_feature_stability_v1"
    / "ml_feature_stability_v1_results.csv"
)

LOSO = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "publication_grade_loso_v1"
    / "publication_grade_loso_sensitivity_v1.csv"
)

NULL_VALIDATION = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "null_validation_v1"
    / "perturbation_null_validation_v1.csv"
)

R_RESULTS = (
    BASE
    / "features"
    / "statistical_analysis_r_v1"
    / "r_mixed_effects_statistical_results_v1.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_evidence_reconciliation_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_FILE = OUTPUT_DIR / "final_evidence_reconciliation_v1.csv"
SUMMARY_FILE = OUTPUT_DIR / "final_evidence_reconciliation_summary_v1.csv"
QC_FILE = OUTPUT_DIR / "final_evidence_reconciliation_qc_v1.csv"


# =============================================================================
# HELPERS
# =============================================================================

def read_required(path, name):
    print(f"Loading {name}:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
            f"Please verify that the previous pipeline step was completed."
        )

    df = pd.read_csv(path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()

    return df


def first_existing(df, candidates, default=np.nan):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(default, index=df.index)


def numeric_series(df, candidates, default=0.0):
    s = first_existing(df, candidates, default)

    return pd.to_numeric(s, errors="coerce").fillna(default)


def text_series(df, candidates, default="unknown"):
    s = first_existing(df, candidates, default)

    return s.astype(str).replace(
        {"nan": default, "None": default}
    )


def normalize_feature_column(df):
    candidates = [
        "feature",
        "scientific_feature",
        "feature_name"
    ]

    for c in candidates:
        if c in df.columns:
            return df[c].astype(str)

    raise ValueError(
        "Could not identify feature column. "
        f"Available columns: {list(df.columns)}"
    )


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 90)
print("FINAL SCIENTIFIC EVIDENCE RECONCILIATION V1")
print("=" * 90)
print(f"Project root: {BASE}")
print()


print("=" * 90)
print("LOADING EVIDENCE LAYERS")
print("=" * 90)

pert = read_required(
    PERTURBATION,
    "FINAL PERTURBATION EVIDENCE"
)

cross = read_required(
    CROSS_TARGET,
    "CROSS-TARGET EVIDENCE"
)

ml = read_required(
    ML_STABILITY,
    "ML FEATURE STABILITY"
)

loso = read_required(
    LOSO,
    "LOSO SENSITIVITY"
)

null = read_required(
    NULL_VALIDATION,
    "NULL VALIDATION"
)

r = read_required(
    R_RESULTS,
    "R MIXED-EFFECTS RESULTS"
)


# =============================================================================
# IDENTIFY COMMON FEATURES
# =============================================================================

print("=" * 90)
print("IDENTIFYING COMMON SCIENTIFIC FEATURES")
print("=" * 90)

pert["feature_std"] = normalize_feature_column(pert)
cross["feature_std"] = normalize_feature_column(cross)
ml["feature_std"] = normalize_feature_column(ml)
loso["feature_std"] = normalize_feature_column(loso)
null["feature_std"] = normalize_feature_column(null)
r["feature_std"] = normalize_feature_column(r)

feature_sets = [
    set(pert["feature_std"]),
    set(cross["feature_std"]),
    set(ml["feature_std"]),
    set(loso["feature_std"]),
    set(null["feature_std"]),
    set(r["feature_std"]),
]

common_features = sorted(set.intersection(*feature_sets))

print(f"Common features across all layers: {len(common_features)}")
print()


if len(common_features) == 0:
    raise ValueError(
        "No common features were found across all evidence layers."
    )


# =============================================================================
# TARGET-LEVEL PERTURBATION EVIDENCE
# =============================================================================

print("=" * 90)
print("BUILDING PERTURBATION EVIDENCE")
print("=" * 90)

pert2 = pert[pert["feature_std"].isin(common_features)].copy()

pert_sig = (
    pert2
    .groupby("feature_std")
    .agg(
        perturbation_rows=("feature_std", "size"),
        perturbation_significant=(
            "p_fdr",
            lambda x: int((pd.to_numeric(x, errors="coerce") < 0.05).sum())
        ),
        perturbation_min_p=(
            "p_fdr",
            lambda x: pd.to_numeric(x, errors="coerce").min()
        ),
        perturbation_max_abs_d=(
            "cohen_d",
            lambda x: pd.to_numeric(
                x, errors="coerce"
            ).abs().max()
        ),
    )
    .reset_index()
)


# =============================================================================
# CROSS-TARGET EVIDENCE
# =============================================================================

print("=" * 90)
print("BUILDING CROSS-TARGET EVIDENCE")
print("=" * 90)

cross2 = cross[cross["feature_std"].isin(common_features)].copy()

cross_sig = (
    cross2
    .groupby("feature_std")
    .agg(
        cross_target_rows=("feature_std", "size"),
        cross_target_pattern=(
            "cross_target_pattern",
            lambda x: "; ".join(sorted(set(x.astype(str))))
        ),
    )
    .reset_index()
)


# =============================================================================
# ML FEATURE STABILITY
# =============================================================================

print("=" * 90)
print("BUILDING ML STABILITY EVIDENCE")
print("=" * 90)

ml2 = ml[ml["feature_std"].isin(common_features)].copy()

ml_imp = numeric_series(
    ml2,
    [
        "mean_abs_permutation_importance",
        "permutation_importance",
        "mean_permutation_importance"
    ],
    0.0
)

ml_consistency = numeric_series(
    ml2,
    [
        "coefficient_sign_consistency",
        "sign_consistency"
    ],
    0.0
)

ml2["_importance"] = ml_imp
ml2["_consistency"] = ml_consistency

ml_sig = (
    ml2
    .groupby("feature_std")
    .agg(
        ml_mean_importance=("_importance", "mean"),
        ml_max_importance=("_importance", "max"),
        ml_mean_sign_consistency=("_consistency", "mean"),
    )
    .reset_index()
)


# =============================================================================
# LOSO EVIDENCE
# =============================================================================

print("=" * 90)
print("BUILDING LOSO EVIDENCE")
print("=" * 90)

loso2 = loso[loso["feature_std"].isin(common_features)].copy()

loso_stability = text_series(
    loso2,
    ["stability_class", "loso_stability", "classification"],
    "unknown"
)

loso2["_stable"] = (
    loso_stability
    .str.lower()
    .str.contains("high|stable", regex=True)
    .astype(int)
)

loso_abs = numeric_series(
    loso2,
    [
        "mean_abs_effect",
        "mean_abs_importance",
        "mean_effect"
    ],
    0.0
)

loso2["_loso_effect"] = loso_abs

loso_sig = (
    loso2
    .groupby("feature_std")
    .agg(
        loso_rows=("feature_std", "size"),
        loso_stable_count=("_stable", "sum"),
        loso_mean_effect=("_loso_effect", "mean"),
    )
    .reset_index()
)


# =============================================================================
# NULL VALIDATION
# =============================================================================

print("=" * 90)
print("BUILDING NULL VALIDATION EVIDENCE")
print("=" * 90)

null2 = null[null["feature_std"].isin(common_features)].copy()

null_p = numeric_series(
    null2,
    [
        "p_value",
        "p",
        "null_p",
        "p_fdr"
    ],
    1.0
)

null2["_null_p"] = null_p

null_sig = (
    null2
    .groupby("feature_std")
    .agg(
        null_rows=("feature_std", "size"),
        null_min_p=("_null_p", "min"),
        null_significant=(
            "_null_p",
            lambda x: int((x < 0.05).sum())
        ),
    )
    .reset_index()
)


# =============================================================================
# R MIXED-EFFECTS
# =============================================================================

print("=" * 90)
print("BUILDING R MIXED-EFFECTS EVIDENCE")
print("=" * 90)

r2 = r[r["feature_std"].isin(common_features)].copy()

r_p = numeric_series(
    r2,
    [
        "p_fdr",
        "fdr_p",
        "p_adjusted",
        "p"
    ],
    1.0
)

r2["_r_p"] = r_p

r_sig = (
    r2
    .groupby("feature_std")
    .agg(
        r_rows=("feature_std", "size"),
        r_min_p=("_r_p", "min"),
        r_fdr_significant=(
            "_r_p",
            lambda x: int((x < 0.05).sum())
        ),
    )
    .reset_index()
)


# =============================================================================
# MERGE ALL LAYERS
# =============================================================================

print("=" * 90)
print("MERGING ALL EVIDENCE LAYERS")
print("=" * 90)

final = pd.DataFrame({
    "feature": common_features
})

final = final.merge(
    pert_sig,
    on="feature_std",
    how="left"
) if "feature_std" in final.columns else final

# Rebuild clean merge key
final = pd.DataFrame({
    "feature_std": common_features
})

for table in [
    pert_sig,
    cross_sig,
    ml_sig,
    loso_sig,
    null_sig,
    r_sig
]:
    final = final.merge(
        table,
        on="feature_std",
        how="left"
    )


# =============================================================================
# EVIDENCE METRICS
# =============================================================================

final["perturbation_significant"] = (
    pd.to_numeric(
        final["perturbation_significant"],
        errors="coerce"
    ).fillna(0)
)

final["r_fdr_significant"] = (
    pd.to_numeric(
        final["r_fdr_significant"],
        errors="coerce"
    ).fillna(0)
)

final["null_significant"] = (
    pd.to_numeric(
        final["null_significant"],
        errors="coerce"
    ).fillna(0)
)

final["loso_stable_count"] = (
    pd.to_numeric(
        final["loso_stable_count"],
        errors="coerce"
    ).fillna(0)
)

final["ml_mean_importance"] = (
    pd.to_numeric(
        final["ml_mean_importance"],
        errors="coerce"
    ).fillna(0)
)

final["ml_mean_sign_consistency"] = (
    pd.to_numeric(
        final["ml_mean_sign_consistency"],
        errors="coerce"
    ).fillna(0)
)


# =============================================================================
# NORMALIZED SCORES
# =============================================================================

def minmax(series):
    series = pd.to_numeric(series, errors="coerce").fillna(0)

    if series.max() == series.min():
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (series - series.min()) / (
        series.max() - series.min()
    )


final["score_perturbation"] = minmax(
    final["perturbation_max_abs_d"]
)

final["score_ml"] = minmax(
    final["ml_mean_importance"]
)

final["score_loso"] = minmax(
    final["loso_mean_effect"]
)

final["score_consistency"] = (
    final["ml_mean_sign_consistency"]
    .clip(0, 1)
)

final["score_r"] = (
    final["r_fdr_significant"] > 0
).astype(float)

final["score_null"] = (
    final["null_significant"] > 0
).astype(float)


# =============================================================================
# FINAL EVIDENCE SCORE
# =============================================================================

final["final_evidence_score"] = (
    0.25 * final["score_perturbation"]
    + 0.20 * final["score_ml"]
    + 0.20 * final["score_loso"]
    + 0.15 * final["score_consistency"]
    + 0.10 * final["score_r"]
    + 0.10 * final["score_null"]
)


# =============================================================================
# EVIDENCE CLASS
# =============================================================================

def classify(row):

    score = row["final_evidence_score"]

    perturbation = row["perturbation_significant"] > 0
    ml_supported = row["ml_mean_importance"] > 0
    loso_supported = row["loso_stable_count"] > 0
    r_supported = row["r_fdr_significant"] > 0
    null_supported = row["null_significant"] > 0

    support_count = sum([
        perturbation,
        ml_supported,
        loso_supported,
        r_supported,
        null_supported
    ])

    if support_count >= 4 and score >= 0.60:
        return "strong_multilayer_evidence"

    if support_count >= 3 and score >= 0.40:
        return "moderate_multilayer_evidence"

    if support_count >= 2:
        return "limited_multilayer_evidence"

    return "weak_evidence"


final["final_evidence_class"] = final.apply(
    classify,
    axis=1
)


# =============================================================================
# PRIORITY RANK
# =============================================================================

class_priority = {
    "strong_multilayer_evidence": 4,
    "moderate_multilayer_evidence": 3,
    "limited_multilayer_evidence": 2,
    "weak_evidence": 1
}

final["evidence_class_priority"] = (
    final["final_evidence_class"]
    .map(class_priority)
    .fillna(0)
)

final = final.sort_values(
    [
        "evidence_class_priority",
        "final_evidence_score",
        "ml_mean_importance"
    ],
    ascending=[False, False, False]
).reset_index(drop=True)

final["final_rank"] = np.arange(1, len(final) + 1)


# =============================================================================
# CLEAN COLUMN
# =============================================================================

final = final.rename(
    columns={
        "feature_std": "feature"
    }
)


# =============================================================================
# SUMMARY
# =============================================================================

summary = pd.DataFrame({
    "metric": [
        "Features reconciled",
        "Strong multilayer evidence",
        "Moderate multilayer evidence",
        "Limited multilayer evidence",
        "Weak evidence",
        "R FDR significant features",
        "LOSO supported features",
        "ML supported features",
        "Null supported features"
    ],
    "value": [
        len(final),
        int(
            (final["final_evidence_class"]
             == "strong_multilayer_evidence").sum()
        ),
        int(
            (final["final_evidence_class"]
             == "moderate_multilayer_evidence").sum()
        ),
        int(
            (final["final_evidence_class"]
             == "limited_multilayer_evidence").sum()
        ),
        int(
            (final["final_evidence_class"]
             == "weak_evidence").sum()
        ),
        int(
            (final["r_fdr_significant"] > 0).sum()
        ),
        int(
            (final["loso_stable_count"] > 0).sum()
        ),
        int(
            (final["ml_mean_importance"] > 0).sum()
        ),
        int(
            (final["null_significant"] > 0).sum()
        )
    ]
})


# =============================================================================
# QC
# =============================================================================

numeric_cols = final.select_dtypes(
    include=[np.number]
).columns

nan_cells = int(
    final[numeric_cols].isna().sum().sum()
)

inf_cells = int(
    np.isinf(
        final[numeric_cols].to_numpy(
            dtype=float
        )
    ).sum()
)

duplicate_features = int(
    final["feature"].duplicated().sum()
)

qc = pd.DataFrame({
    "metric": [
        "Rows",
        "Unique features",
        "Numeric columns",
        "NaN numeric cells",
        "Inf numeric cells",
        "Duplicate features"
    ],
    "value": [
        len(final),
        final["feature"].nunique(),
        len(numeric_cols),
        nan_cells,
        inf_cells,
        duplicate_features
    ]
})


# =============================================================================
# SAVE
# =============================================================================

print("=" * 90)
print("SAVING FINAL EVIDENCE RECONCILIATION")
print("=" * 90)

final.to_csv(
    FINAL_FILE,
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)

qc.to_csv(
    QC_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =============================================================================
# DISPLAY TOP EVIDENCE
# =============================================================================

print()
print("=" * 90)
print("TOP FINAL SCIENTIFIC EVIDENCE")
print("=" * 90)

display_cols = [
    "final_rank",
    "feature",
    "final_evidence_score",
    "final_evidence_class",
    "perturbation_max_abs_d",
    "ml_mean_importance",
    "ml_mean_sign_consistency",
    "loso_mean_effect",
    "r_fdr_significant",
    "null_significant"
]

display_cols = [
    c for c in display_cols
    if c in final.columns
]

print(
    final[display_cols].head(30).to_string(
        index=False
    )
)

print()
print("=" * 90)
print("FINAL RECONCILIATION QC")
print("=" * 90)

print(f"Rows:                         {len(final)}")
print(f"Unique features:              {final['feature'].nunique()}")
print(f"NaN numeric cells:            {nan_cells}")
print(f"Inf numeric cells:            {inf_cells}")
print(f"Duplicate features:           {duplicate_features}")

print()
print("=" * 90)
print("SAVED")
print("=" * 90)

print(FINAL_FILE)
print(SUMMARY_FILE)
print(QC_FILE)

print()
print("=" * 90)
print("FINAL SCIENTIFIC EVIDENCE RECONCILIATION V1 COMPLETE")
print("=" * 90)
print("STATUS: PASS - MULTI-LAYER EVIDENCE RECONCILIATION CREATED")