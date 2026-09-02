from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# FINAL SCIENTIFIC EVIDENCE RANKING V2
# EEG PERTURBATION / WHAT-IF PROJECT
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

PERT = BASE / "features" / "perturbation_analysis"

SYNTHESIS = (
    PERT
    / "final_scientific_synthesis_v1"
    / "final_scientific_feature_synthesis_v1.csv"
)

MASTER = (
    PERT
    / "final_master_v1"
    / "final_perturbation_master_v1.csv"
)

CROSS_TARGET = (
    PERT
    / "cross_target_v1"
    / "cross_target_perturbation_comparison_v1.csv"
)

NETWORK = (
    PERT
    / "network_analysis_v1"
    / "scientific_perturbation_network_v1.csv"
)

ML_GENERALIZATION = (
    PERT
    / "ml_subject_generalization_v1"
    / "subject_generalization_v1_results.csv"
)

ML_STABILITY = (
    PERT
    / "ml_feature_stability_v1"
    / "ml_feature_stability_v1_results.csv"
)

OUT = PERT / "final_scientific_ranking_v2"
OUT.mkdir(parents=True, exist_ok=True)

RESULT = OUT / "final_scientific_evidence_ranking_v2.csv"
SUMMARY = OUT / "final_scientific_evidence_ranking_summary_v2.csv"
QC = OUT / "final_scientific_evidence_ranking_qc_v2.csv"


# =============================================================================
# HELPERS
# =============================================================================

def read_csv(path, name):
    print(f"\n{name}:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(f"Required input not found:\n{path}")

    df = pd.read_csv(path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_numeric(df, columns):
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_series(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)

    mn = s.min()
    mx = s.max()

    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - mn) / (mx - mn)


# =============================================================================
# LOAD INPUTS
# =============================================================================

print("=" * 90)
print("FINAL SCIENTIFIC EEG PERTURBATION / WHAT-IF EVIDENCE RANKING V2")
print("=" * 90)

print(f"Project root: {BASE}")

synthesis = read_csv(SYNTHESIS, "SCIENTIFIC SYNTHESIS INPUT")
master = read_csv(MASTER, "FINAL MASTER INPUT")
cross = read_csv(CROSS_TARGET, "CROSS-TARGET INPUT")
network = read_csv(NETWORK, "NETWORK INPUT")
ml_gen = read_csv(ML_GENERALIZATION, "ML GENERALIZATION INPUT")
ml_stability = read_csv(ML_STABILITY, "ML FEATURE STABILITY INPUT")


# =============================================================================
# IDENTIFY CORE COLUMNS
# =============================================================================

feature_col_syn = find_col(synthesis, ["feature"])
feature_col_master = find_col(master, ["feature"])
feature_col_cross = find_col(cross, ["feature"])
feature_col_network = find_col(network, ["feature"])
feature_col_stability = find_col(ml_stability, ["feature"])

if feature_col_syn is None:
    raise ValueError("Feature column not found in scientific synthesis.")

if feature_col_master is None:
    raise ValueError("Feature column not found in final master.")

if feature_col_cross is None:
    raise ValueError("Feature column not found in cross-target analysis.")

if feature_col_stability is None:
    raise ValueError("Feature column not found in ML stability results.")


# =============================================================================
# BASIC VALIDATION
# =============================================================================

print("\n" + "=" * 90)
print("INPUT FEATURE COVERAGE")
print("=" * 90)

syn_features = set(synthesis[feature_col_syn].dropna().astype(str))
master_features = set(master[feature_col_master].dropna().astype(str))
cross_features = set(cross[feature_col_cross].dropna().astype(str))
stability_features = set(
    ml_stability[feature_col_stability].dropna().astype(str)
)

common_features = (
    syn_features
    & master_features
    & cross_features
    & stability_features
)

print(f"Scientific synthesis features: {len(syn_features)}")
print(f"Master features:               {len(master_features)}")
print(f"Cross-target features:         {len(cross_features)}")
print(f"ML stability features:         {len(stability_features)}")
print(f"Common core features:          {len(common_features)}")


# =============================================================================
# PREPARE SYNTHESIS
# =============================================================================

syn = synthesis.copy()

syn[feature_col_syn] = syn[feature_col_syn].astype(str)

target_col = find_col(syn, ["target"])

if target_col is None:
    raise ValueError("Target column not found in scientific synthesis.")

# Prefer existing scientific statistics
numeric_candidates = [
    "cohen_d",
    "abs_cohen_d",
    "p_fdr",
    "final_priority_score_v1",
    "scientific_priority_score",
    "mean_difference",
    "direction_consistency",
    "positive_fraction",
    "negative_fraction",
]

syn = safe_numeric(syn, numeric_candidates)


# =============================================================================
# PREPARE MASTER
# =============================================================================

mst = master.copy()
mst[feature_col_master] = mst[feature_col_master].astype(str)

if "target" not in mst.columns:
    raise ValueError("Target column not found in final master.")

master_numeric = [
    "cohen_d",
    "abs_cohen_d",
    "p_fdr",
    "scientific_priority_score",
    "final_scientific_priority_score",
    "subject_count",
    "direction_consistency",
    "positive_fraction",
    "negative_fraction",
]

mst = safe_numeric(mst, master_numeric)


# =============================================================================
# PREPARE CROSS-TARGET
# =============================================================================

crt = cross.copy()
crt[feature_col_cross] = crt[feature_col_cross].astype(str)

cross_numeric = [
    "remember_effect_size",
    "correct_effect_size",
    "remember_abs_effect",
    "correct_abs_effect",
]

crt = safe_numeric(crt, cross_numeric)


# =============================================================================
# PREPARE ML STABILITY
# =============================================================================

mls = ml_stability.copy()
mls[feature_col_stability] = mls[feature_col_stability].astype(str)

stability_numeric = [
    "mean_abs_permutation_importance",
    "coefficient_sign_consistency",
    "mean_coefficient",
    "stability_score",
    "scientific_ml_rank",
]

mls = safe_numeric(mls, stability_numeric)


# =============================================================================
# BUILD FEATURE-LEVEL ML STABILITY SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING FEATURE-LEVEL ML STABILITY SUMMARY")
print("=" * 90)

if len(mls) > 0:

    ml_feature_summary = (
        mls.groupby(feature_col_stability, as_index=False)
        .agg(
            ml_mean_abs_permutation_importance=(
                "mean_abs_permutation_importance",
                "mean"
            ),
            ml_mean_sign_consistency=(
                "coefficient_sign_consistency",
                "mean"
            ),
            ml_mean_stability_score=(
                "stability_score",
                "mean"
            ),
            ml_best_rank=("scientific_ml_rank", "min"),
        )
    )

else:

    ml_feature_summary = pd.DataFrame(
        columns=[
            feature_col_stability,
            "ml_mean_abs_permutation_importance",
            "ml_mean_sign_consistency",
            "ml_mean_stability_score",
            "ml_best_rank",
        ]
    )


# =============================================================================
# BUILD CROSS-TARGET SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING CROSS-TARGET SUMMARY")
print("=" * 90)

cross_agg = {}

if "remember_abs_effect" in crt.columns:
    cross_agg["remember_abs_effect"] = ("remember_abs_effect", "max")

if "correct_abs_effect" in crt.columns:
    cross_agg["correct_abs_effect"] = ("correct_abs_effect", "max")

if "cross_target_pattern" in crt.columns:
    cross_agg["cross_target_pattern"] = (
        "cross_target_pattern",
        lambda x: x.dropna().iloc[0] if x.dropna().any() else "unknown"
    )

if "direction_relationship" in crt.columns:
    cross_agg["direction_relationship"] = (
        "direction_relationship",
        lambda x: x.dropna().iloc[0] if x.dropna().any() else "unknown"
    )

if "significance_pattern" in crt.columns:
    cross_agg["significance_pattern"] = (
        "significance_pattern",
        lambda x: x.dropna().iloc[0] if x.dropna().any() else "unknown"
    )

if cross_agg:

    cross_summary = (
        crt.groupby(feature_col_cross, as_index=False)
        .agg(**cross_agg)
    )

else:

    cross_summary = crt[[feature_col_cross]].drop_duplicates()


# =============================================================================
# MERGE SCIENTIFIC SYNTHESIS
# =============================================================================

print("\n" + "=" * 90)
print("MERGING SCIENTIFIC EVIDENCE")
print("=" * 90)

# Use synthesis as the primary scientific backbone.
# It normally contains target x feature rows.

result = syn.copy()

# Standardize feature name
result = result.rename(columns={feature_col_syn: "feature"})

# -------------------------------------------------------------------------
# Add ML stability
# -------------------------------------------------------------------------

ml_feature_summary = ml_feature_summary.rename(
    columns={feature_col_stability: "feature"}
)

result = result.merge(
    ml_feature_summary,
    on="feature",
    how="left"
)

# -------------------------------------------------------------------------
# Add cross-target evidence
# -------------------------------------------------------------------------

cross_summary = cross_summary.rename(
    columns={feature_col_cross: "feature"}
)

result = result.merge(
    cross_summary,
    on="feature",
    how="left"
)


# =============================================================================
# NORMALIZED EVIDENCE COMPONENTS
# =============================================================================

print("\n" + "=" * 90)
print("CALCULATING MULTI-LAYER EVIDENCE COMPONENTS")
print("=" * 90)

# Statistical strength
if "abs_cohen_d" in result.columns:
    result["evidence_statistical_effect"] = normalize_series(
        result["abs_cohen_d"]
    )
elif "cohen_d" in result.columns:
    result["evidence_statistical_effect"] = normalize_series(
        result["cohen_d"].abs()
    )
else:
    result["evidence_statistical_effect"] = 0.0


# Significance
if "p_fdr" in result.columns:

    p = pd.to_numeric(result["p_fdr"], errors="coerce")

    result["evidence_fdr_significance"] = np.where(
        p.notna(),
        (p <= 0.05).astype(float),
        0.0
    )

else:

    result["evidence_fdr_significance"] = 0.0


# Robustness
if "direction_consistency" in result.columns:

    result["evidence_direction_consistency"] = pd.to_numeric(
        result["direction_consistency"],
        errors="coerce"
    ).fillna(0)

else:

    result["evidence_direction_consistency"] = 0.0


# ML stability
if "ml_mean_sign_consistency" in result.columns:

    result["evidence_ml_stability"] = (
        pd.to_numeric(
            result["ml_mean_sign_consistency"],
            errors="coerce"
        )
        .fillna(0)
        .clip(0, 1)
    )

else:

    result["evidence_ml_stability"] = 0.0


# ML importance
if "ml_mean_abs_permutation_importance" in result.columns:

    result["evidence_ml_importance"] = normalize_series(
        result["ml_mean_abs_permutation_importance"]
    )

else:

    result["evidence_ml_importance"] = 0.0


# =============================================================================
# CROSS-TARGET EVIDENCE
# =============================================================================

def cross_target_score(row):

    pattern = str(row.get("cross_target_pattern", "")).lower()

    if "target_shared" in pattern:
        return 1.0

    if "target_preferential" in pattern:
        return 0.75

    if "weak_cross_target" in pattern:
        return 0.50

    if "target_dissociated" in pattern:
        return 0.25

    return 0.0


result["evidence_cross_target"] = result.apply(
    cross_target_score,
    axis=1
)


# =============================================================================
# SCIENTIFIC EVIDENCE SCORE
# =============================================================================

print("\nCalculating final multi-layer scientific evidence score...")

result["final_scientific_evidence_score"] = (
    0.25 * result["evidence_statistical_effect"]
    + 0.20 * result["evidence_fdr_significance"]
    + 0.15 * result["evidence_direction_consistency"]
    + 0.20 * result["evidence_ml_stability"]
    + 0.10 * result["evidence_ml_importance"]
    + 0.10 * result["evidence_cross_target"]
)


# =============================================================================
# EVIDENCE CLASSIFICATION
# =============================================================================

def classify_evidence(row):

    score = float(row["final_scientific_evidence_score"])

    fdr = float(row["evidence_fdr_significance"])
    direction = float(row["evidence_direction_consistency"])
    ml_stability = float(row["evidence_ml_stability"])
    ml_importance = float(row["evidence_ml_importance"])
    cross_score = float(row["evidence_cross_target"])

    if score >= 0.75:
        return "high_confidence_multilayer"

    if score >= 0.60 and fdr >= 1 and ml_stability >= 0.60:
        return "strong_multilayer_evidence"

    if score >= 0.45 and fdr >= 1:
        return "moderate_multilayer_evidence"

    if fdr >= 1 and direction >= 0.50:
        return "statistical_directional_evidence"

    if ml_stability >= 0.60 and ml_importance > 0:
        return "ml_supported_evidence"

    if cross_score >= 0.75:
        return "cross_target_supported"

    return "exploratory_evidence"


result["final_evidence_class_v2"] = result.apply(
    classify_evidence,
    axis=1
)


# =============================================================================
# TARGET-AWARE RANKING
# =============================================================================

result = result.sort_values(
    [
        "final_scientific_evidence_score",
        "evidence_fdr_significance",
        "evidence_statistical_effect",
    ],
    ascending=[False, False, False]
).reset_index(drop=True)

result["final_scientific_rank_v2"] = (
    result.groupby("target").cumcount() + 1
)


# =============================================================================
# FEATURE GLOBAL RANK
# =============================================================================

feature_global = (
    result.groupby("feature")["final_scientific_evidence_score"]
    .mean()
    .sort_values(ascending=False)
)

feature_rank_map = {
    feature: rank
    for rank, feature in enumerate(feature_global.index, start=1)
}

result["global_feature_rank_v2"] = result["feature"].map(
    feature_rank_map
)


# =============================================================================
# FINAL QC
# =============================================================================

print("\n" + "=" * 90)
print("FINAL SCIENTIFIC EVIDENCE RANKING QC")
print("=" * 90)

duplicate_target_feature = result.duplicated(
    subset=["target", "feature"]
).sum()

numeric_cols = result.select_dtypes(
    include=[np.number]
).columns

nan_numeric = int(
    result[numeric_cols].isna().sum().sum()
)

inf_numeric = int(
    np.isinf(
        result[numeric_cols].to_numpy(
            dtype=float
        )
    ).sum()
)

print(f"Rows:                         {len(result)}")
print(f"Unique features:              {result['feature'].nunique()}")
print(f"Targets:                      {result['target'].nunique()}")
print(f"FDR-significant rows:         {int(result['evidence_fdr_significance'].sum())}")
print(f"NaN numeric cells:            {nan_numeric}")
print(f"Inf numeric cells:            {inf_numeric}")
print(f"Duplicate target-feature:     {duplicate_target_feature}")


# =============================================================================
# EVIDENCE CLASS SUMMARY
# =============================================================================

class_summary = (
    result["final_evidence_class_v2"]
    .value_counts()
    .rename_axis("evidence_class")
    .reset_index(name="count")
)

print("\n" + "=" * 90)
print("FINAL EVIDENCE CLASS DISTRIBUTION")
print("=" * 90)

print(class_summary.to_string(index=False))


# =============================================================================
# TOP SCIENTIFIC EVIDENCE
# =============================================================================

print("\n" + "=" * 90)
print("TOP FINAL SCIENTIFIC EEG PERTURBATION EVIDENCE")
print("=" * 90)

display_cols = [
    "target",
    "feature",
    "final_scientific_evidence_score",
    "final_evidence_class_v2",
]

optional_display = [
    "cohen_d",
    "p_fdr",
    "direction_consistency",
    "ml_mean_sign_consistency",
    "ml_mean_abs_permutation_importance",
    "cross_target_pattern",
]

for c in optional_display:
    if c in result.columns:
        display_cols.append(c)

print(
    result[display_cols]
    .head(30)
    .to_string(index=False)
)


# =============================================================================
# SUMMARY TABLE
# =============================================================================

summary = (
    result.groupby("feature", as_index=False)
    .agg(
        mean_final_evidence_score=(
            "final_scientific_evidence_score",
            "mean"
        ),
        max_final_evidence_score=(
            "final_scientific_evidence_score",
            "max"
        ),
        targets_supported=(
            "target",
            "nunique"
        ),
        fdr_significant_targets=(
            "evidence_fdr_significance",
            "sum"
        ),
        mean_direction_consistency=(
            "evidence_direction_consistency",
            "mean"
        ),
        mean_ml_stability=(
            "evidence_ml_stability",
            "mean"
        ),
        mean_ml_importance=(
            "evidence_ml_importance",
            "mean"
        ),
        mean_cross_target_support=(
            "evidence_cross_target",
            "mean"
        ),
    )
)

summary["global_feature_rank_v2"] = summary[
    "feature"
].map(feature_rank_map)

summary = summary.sort_values(
    "mean_final_evidence_score",
    ascending=False
).reset_index(drop=True)


# =============================================================================
# QC TABLE
# =============================================================================

qc = pd.DataFrame(
    {
        "metric": [
            "rows",
            "unique_features",
            "targets",
            "fdr_significant_rows",
            "nan_numeric_cells",
            "inf_numeric_cells",
            "duplicate_target_feature",
            "high_confidence_multilayer",
            "strong_multilayer_evidence",
            "moderate_multilayer_evidence",
            "statistical_directional_evidence",
            "ml_supported_evidence",
            "cross_target_supported",
            "exploratory_evidence",
        ],
        "value": [
            len(result),
            result["feature"].nunique(),
            result["target"].nunique(),
            int(result["evidence_fdr_significance"].sum()),
            nan_numeric,
            inf_numeric,
            duplicate_target_feature,
            int(
                (
                    result["final_evidence_class_v2"]
                    == "high_confidence_multilayer"
                ).sum()
            ),
            int(
                (
                    result["final_evidence_class_v2"]
                    == "strong_multilayer_evidence"
                ).sum()
            ),
            int(
                (
                    result["final_evidence_class_v2"]
                    == "moderate_multilayer_evidence"
                ).sum()
            ),
            int(
                (
                    result["final_evidence_class_v2"]
                    == "statistical_directional_evidence"
                ).sum()
            ),
            int(
                (
                    result["final_evidence_class_v2"]
                    == "ml_supported_evidence"
                ).sum()
            ),
            int(
                (
                    result["final_evidence_class_v2"]
                    == "cross_target_supported"
                ).sum()
            ),
            int(
                (
                    result["final_evidence_class_v2"]
                    == "exploratory_evidence"
                ).sum()
            ),
        ],
    }
)


# =============================================================================
# SAVE
# =============================================================================

print("\n" + "=" * 90)
print("SAVING FINAL SCIENTIFIC EVIDENCE RANKING")
print("=" * 90)

result.to_csv(
    RESULT,
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    SUMMARY,
    index=False,
    encoding="utf-8-sig"
)

qc.to_csv(
    QC,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:")
print(RESULT)
print(SUMMARY)
print(QC)


# =============================================================================
# FINAL STATUS
# =============================================================================

if (
    nan_numeric == 0
    and inf_numeric == 0
    and duplicate_target_feature == 0
):
    status = "PASS"
else:
    status = "REVIEW_REQUIRED"

print("\n" + "=" * 90)
print("FINAL SCIENTIFIC EEG PERTURBATION / WHAT-IF EVIDENCE RANKING V2 COMPLETE")
print("=" * 90)

print(f"STATUS: {status} - FINAL MULTI-LAYER SCIENTIFIC RANKING CREATED")