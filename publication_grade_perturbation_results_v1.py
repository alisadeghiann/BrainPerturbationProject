from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# PUBLICATION-GRADE PERTURBATION / WHAT-IF RESULTS V1
# =============================================================================
# Purpose:
# Build a publication-oriented evidence profile from the already validated
# perturbation analysis layers.
#
# IMPORTANT:
# This script does NOT retrain ML models.
# This script does NOT alter statistical results.
# This script does NOT replace subject-aware ML results.
# It only integrates and summarizes already validated outputs.
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

PERT = BASE / "features" / "perturbation_analysis"

RANKING = (
    PERT
    / "final_scientific_ranking_v2"
    / "final_scientific_evidence_ranking_v2.csv"
)

RANKING_SUMMARY = (
    PERT
    / "final_scientific_ranking_v2"
    / "final_scientific_evidence_ranking_summary_v2.csv"
)

SYNTHESIS = (
    PERT
    / "final_scientific_synthesis_v1"
    / "final_scientific_feature_synthesis_v1.csv"
)

NETWORK = (
    PERT
    / "network_analysis_v1"
    / "scientific_perturbation_network_v1.csv"
)

CROSS_TARGET = (
    PERT
    / "cross_target_v1"
    / "cross_target_perturbation_comparison_v1.csv"
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

OUT = PERT / "publication_grade_v1"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_RESULTS = (
    OUT / "publication_grade_feature_results_v1.csv"
)

TARGET_RESULTS = (
    OUT / "publication_grade_target_results_v1.csv"
)

FREQUENCY_RESULTS = (
    OUT / "publication_grade_frequency_results_v1.csv"
)

REGION_RESULTS = (
    OUT / "publication_grade_region_results_v1.csv"
)

CROSS_TARGET_RESULTS = (
    OUT / "publication_grade_cross_target_results_v1.csv"
)

TOP_EVIDENCE = (
    OUT / "publication_grade_top_evidence_v1.csv"
)

QC = (
    OUT / "publication_grade_results_qc_v1.csv"
)


# =============================================================================
# HELPERS
# =============================================================================

def load_csv(path, label):
    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired input file was not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df)}")

    return df


def numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )
    return df


def first_valid(series, default=np.nan):
    x = series.dropna()

    if len(x) == 0:
        return default

    return x.iloc[0]


def normalize(s):
    s = pd.to_numeric(s, errors="coerce")

    if s.notna().sum() == 0:
        return pd.Series(
            np.zeros(len(s)),
            index=s.index
        )

    mn = s.min()
    mx = s.max()

    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(
            np.zeros(len(s)),
            index=s.index
        )

    return (s - mn) / (mx - mn)


# =============================================================================
# LOAD DATA
# =============================================================================

ranking = load_csv(
    RANKING,
    "FINAL SCIENTIFIC EVIDENCE RANKING V2"
)

ranking_summary = load_csv(
    RANKING_SUMMARY,
    "FINAL SCIENTIFIC FEATURE SUMMARY V2"
)

synthesis = load_csv(
    SYNTHESIS,
    "FINAL SCIENTIFIC SYNTHESIS"
)

network = load_csv(
    NETWORK,
    "SCIENTIFIC NETWORK ANALYSIS"
)

cross = load_csv(
    CROSS_TARGET,
    "CROSS-TARGET ANALYSIS"
)

ml_generalization = load_csv(
    ML_GENERALIZATION,
    "SUBJECT-AWARE ML GENERALIZATION"
)

ml_stability = load_csv(
    ML_STABILITY,
    "ML FEATURE STABILITY"
)


# =============================================================================
# BASIC VALIDATION
# =============================================================================

print("\n" + "=" * 90)
print("INPUT VALIDATION")
print("=" * 90)

required_ranking = [
    "target",
    "feature",
    "final_scientific_evidence_score",
    "final_evidence_class_v2",
]

for col in required_ranking:
    if col not in ranking.columns:
        raise ValueError(
            f"Required column missing from ranking file: {col}"
        )

if "feature" not in synthesis.columns:
    raise ValueError(
        "Feature column missing from scientific synthesis."
    )

if "feature" not in cross.columns:
    raise ValueError(
        "Feature column missing from cross-target analysis."
    )

if "feature" not in ml_stability.columns:
    raise ValueError(
        "Feature column missing from ML stability."
    )


# =============================================================================
# NUMERIC CONVERSION
# =============================================================================

ranking = numeric(
    ranking,
    [
        "cohen_d",
        "abs_cohen_d",
        "p_fdr",
        "mean_difference",
        "direction_consistency",
        "positive_fraction",
        "negative_fraction",
        "final_scientific_evidence_score",
        "evidence_statistical_effect",
        "evidence_fdr_significance",
        "evidence_direction_consistency",
        "evidence_ml_stability",
        "evidence_ml_importance",
        "evidence_cross_target",
    ]
)

synthesis = numeric(
    synthesis,
    [
        "cohen_d",
        "abs_cohen_d",
        "p_fdr",
        "mean_difference",
        "final_priority_score_v1",
    ]
)

cross = numeric(
    cross,
    [
        "remember_effect_size",
        "correct_effect_size",
        "remember_abs_effect",
        "correct_abs_effect",
    ]
)

ml_stability = numeric(
    ml_stability,
    [
        "mean_abs_permutation_importance",
        "coefficient_sign_consistency",
        "mean_coefficient",
        "stability_score",
        "scientific_ml_rank",
    ]
)

ml_generalization = numeric(
    ml_generalization,
    [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
    ]
)


# =============================================================================
# STANDARDIZE FEATURE NAMES
# =============================================================================

ranking["feature"] = ranking["feature"].astype(str)
synthesis["feature"] = synthesis["feature"].astype(str)
cross["feature"] = cross["feature"].astype(str)
ml_stability["feature"] = ml_stability["feature"].astype(str)


# =============================================================================
# ML STABILITY FEATURE SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING ML FEATURE STABILITY PROFILE")
print("=" * 90)

ml_profile = (
    ml_stability
    .groupby("feature", as_index=False)
    .agg(
        ml_mean_permutation_importance=(
            "mean_abs_permutation_importance",
            "mean"
        ),
        ml_max_permutation_importance=(
            "mean_abs_permutation_importance",
            "max"
        ),
        ml_mean_sign_consistency=(
            "coefficient_sign_consistency",
            "mean"
        ),
        ml_mean_stability_score=(
            "stability_score",
            "mean"
        ),
        ml_best_rank=(
            "scientific_ml_rank",
            "min"
        ),
    )
)


# =============================================================================
# CROSS-TARGET FEATURE PROFILE
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING CROSS-TARGET PROFILE")
print("=" * 90)

cross_agg = {
    "remember_abs_effect": (
        "remember_abs_effect",
        "max"
    ),
    "correct_abs_effect": (
        "correct_abs_effect",
        "max"
    ),
}

if "cross_target_pattern" in cross.columns:
    cross_agg["cross_target_pattern"] = (
        "cross_target_pattern",
        first_valid
    )

if "significance_pattern" in cross.columns:
    cross_agg["significance_pattern"] = (
        "significance_pattern",
        first_valid
    )

if "direction_relationship" in cross.columns:
    cross_agg["direction_relationship"] = (
        "direction_relationship",
        first_valid
    )

cross_profile = (
    cross
    .groupby("feature", as_index=False)
    .agg(**cross_agg)
)


# =============================================================================
# NETWORK PROFILE
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING NETWORK PROFILE")
print("=" * 90)

network_profile = None

if "feature" in network.columns:

    network["feature"] = network["feature"].astype(str)

    network_numeric = [
        c
        for c in [
            "abs_effect",
            "effect_size",
            "frequency_score",
            "region_score",
            "direction_consistency",
        ]
        if c in network.columns
    ]

    network = numeric(
        network,
        network_numeric
    )

    if network_numeric:

        agg_dict = {}

        for col in network_numeric:
            agg_dict[
                f"network_{col}_mean"
            ] = (col, "mean")

            agg_dict[
                f"network_{col}_max"
            ] = (col, "max")

        network_profile = (
            network
            .groupby("feature", as_index=False)
            .agg(**agg_dict)
        )

    else:

        network_profile = (
            network[["feature"]]
            .drop_duplicates()
        )

else:

    network_profile = pd.DataFrame(
        {"feature": []}
    )


# =============================================================================
# FEATURE-LEVEL SCIENTIFIC PROFILE
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING PUBLICATION-GRADE FEATURE PROFILES")
print("=" * 90)

feature_results = ranking.copy()

# -------------------------------------------------------------------------
# Add ML stability
# -------------------------------------------------------------------------

feature_results = feature_results.merge(
    ml_profile,
    on="feature",
    how="left"
)

# -------------------------------------------------------------------------
# Add cross-target evidence
# -------------------------------------------------------------------------

feature_results = feature_results.merge(
    cross_profile,
    on="feature",
    how="left"
)

# -------------------------------------------------------------------------
# Add network evidence
# -------------------------------------------------------------------------

feature_results = feature_results.merge(
    network_profile,
    on="feature",
    how="left"
)


# =============================================================================
# PUBLICATION INTERPRETATION FLAGS
# =============================================================================

def publication_flag(row):

    score = row.get(
        "final_scientific_evidence_score",
        np.nan
    )

    fdr = row.get(
        "evidence_fdr_significance",
        0
    )

    direction = row.get(
        "evidence_direction_consistency",
        0
    )

    ml_stability_value = row.get(
        "ml_mean_sign_consistency",
        np.nan
    )

    if pd.isna(score):
        return "not_evaluable"

    if (
        score >= 0.60
        and fdr >= 1
        and not pd.isna(ml_stability_value)
        and ml_stability_value >= 0.60
    ):
        return "high_priority_candidate"

    if (
        score >= 0.50
        and fdr >= 1
        and direction >= 0.50
    ):
        return "moderate_priority_candidate"

    if (
        fdr >= 1
        and direction >= 0.50
    ):
        return "statistically_supported"

    if (
        not pd.isna(ml_stability_value)
        and ml_stability_value >= 0.60
    ):
        return "ml_supported_exploratory"

    return "exploratory"


feature_results["publication_priority"] = (
    feature_results.apply(
        publication_flag,
        axis=1
    )
)


# =============================================================================
# DIRECTION LABEL
# =============================================================================

def direction_label(value):

    if pd.isna(value):
        return "unknown"

    if value > 0:
        return "positive"

    if value < 0:
        return "negative"

    return "zero"


if "cohen_d" in feature_results.columns:

    feature_results["effect_direction"] = (
        feature_results["cohen_d"]
        .apply(direction_label)
    )


# =============================================================================
# EFFECT MAGNITUDE LABEL
# =============================================================================

def effect_magnitude(d):

    if pd.isna(d):
        return "unknown"

    d = abs(float(d))

    if d < 0.10:
        return "very_small"

    if d < 0.20:
        return "small"

    if d < 0.50:
        return "moderate"

    if d < 0.80:
        return "large"

    return "very_large"


if "cohen_d" in feature_results.columns:

    feature_results["publication_effect_magnitude"] = (
        feature_results["cohen_d"]
        .apply(effect_magnitude)
    )


# =============================================================================
# TARGET-LEVEL SYNTHESIS
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING TARGET-LEVEL RESULTS")
print("=" * 90)

target_agg = {}

if "final_scientific_evidence_score" in feature_results.columns:

    target_agg["mean_evidence_score"] = (
        "final_scientific_evidence_score",
        "mean"
    )

    target_agg["max_evidence_score"] = (
        "final_scientific_evidence_score",
        "max"
    )

if "evidence_fdr_significance" in feature_results.columns:

    target_agg["fdr_significant_features"] = (
        "evidence_fdr_significance",
        "sum"
    )

if "cohen_d" in feature_results.columns:

    target_agg["mean_abs_cohen_d"] = (
        "cohen_d",
        lambda x: np.nanmean(np.abs(x))
    )

if "publication_priority" in feature_results.columns:

    target_agg["high_priority_features"] = (
        "publication_priority",
        lambda x: (
            x == "high_priority_candidate"
        ).sum()
    )

target_results = (
    feature_results
    .groupby("target", as_index=False)
    .agg(**target_agg)
)


# =============================================================================
# FREQUENCY EXTRACTION
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING FREQUENCY-LEVEL RESULTS")
print("=" * 90)

frequency_rows = []

for _, row in feature_results.iterrows():

    feature = str(row["feature"])

    parts = feature.split("_")

    frequency = None

    known = [
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma",
    ]

    for p in parts:

        if p in known:
            frequency = p
            break

    if frequency is None:
        frequency = "derived"

    frequency_rows.append(
        {
            "target": row.get("target", np.nan),
            "feature": feature,
            "frequency": frequency,
            "evidence_score": row.get(
                "final_scientific_evidence_score",
                np.nan
            ),
            "cohen_d": row.get(
                "cohen_d",
                np.nan
            ),
            "p_fdr": row.get(
                "p_fdr",
                np.nan
            ),
            "publication_priority": row.get(
                "publication_priority",
                "unknown"
            ),
        }
    )

frequency_results = pd.DataFrame(
    frequency_rows
)

if len(frequency_results) > 0:

    frequency_summary = (
        frequency_results
        .groupby(
            ["target", "frequency"],
            as_index=False
        )
        .agg(
            feature_count=(
                "feature",
                "count"
            ),
            mean_evidence_score=(
                "evidence_score",
                "mean"
            ),
            max_evidence_score=(
                "evidence_score",
                "max"
            ),
            mean_abs_cohen_d=(
                "cohen_d",
                lambda x: np.nanmean(
                    np.abs(x)
                )
            ),
            fdr_significant_features=(
                "p_fdr",
                lambda x: (
                    pd.to_numeric(
                        x,
                        errors="coerce"
                    ) <= 0.05
                ).sum()
            ),
        )
    )

else:

    frequency_summary = pd.DataFrame()


# =============================================================================
# REGION EXTRACTION
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING REGION-LEVEL RESULTS")
print("=" * 90)

regions = [
    "frontal",
    "central",
    "parietal",
    "occipital",
    "temporal",
    "frontoparietal",
    "global",
]

region_rows = []

for _, row in feature_results.iterrows():

    feature = str(row["feature"])

    region = "other"

    for r in regions:

        if r in feature:
            region = r
            break

    region_rows.append(
        {
            "target": row.get("target", np.nan),
            "feature": feature,
            "region": region,
            "evidence_score": row.get(
                "final_scientific_evidence_score",
                np.nan
            ),
            "cohen_d": row.get(
                "cohen_d",
                np.nan
            ),
            "p_fdr": row.get(
                "p_fdr",
                np.nan
            ),
        }
    )

region_detail = pd.DataFrame(
    region_rows
)

if len(region_detail) > 0:

    region_results = (
        region_detail
        .groupby(
            ["target", "region"],
            as_index=False
        )
        .agg(
            feature_count=(
                "feature",
                "count"
            ),
            mean_evidence_score=(
                "evidence_score",
                "mean"
            ),
            max_evidence_score=(
                "evidence_score",
                "max"
            ),
            mean_abs_cohen_d=(
                "cohen_d",
                lambda x: np.nanmean(
                    np.abs(x)
                )
            ),
            fdr_significant_features=(
                "p_fdr",
                lambda x: (
                    pd.to_numeric(
                        x,
                        errors="coerce"
                    ) <= 0.05
                ).sum()
            ),
        )
    )

else:

    region_results = pd.DataFrame()


# =============================================================================
# CROSS-TARGET PUBLICATION TABLE
# =============================================================================

print("\n" + "=" * 90)
print("BUILDING CROSS-TARGET PUBLICATION RESULTS")
print("=" * 90)

cross_target_results = cross_profile.copy()

if len(cross_target_results) > 0:

    # Add scientific score averaged across targets
    score_lookup = (
        feature_results
        .groupby("feature")[
            "final_scientific_evidence_score"
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "final_scientific_evidence_score":
                    "mean_scientific_evidence_score"
            }
        )
    )

    cross_target_results = (
        cross_target_results
        .merge(
            score_lookup,
            on="feature",
            how="left"
        )
    )

    cross_target_results = (
        cross_target_results
        .sort_values(
            "mean_scientific_evidence_score",
            ascending=False
        )
        .reset_index(drop=True)
    )


# =============================================================================
# TOP EVIDENCE
# =============================================================================

print("\n" + "=" * 90)
print("SELECTING TOP PUBLICATION-GRADE EVIDENCE")
print("=" * 90)

top_columns = [
    "target",
    "feature",
    "final_scientific_evidence_score",
    "final_evidence_class_v2",
    "publication_priority",
]

optional = [
    "cohen_d",
    "p_fdr",
    "direction_consistency",
    "ml_mean_sign_consistency",
    "ml_mean_permutation_importance",
    "cross_target_pattern",
    "significance_pattern",
    "direction_relationship",
]

for col in optional:

    if col in feature_results.columns:
        top_columns.append(col)

top_evidence = (
    feature_results[
        top_columns
    ]
    .sort_values(
        "final_scientific_evidence_score",
        ascending=False
    )
    .head(30)
    .reset_index(drop=True)
)

print(
    top_evidence.to_string(
        index=False
    )
)


# =============================================================================
# PUBLICATION CANDIDATE TABLE
# =============================================================================

publication_candidates = feature_results[
    feature_results["publication_priority"]
    .isin(
        [
            "high_priority_candidate",
            "moderate_priority_candidate",
        ]
    )
].copy()

publication_candidates = (
    publication_candidates
    .sort_values(
        "final_scientific_evidence_score",
        ascending=False
    )
    .reset_index(drop=True)
)


# =============================================================================
# FINAL QC
# =============================================================================

print("\n" + "=" * 90)
print("FINAL PUBLICATION-GRADE RESULTS QC")
print("=" * 90)

duplicate_target_feature = (
    feature_results
    .duplicated(
        subset=["target", "feature"]
    )
    .sum()
)

numeric_cols = (
    feature_results
    .select_dtypes(
        include=[np.number]
    )
    .columns
)

nan_numeric = int(
    feature_results[
        numeric_cols
    ]
    .isna()
    .sum()
    .sum()
)

inf_numeric = int(
    np.isinf(
        feature_results[
            numeric_cols
        ]
        .to_numpy(
            dtype=float
        )
    ).sum()
)

print(
    f"Final rows:                    {len(feature_results)}"
)

print(
    f"Unique features:               "
    f"{feature_results['feature'].nunique()}"
)

print(
    f"Targets:                       "
    f"{feature_results['target'].nunique()}"
)

print(
    f"FDR-significant rows:          "
    f"{int(feature_results['evidence_fdr_significance'].sum())}"
)

print(
    f"Publication candidates:        "
    f"{len(publication_candidates)}"
)

print(
    f"NaN numeric cells:             "
    f"{nan_numeric}"
)

print(
    f"Inf numeric cells:             "
    f"{inf_numeric}"
)

print(
    f"Duplicate target-feature:      "
    f"{duplicate_target_feature}"
)


# =============================================================================
# QC SUMMARY
# =============================================================================

qc_rows = [
    ["final_rows", len(feature_results)],
    [
        "unique_features",
        feature_results["feature"].nunique()
    ],
    [
        "targets",
        feature_results["target"].nunique()
    ],
    [
        "fdr_significant_rows",
        int(
            feature_results[
                "evidence_fdr_significance"
            ].sum()
        )
    ],
    [
        "publication_candidates",
        len(publication_candidates)
    ],
    [
        "high_priority_candidates",
        int(
            (
                feature_results[
                    "publication_priority"
                ]
                == "high_priority_candidate"
            ).sum()
        )
    ],
    [
        "moderate_priority_candidates",
        int(
            (
                feature_results[
                    "publication_priority"
                ]
                == "moderate_priority_candidate"
            ).sum()
        )
    ],
    [
        "statistically_supported",
        int(
            (
                feature_results[
                    "publication_priority"
                ]
                == "statistically_supported"
            ).sum()
        )
    ],
    [
        "ml_supported_exploratory",
        int(
            (
                feature_results[
                    "publication_priority"
                ]
                == "ml_supported_exploratory"
            ).sum()
        )
    ],
    [
        "exploratory",
        int(
            (
                feature_results[
                    "publication_priority"
                ]
                == "exploratory"
            ).sum()
        )
    ],
    [
        "nan_numeric_cells",
        nan_numeric
    ],
    [
        "inf_numeric_cells",
        inf_numeric
    ],
    [
        "duplicate_target_feature",
        duplicate_target_feature
    ],
]

qc = pd.DataFrame(
    qc_rows,
    columns=["metric", "value"]
)


# =============================================================================
# SAVE ALL RESULTS
# =============================================================================

print("\n" + "=" * 90)
print("SAVING PUBLICATION-GRADE RESULTS")
print("=" * 90)

feature_results.to_csv(
    FEATURE_RESULTS,
    index=False,
    encoding="utf-8-sig"
)

target_results.to_csv(
    TARGET_RESULTS,
    index=False,
    encoding="utf-8-sig"
)

frequency_results.to_csv(
    FREQUENCY_RESULTS,
    index=False,
    encoding="utf-8-sig"
)

region_results.to_csv(
    REGION_RESULTS,
    index=False,
    encoding="utf-8-sig"
)

cross_target_results.to_csv(
    CROSS_TARGET_RESULTS,
    index=False,
    encoding="utf-8-sig"
)

top_evidence.to_csv(
    TOP_EVIDENCE,
    index=False,
    encoding="utf-8-sig"
)

qc.to_csv(
    QC,
    index=False,
    encoding="utf-8-sig"
)


# =============================================================================
# PRINT OUTPUTS
# =============================================================================

print("\nSaved:")

print(FEATURE_RESULTS)
print(TARGET_RESULTS)
print(FREQUENCY_RESULTS)
print(REGION_RESULTS)
print(CROSS_TARGET_RESULTS)
print(TOP_EVIDENCE)
print(QC)


# =============================================================================
# FINAL STATUS
# =============================================================================

if (
    nan_numeric == 0
    and inf_numeric == 0
    and duplicate_target_feature == 0
):

    print("\n" + "=" * 90)
    print(
        "PUBLICATION-GRADE PERTURBATION / WHAT-IF RESULTS V1 COMPLETE"
    )
    print("=" * 90)

    print(
        "STATUS: PASS - PUBLICATION-GRADE RESULTS PROFILE CREATED"
    )

else:

    print("\n" + "=" * 90)
    print(
        "PUBLICATION-GRADE PERTURBATION / WHAT-IF RESULTS V1 COMPLETE"
    )
    print("=" * 90)

    print(
        "STATUS: REVIEW_REQUIRED"
    )