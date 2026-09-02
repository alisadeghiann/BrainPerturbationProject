# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# FINAL SCIENTIFIC PERTURBATION / WHAT-IF EVIDENCE INTEGRATION V1.1
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_evidence_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "final_perturbation_evidence_v1.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "final_perturbation_evidence_summary_v1.csv"
)

QC_FILE = (
    OUTPUT_DIR
    / "final_perturbation_evidence_qc_v1.csv"
)


print("=" * 90)
print("FINAL SCIENTIFIC PERTURBATION / WHAT-IF EVIDENCE INTEGRATION V1.1")
print("=" * 90)
print(f"Project root: {BASE}")


# =============================================================================
# HELPER
# =============================================================================

def find_csv(required_columns, preferred_terms=None, exclude_terms=None):

    required_columns = set(required_columns)
    preferred_terms = preferred_terms or []
    exclude_terms = exclude_terms or []

    candidates = []

    for path in BASE.rglob("*.csv"):

        if OUTPUT_DIR in path.parents:
            continue

        name = path.name.lower()

        if any(term.lower() in name for term in exclude_terms):
            continue

        try:
            header = pd.read_csv(path, nrows=0)
            columns = set(header.columns)

            if required_columns.issubset(columns):

                score = 0

                for term in preferred_terms:
                    if term.lower() in name:
                        score += 1

                candidates.append((score, path))

        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (x[0], x[1].stat().st_mtime),
        reverse=True
    )

    return candidates[0][1]


# =============================================================================
# 1. STATISTICAL RESULTS
# =============================================================================

print("\n" + "=" * 90)
print("SEARCHING FOR STATISTICAL PERTURBATION RESULTS")
print("=" * 90)

stats_file = find_csv(
    required_columns={
        "target",
        "feature",
        "mean_difference",
        "cohen_d",
        "p_value",
        "p_fdr",
    },
    preferred_terms=[
        "perturbation_statistical_results_v2",
        "statistical_results",
    ],
)

if stats_file is None:
    raise FileNotFoundError(
        "Could not find perturbation statistical results."
    )

print("Statistical input:")
print(stats_file)


# =============================================================================
# 2. SUBJECT-LEVEL ROBUSTNESS
# =============================================================================

print("\n" + "=" * 90)
print("SEARCHING FOR SUBJECT-LEVEL ROBUSTNESS")
print("=" * 90)

robustness_file = find_csv(
    required_columns={
        "feature",
        "remember_subject_count",
        "remember_subject_effect_mean",
        "remember_subject_effect_median",
        "remember_subject_effect_std",
        "remember_direction_consistency",
        "remember_direction",
        "remember_robustness_class",
        "correct_subject_count",
        "correct_subject_effect_mean",
        "correct_subject_effect_median",
        "correct_subject_effect_std",
        "correct_direction_consistency",
        "correct_direction",
        "correct_robustness_class",
        "overall_robustness_class",
    },
    preferred_terms=[
        "subject_level_perturbation_robustness_v6",
    ],
)

if robustness_file is None:
    raise FileNotFoundError(
        "Could not find V6 subject-level robustness results."
    )

print("Robustness input:")
print(robustness_file)


# =============================================================================
# 3. FINAL SCIENTIFIC RANKING
# =============================================================================

print("\n" + "=" * 90)
print("SEARCHING FOR FINAL SCIENTIFIC RANKING")
print("=" * 90)

ranking_file = find_csv(
    required_columns={
        "target",
        "feature",
    },
    preferred_terms=[
        "final_scientific_perturbation_ranking_v3",
    ],
)

if ranking_file is None:
    raise FileNotFoundError(
        "Could not find final scientific perturbation ranking."
    )

print("Ranking input:")
print(ranking_file)


# =============================================================================
# 4. LOAD
# =============================================================================

stats = pd.read_csv(stats_file)
robustness = pd.read_csv(robustness_file)
ranking = pd.read_csv(ranking_file)

print("\n" + "=" * 90)
print("INPUT DATA SUMMARY")
print("=" * 90)

print(f"Statistical rows:       {len(stats):,}")
print(f"Robustness rows:        {len(robustness):,}")
print(f"Ranking rows:           {len(ranking):,}")

print(f"Statistical features:   {stats['feature'].nunique():,}")
print(f"Robustness features:    {robustness['feature'].nunique():,}")
print(f"Ranking features:       {ranking['feature'].nunique():,}")


# =============================================================================
# 5. DUPLICATE CONTROL
# =============================================================================

stats = stats.drop_duplicates(
    subset=["target", "feature"],
    keep="first"
).copy()

ranking = ranking.drop_duplicates(
    subset=["target", "feature"],
    keep="first"
).copy()

robustness = robustness.drop_duplicates(
    subset=["feature"],
    keep="first"
).copy()


# =============================================================================
# 6. FEATURE COVERAGE
# =============================================================================

stats_features = set(
    stats["feature"].astype(str)
)

robust_features = set(
    robustness["feature"].astype(str)
)

ranking_features = set(
    ranking["feature"].astype(str)
)

common_features = (
    stats_features
    & robust_features
)

print("\n" + "=" * 90)
print("FEATURE COVERAGE")
print("=" * 90)

print(f"Statistical features:        {len(stats_features):,}")
print(f"Robustness features:         {len(robust_features):,}")
print(f"Ranking features:             {len(ranking_features):,}")
print(f"Common statistical/robust:   {len(common_features):,}")

print(
    f"Statistics-only features:    "
    f"{len(stats_features - robust_features):,}"
)

print(
    f"Robustness-only features:    "
    f"{len(robust_features - stats_features):,}"
)


# =============================================================================
# 7. MERGE STATISTICS + ROBUSTNESS
# =============================================================================

print("\n" + "=" * 90)
print("MERGING STATISTICAL + SUBJECT-LEVEL ROBUSTNESS")
print("=" * 90)

evidence = stats.merge(
    robustness,
    on="feature",
    how="left",
    suffixes=("", "_robustness"),
    validate="many_to_one",
)


# =============================================================================
# 8. MERGE FINAL RANKING
# =============================================================================

ranking_columns = [
    "target",
    "feature",
]

optional_ranking_columns = [
    "scientific_rank",
    "scientific_priority_score",
    "robustness_class",
    "direction_agreement",
    "abs_cohen_d",
    "fdr_significant",
]

available_ranking_columns = [
    c for c in optional_ranking_columns
    if c in ranking.columns
]

ranking_subset = ranking[
    ranking_columns + available_ranking_columns
].copy()

ranking_subset = ranking_subset.drop_duplicates(
    subset=["target", "feature"]
)

evidence = evidence.merge(
    ranking_subset,
    on=["target", "feature"],
    how="left",
    suffixes=("", "_ranking"),
)


# =============================================================================
# 9. SUBJECT ROBUSTNESS AVAILABILITY
# =============================================================================

evidence["subject_robustness_available"] = (
    evidence["remember_subject_count"].notna()
    & evidence["correct_subject_count"].notna()
)


# =============================================================================
# 10. TARGET-SPECIFIC INFORMATION
# =============================================================================

def get_target_consistency(row):

    if row["target"] == "remember":
        return row["remember_direction_consistency"]

    if row["target"] == "correct":
        return row["correct_direction_consistency"]

    return np.nan


def get_target_direction(row):

    if row["target"] == "remember":
        return row["remember_direction"]

    if row["target"] == "correct":
        return row["correct_direction"]

    return "not_available"


def get_target_robustness(row):

    if row["target"] == "remember":
        return row["remember_robustness_class"]

    if row["target"] == "correct":
        return row["correct_robustness_class"]

    return "not_available"


def get_target_subject_count(row):

    if row["target"] == "remember":
        return row["remember_subject_count"]

    if row["target"] == "correct":
        return row["correct_subject_count"]

    return np.nan


evidence["target_subject_count"] = evidence.apply(
    get_target_subject_count,
    axis=1
)

evidence["target_direction_consistency"] = evidence.apply(
    get_target_consistency,
    axis=1
)

evidence["target_direction"] = evidence.apply(
    get_target_direction,
    axis=1
)

evidence["target_robustness_class"] = evidence.apply(
    get_target_robustness,
    axis=1
)


# =============================================================================
# 11. FDR + EFFECT SIZE
# =============================================================================

evidence["p_value"] = pd.to_numeric(
    evidence["p_value"],
    errors="coerce"
)

evidence["p_fdr"] = pd.to_numeric(
    evidence["p_fdr"],
    errors="coerce"
)

evidence["cohen_d"] = pd.to_numeric(
    evidence["cohen_d"],
    errors="coerce"
)

evidence["fdr_significant"] = (
    evidence["p_fdr"] < 0.05
)

evidence["abs_cohen_d"] = (
    evidence["cohen_d"].abs()
)


# =============================================================================
# 12. EVIDENCE CLASS
# =============================================================================

def classify_evidence(row):

    fdr = bool(row["fdr_significant"])

    consistency = row["target_direction_consistency"]

    if pd.isna(consistency):
        return "statistical_only"

    if fdr and consistency >= 0.80:
        return "strong_replicated"

    if fdr and consistency >= 0.65:
        return "moderate_replicated"

    if fdr and consistency >= 0.55:
        return "weak_replicated"

    if fdr:
        return "significant_but_inconsistent"

    if consistency >= 0.80:
        return "robust_but_not_fdr_significant"

    if consistency >= 0.65:
        return "moderately_consistent"

    return "weak_or_unresolved"


evidence["scientific_evidence_class"] = evidence.apply(
    classify_evidence,
    axis=1
)


# =============================================================================
# 13. DIRECTION AGREEMENT
# =============================================================================

def direction_agreement(row):

    statistical_direction = str(
        row.get("direction", "")
    ).lower()

    subject_direction = str(
        row["target_direction"]
    ).lower()

    if statistical_direction not in {
        "positive",
        "negative"
    }:
        return "not_available"

    if subject_direction not in {
        "positive",
        "negative"
    }:
        return "not_available"

    if statistical_direction == subject_direction:
        return "agreement"

    return "disagreement"


evidence["subject_vs_statistical_direction"] = evidence.apply(
    direction_agreement,
    axis=1
)


# =============================================================================
# 14. SCIENTIFIC PRIORITY SCORE
# =============================================================================

def safe_abs(value):

    try:
        return abs(float(value))
    except Exception:
        return 0.0


def calculate_priority(row):

    score = 0.0

    # FDR significance
    if bool(row["fdr_significant"]):
        score += 3.0

    # Effect size
    d = safe_abs(row["cohen_d"])

    if d >= 0.50:
        score += 3.0
    elif d >= 0.30:
        score += 2.0
    elif d >= 0.10:
        score += 1.0

    # Subject consistency
    consistency = row["target_direction_consistency"]

    if not pd.isna(consistency):

        if consistency >= 0.80:
            score += 3.0
        elif consistency >= 0.65:
            score += 2.0
        elif consistency >= 0.55:
            score += 1.0

    # Direction agreement
    if row["subject_vs_statistical_direction"] == "agreement":
        score += 2.0

    elif row["subject_vs_statistical_direction"] == "disagreement":
        score -= 1.0

    return score


evidence["final_scientific_priority_score"] = evidence.apply(
    calculate_priority,
    axis=1
)


# =============================================================================
# 15. TARGET-SPECIFIC RANK
# =============================================================================

evidence["final_scientific_rank"] = (
    evidence
    .groupby("target")[
        "final_scientific_priority_score"
    ]
    .rank(
        ascending=False,
        method="dense"
    )
)


# =============================================================================
# 16. SORT
# =============================================================================

evidence = evidence.sort_values(
    [
        "target",
        "final_scientific_priority_score",
        "abs_cohen_d",
        "target_direction_consistency",
    ],
    ascending=[
        True,
        False,
        False,
        False,
    ]
).reset_index(drop=True)


# =============================================================================
# 17. QC
# =============================================================================

numeric_columns = evidence.select_dtypes(
    include=[np.number]
).columns

nan_numeric = int(
    evidence[numeric_columns]
    .isna()
    .sum()
    .sum()
)

inf_numeric = int(
    np.isinf(
        evidence[numeric_columns]
        .to_numpy(
            dtype=float,
            na_value=np.nan
        )
    ).sum()
)

duplicate_keys = int(
    evidence.duplicated(
        subset=["target", "feature"]
    ).sum()
)

fdr_count = int(
    evidence["fdr_significant"].sum()
)

robust_count = int(
    evidence["subject_robustness_available"].sum()
)


# =============================================================================
# 18. SUMMARY
# =============================================================================

summary_rows = []

for target in sorted(
    evidence["target"].dropna().unique()
):

    sub = evidence[
        evidence["target"] == target
    ].copy()

    summary_rows.append({
        "target": target,

        "features":
            sub["feature"].nunique(),

        "fdr_significant":
            int(sub["fdr_significant"].sum()),

        "robustness_available":
            int(sub["subject_robustness_available"].sum()),

        "strong_replicated":
            int(
                (
                    sub["scientific_evidence_class"]
                    == "strong_replicated"
                ).sum()
            ),

        "moderate_replicated":
            int(
                (
                    sub["scientific_evidence_class"]
                    == "moderate_replicated"
                ).sum()
            ),

        "weak_replicated":
            int(
                (
                    sub["scientific_evidence_class"]
                    == "weak_replicated"
                ).sum()
            ),

        "significant_but_inconsistent":
            int(
                (
                    sub["scientific_evidence_class"]
                    == "significant_but_inconsistent"
                ).sum()
            ),

        "robust_but_not_fdr_significant":
            int(
                (
                    sub["scientific_evidence_class"]
                    == "robust_but_not_fdr_significant"
                ).sum()
            ),

        "weak_or_unresolved":
            int(
                (
                    sub["scientific_evidence_class"]
                    == "weak_or_unresolved"
                ).sum()
            ),
    })


summary = pd.DataFrame(summary_rows)


# =============================================================================
# 19. QC TABLE
# =============================================================================

qc = pd.DataFrame({
    "metric": [
        "statistical_input_rows",
        "robustness_input_rows",
        "ranking_input_rows",
        "statistical_features",
        "robustness_features",
        "ranking_features",
        "common_features",
        "final_rows",
        "final_features",
        "targets",
        "fdr_significant_rows",
        "robustness_available_rows",
        "nan_numeric_values",
        "inf_numeric_values",
        "duplicate_target_feature",
    ],

    "value": [
        len(stats),
        len(robustness),
        len(ranking),
        len(stats_features),
        len(robust_features),
        len(ranking_features),
        len(common_features),
        len(evidence),
        evidence["feature"].nunique(),
        evidence["target"].nunique(),
        fdr_count,
        robust_count,
        nan_numeric,
        inf_numeric,
        duplicate_keys,
    ]
})


# =============================================================================
# 20. PRINT FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 90)
print("FINAL SCIENTIFIC EVIDENCE SUMMARY")
print("=" * 90)

print(f"Final rows:                  {len(evidence):,}")
print(f"Features:                    {evidence['feature'].nunique():,}")
print(f"Targets:                     {evidence['target'].nunique():,}")
print(f"FDR significant:             {fdr_count:,}")
print(f"Robustness available:        {robust_count:,}")
print(f"NaN numeric values:          {nan_numeric:,}")
print(f"Inf numeric values:          {inf_numeric:,}")
print(f"Duplicate target-feature:    {duplicate_keys:,}")


# =============================================================================
# 21. TOP RESULTS
# =============================================================================

print("\n" + "=" * 90)
print("TOP SCIENTIFIC PERTURBATION / WHAT-IF EVIDENCE")
print("=" * 90)

display_columns = [
    "target",
    "feature",
    "mean_difference",
    "cohen_d",
    "p_fdr",
    "target_direction_consistency",
    "target_direction",
    "target_robustness_class",
    "subject_vs_statistical_direction",
    "scientific_evidence_class",
    "final_scientific_priority_score",
]

display_columns = [
    c for c in display_columns
    if c in evidence.columns
]

print(
    evidence[
        display_columns
    ]
    .head(30)
    .to_string(index=False)
)


# =============================================================================
# 22. SAVE
# =============================================================================

evidence.to_csv(
    OUTPUT_FILE,
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
# 23. FINAL STATUS
# =============================================================================

print("\n" + "=" * 90)
print("FINAL SCIENTIFIC PERTURBATION / WHAT-IF EVIDENCE INTEGRATION V1.1 COMPLETE")
print("=" * 90)

print("\nSaved:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)
print(QC_FILE)

if (
    nan_numeric == 0
    and inf_numeric == 0
    and duplicate_keys == 0
    and robust_count > 0
):
    print(
        "\nSTATUS: PASS - FINAL PERTURBATION EVIDENCE INTEGRATION CREATED"
    )
else:
    print("\nSTATUS: REVIEW_REQUIRED")