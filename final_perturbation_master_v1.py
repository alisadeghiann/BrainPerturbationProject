# ============================================================
# FINAL PERTURBATION / WHAT-IF MASTER INTEGRATION V1
# ============================================================
# BrainPerturbationProject
#
# Purpose:
# Integrate all validated EEG perturbation / what-if evidence
# into one final scientific master table.
#
# IMPORTANT:
# - No target-derived feature generation
# - No new ML training
# - No modification of source files
# - EEG scientific features only
# - Keeps REMEMBER and CORRECT separate
# - Preserves statistical, robustness, interpretation,
#   network and cross-target evidence
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PROJECT ROOT
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

OUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_master_v1"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

STAT_FILE = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "statistical_v2"
    / "perturbation_statistical_results_v2.csv"
)

ROBUST_FILE = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "robustness_v6"
    / "subject_level_perturbation_robustness_v6.csv"
)

EVIDENCE_FILE = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_evidence_v2"
    / "final_scientific_perturbation_evidence_v2.csv"
)

INTERPRETATION_FILE = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "scientific_interpretation_v1"
    / "scientific_feature_interpretation_v1.csv"
)

NETWORK_FILE = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "network_analysis_v1"
    / "scientific_perturbation_network_v1.csv"
)

CROSS_TARGET_FILE = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "cross_target_v1"
    / "cross_target_perturbation_comparison_v1.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

MASTER_FILE = OUT_DIR / "final_perturbation_master_v1.csv"
SUMMARY_FILE = OUT_DIR / "final_perturbation_master_summary_v1.csv"
QC_FILE = OUT_DIR / "final_perturbation_master_qc_v1.csv"


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("FINAL EEG PERTURBATION / WHAT-IF MASTER INTEGRATION V1")
print("=" * 90)

print(f"Project root:\n{BASE}")


# ============================================================
# LOAD FUNCTION
# ============================================================

def load_csv(path, name):
    print("\n" + "=" * 90)
    print(f"LOADING: {name}")
    print("=" * 90)

    if not path.exists():
        print(f"NOT FOUND: {path}")
        return None

    df = pd.read_csv(path)

    print(f"File:    {path}")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# LOAD ALL VALIDATED SOURCES
# ============================================================

stat = load_csv(
    STAT_FILE,
    "STATISTICAL PERTURBATION V2"
)

robust = load_csv(
    ROBUST_FILE,
    "SUBJECT-LEVEL ROBUSTNESS V6"
)

evidence = load_csv(
    EVIDENCE_FILE,
    "FINAL SCIENTIFIC EVIDENCE V2"
)

interpretation = load_csv(
    INTERPRETATION_FILE,
    "SCIENTIFIC INTERPRETATION V1"
)

network = load_csv(
    NETWORK_FILE,
    "SCIENTIFIC NETWORK V1"
)

cross_target = load_csv(
    CROSS_TARGET_FILE,
    "CROSS-TARGET ANALYSIS V1"
)


# ============================================================
# REQUIRED SOURCES
# ============================================================

required_sources = {
    "statistical_v2": stat,
    "robustness_v6": robust,
    "final_evidence_v2": evidence,
    "interpretation_v1": interpretation,
    "network_v1": network,
    "cross_target_v1": cross_target,
}

missing = [
    name
    for name, df in required_sources.items()
    if df is None
]

if missing:
    print("\n" + "=" * 90)
    print("ERROR - REQUIRED INPUTS MISSING")
    print("=" * 90)

    for item in missing:
        print(item)

    raise FileNotFoundError(
        "Required perturbation source files are missing."
    )


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

def standardize_columns(df):

    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


stat = standardize_columns(stat)
robust = standardize_columns(robust)
evidence = standardize_columns(evidence)
interpretation = standardize_columns(interpretation)
network = standardize_columns(network)
cross_target = standardize_columns(cross_target)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 90)
print("SOURCE VALIDATION")
print("=" * 90)

for name, df in required_sources.items():

    print(
        f"{name:<25} "
        f"rows={len(df):>5,} "
        f"columns={len(df.columns):>3}"
    )


# ============================================================
# IDENTIFY EEG SCIENTIFIC FEATURES
# ============================================================

NON_EEG_COLUMNS = {
    "sfreq",
    "n_channels",
    "n_timepoints",
    "memory_cond",
}

def get_feature_column(df):

    if "feature" in df.columns:
        return "feature"

    if "scientific_feature" in df.columns:
        return "scientific_feature"

    raise KeyError(
        "No feature column found."
    )


def get_target_column(df):

    if "target" in df.columns:
        return "target"

    return None


# ============================================================
# FEATURE SETS
# ============================================================

stat_feature_col = get_feature_column(stat)
robust_feature_col = get_feature_column(robust)
evidence_feature_col = get_feature_column(evidence)

stat_features = set(
    stat[stat_feature_col].dropna().astype(str)
)

robust_features = set(
    robust[robust_feature_col].dropna().astype(str)
)

evidence_features = set(
    evidence[evidence_feature_col].dropna().astype(str)
)

print("\n" + "=" * 90)
print("FEATURE COVERAGE")
print("=" * 90)

print(
    f"Statistical features:       {len(stat_features)}"
)

print(
    f"Robustness features:        {len(robust_features)}"
)

print(
    f"Evidence features:          {len(evidence_features)}"
)

common_features = (
    stat_features
    & robust_features
    & evidence_features
)

print(
    f"Common core features:       {len(common_features)}"
)


# ============================================================
# REMOVE NON-EEG FEATURES
# ============================================================

def remove_non_eeg(df, feature_col):

    df = df.copy()

    df[feature_col] = (
        df[feature_col]
        .astype(str)
        .str.strip()
    )

    df = df[
        ~df[feature_col].isin(NON_EEG_COLUMNS)
    ].copy()

    return df


stat = remove_non_eeg(
    stat,
    stat_feature_col
)

robust = remove_non_eeg(
    robust,
    robust_feature_col
)

evidence = remove_non_eeg(
    evidence,
    evidence_feature_col
)


# ============================================================
# STATISTICAL CORE
# ============================================================

print("\n" + "=" * 90)
print("BUILDING STATISTICAL CORE")
print("=" * 90)

stat_core = stat.copy()

if "feature" not in stat_core.columns:
    stat_core = stat_core.rename(
        columns={
            stat_feature_col: "feature"
        }
    )

# Keep target-feature unique
if {"target", "feature"}.issubset(stat_core.columns):

    stat_core = (
        stat_core
        .drop_duplicates(
            subset=["target", "feature"],
            keep="first"
        )
        .copy()
    )

print(
    f"Statistical core rows: {len(stat_core):,}"
)


# ============================================================
# ROBUSTNESS CORE
# ============================================================

print("\n" + "=" * 90)
print("BUILDING SUBJECT-LEVEL ROBUSTNESS CORE")
print("=" * 90)

robust_core = robust.copy()

if "feature" not in robust_core.columns:
    robust_core = robust_core.rename(
        columns={
            robust_feature_col: "feature"
        }
    )

# Robustness V6 is feature-level and contains
# subject-level summary information.

if "target" in robust_core.columns:

    robust_merge_keys = [
        "target",
        "feature"
    ]

else:

    robust_merge_keys = [
        "feature"
    ]


robust_core = (
    robust_core
    .drop_duplicates(
        subset=robust_merge_keys,
        keep="first"
    )
    .copy()
)

# Prefix robustness columns
robust_rename = {}

for col in robust_core.columns:

    if col not in robust_merge_keys:

        robust_rename[col] = (
            "robust_" + col
        )

robust_core = robust_core.rename(
    columns=robust_rename
)

print(
    f"Robustness rows: {len(robust_core):,}"
)


# ============================================================
# FINAL EVIDENCE CORE
# ============================================================

print("\n" + "=" * 90)
print("BUILDING FINAL EVIDENCE CORE")
print("=" * 90)

evidence_core = evidence.copy()

if "feature" not in evidence_core.columns:

    evidence_core = evidence_core.rename(
        columns={
            evidence_feature_col: "feature"
        }
    )

if {"target", "feature"}.issubset(
    evidence_core.columns
):

    evidence_core = (
        evidence_core
        .drop_duplicates(
            subset=["target", "feature"],
            keep="first"
        )
        .copy()
    )

evidence_rename = {}

for col in evidence_core.columns:

    if col not in {"target", "feature"}:

        evidence_rename[col] = (
            "evidence_" + col
        )

evidence_core = evidence_core.rename(
    columns=evidence_rename
)

print(
    f"Evidence rows: {len(evidence_core):,}"
)


# ============================================================
# MERGE STATISTICAL + EVIDENCE
# ============================================================

print("\n" + "=" * 90)
print("MERGING STATISTICAL + FINAL EVIDENCE")
print("=" * 90)

master = stat_core.merge(
    evidence_core,
    on=["target", "feature"],
    how="left",
    suffixes=("", "_duplicate")
)


# ============================================================
# MERGE ROBUSTNESS
# ============================================================

print("\n" + "=" * 90)
print("MERGING SUBJECT-LEVEL ROBUSTNESS")
print("=" * 90)

if "target" in robust_core.columns:

    master = master.merge(
        robust_core,
        on=["target", "feature"],
        how="left",
        suffixes=("", "_robust_duplicate")
    )

else:

    master = master.merge(
        robust_core,
        on=["feature"],
        how="left",
        suffixes=("", "_robust_duplicate")
    )


# ============================================================
# REMOVE DUPLICATE COLUMNS
# ============================================================

duplicate_columns = [
    c
    for c in master.columns
    if c.endswith("_duplicate")
]

if duplicate_columns:

    master = master.drop(
        columns=duplicate_columns
    )


# ============================================================
# ADD NETWORK SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("ADDING NETWORK / REGION EVIDENCE")
print("=" * 90)

if "feature" in network.columns:

    network_copy = network.copy()

    network_copy = (
        network_copy
        .drop_duplicates(
            subset=["feature"],
            keep="first"
        )
        .copy()
    )

    network_columns = [
        c
        for c in network_copy.columns
        if c != "feature"
    ]

    network_rename = {
        c: "network_" + c
        for c in network_columns
    }

    network_copy = network_copy.rename(
        columns=network_rename
    )

    master = master.merge(
        network_copy,
        on="feature",
        how="left"
    )


# ============================================================
# ADD CROSS-TARGET INFORMATION
# ============================================================

print("\n" + "=" * 90)
print("ADDING CROSS-TARGET EVIDENCE")
print("=" * 90)

if "feature" in cross_target.columns:

    cross_copy = cross_target.copy()

    cross_copy = (
        cross_copy
        .drop_duplicates(
            subset=["feature"],
            keep="first"
        )
        .copy()
    )

    cross_columns = [
        c
        for c in cross_copy.columns
        if c != "feature"
    ]

    cross_rename = {
        c: "cross_target_" + c
        for c in cross_columns
    }

    cross_copy = cross_copy.rename(
        columns=cross_rename
    )

    master = master.merge(
        cross_copy,
        on="feature",
        how="left"
    )


# ============================================================
# ADD INTERPRETATION
# ============================================================

print("\n" + "=" * 90)
print("ADDING SCIENTIFIC INTERPRETATION")
print("=" * 90)

if "feature" in interpretation.columns:

    interpretation_copy = interpretation.copy()

    interpretation_copy = (
        interpretation_copy
        .drop_duplicates(
            subset=["feature"],
            keep="first"
        )
        .copy()
    )

    interpretation_columns = [
        c
        for c in interpretation_copy.columns
        if c != "feature"
    ]

    interpretation_rename = {
        c: "interpretation_" + c
        for c in interpretation_columns
    }

    interpretation_copy = (
        interpretation_copy.rename(
            columns=interpretation_rename
        )
    )

    master = master.merge(
        interpretation_copy,
        on="feature",
        how="left"
    )


# ============================================================
# FINAL TARGET / FEATURE VALIDATION
# ============================================================

print("\n" + "=" * 90)
print("FINAL TARGET / FEATURE VALIDATION")
print("=" * 90)

if "target" not in master.columns:
    raise KeyError(
        "Final master table has no target column."
    )

if "feature" not in master.columns:
    raise KeyError(
        "Final master table has no feature column."
    )

master["target"] = (
    master["target"]
    .astype(str)
    .str.lower()
    .str.strip()
)

master["feature"] = (
    master["feature"]
    .astype(str)
    .str.strip()
)

allowed_targets = {
    "remember",
    "correct",
}

unexpected_targets = (
    set(master["target"].unique())
    - allowed_targets
)

if unexpected_targets:

    raise ValueError(
        f"Unexpected targets detected: "
        f"{unexpected_targets}"
    )


# ============================================================
# DUPLICATE TARGET-FEATURE CHECK
# ============================================================

duplicate_target_feature = (
    master
    .duplicated(
        subset=["target", "feature"],
        keep=False
    )
)

duplicate_count = int(
    duplicate_target_feature.sum()
)

print(
    f"Duplicate target-feature rows: "
    f"{duplicate_count}"
)

if duplicate_count > 0:

    master = (
        master
        .drop_duplicates(
            subset=["target", "feature"],
            keep="first"
        )
        .copy()
    )


# ============================================================
# NUMERIC QC
# ============================================================

numeric_columns = (
    master
    .select_dtypes(
        include=[np.number]
    )
    .columns
)

nan_numeric = int(
    master[numeric_columns]
    .isna()
    .sum()
    .sum()
)

inf_numeric = int(
    np.isinf(
        master[numeric_columns]
        .to_numpy()
    ).sum()
)

print("\n" + "=" * 90)
print("FINAL NUMERIC QC")
print("=" * 90)

print(
    f"Numeric columns: {len(numeric_columns)}"
)

print(
    f"NaN numeric cells: {nan_numeric}"
)

print(
    f"Inf numeric cells: {inf_numeric}"
)


# ============================================================
# CREATE SCIENTIFIC PRIORITY
# ============================================================

print("\n" + "=" * 90)
print("CREATING FINAL SCIENTIFIC PRIORITY")
print("=" * 90)

def find_column(df, candidates):

    for c in candidates:

        if c in df.columns:
            return c

    return None


priority_column = find_column(
    master,
    [
        "evidence_final_scientific_priority_score",
        "scientific_priority_score",
        "final_scientific_priority_score",
    ]
)

if priority_column is not None:

    master["final_master_priority"] = pd.to_numeric(
        master[priority_column],
        errors="coerce"
    )

else:

    master["final_master_priority"] = np.nan


# ============================================================
# FDR STATUS
# ============================================================

fdr_column = find_column(
    master,
    [
        "p_fdr",
        "evidence_p_fdr",
        "evidence_fdr_p_value",
    ]
)

if fdr_column is not None:

    master["final_fdr_significant"] = (
        pd.to_numeric(
            master[fdr_column],
            errors="coerce"
        )
        < 0.05
    )

else:

    master["final_fdr_significant"] = False


# ============================================================
# SUBJECT ROBUSTNESS STATUS
# ============================================================

robustness_columns = [
    c
    for c in master.columns
    if c.startswith("robust_")
]

if robustness_columns:

    master["subject_robustness_available"] = (
        master[robustness_columns]
        .notna()
        .any(axis=1)
    )

else:

    master["subject_robustness_available"] = False


# ============================================================
# SCIENTIFIC EVIDENCE CLASS
# ============================================================

evidence_class_column = find_column(
    master,
    [
        "evidence_scientific_evidence_class",
        "scientific_evidence_class",
    ]
)

if evidence_class_column is not None:

    master["final_evidence_class"] = (
        master[evidence_class_column]
        .astype(str)
    )

else:

    master["final_evidence_class"] = (
        "not_available"
    )


# ============================================================
# SORT FINAL MASTER TABLE
# ============================================================

master = master.sort_values(
    by=[
        "target",
        "final_fdr_significant",
        "final_master_priority",
    ],
    ascending=[
        True,
        False,
        False,
    ],
    na_position="last"
).reset_index(drop=True)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("FINAL PERTURBATION MASTER SUMMARY")
print("=" * 90)

print(
    f"Final rows:                 {len(master):,}"
)

print(
    f"Unique features:            "
    f"{master['feature'].nunique():,}"
)

print(
    f"Targets:                    "
    f"{master['target'].nunique():,}"
)

print(
    f"FDR significant rows:       "
    f"{int(master['final_fdr_significant'].sum()):,}"
)

print(
    f"Subject robustness rows:    "
    f"{int(master['subject_robustness_available'].sum()):,}"
)

print(
    f"NaN numeric cells:          {nan_numeric:,}"
)

print(
    f"Inf numeric cells:          {inf_numeric:,}"
)

print(
    f"Duplicate target-feature:   "
    f"{master.duplicated(['target','feature']).sum()}"
)


# ============================================================
# TARGET SUMMARY
# ============================================================

target_summary = (
    master
    .groupby("target")
    .agg(
        rows=("feature", "count"),
        unique_features=("feature", "nunique"),
        fdr_significant=(
            "final_fdr_significant",
            "sum"
        ),
        robustness_available=(
            "subject_robustness_available",
            "sum"
        ),
    )
    .reset_index()
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

feature_summary = (
    master
    .groupby("feature")
    .agg(
        target_count=("target", "nunique"),
        fdr_significant_count=(
            "final_fdr_significant",
            "sum"
        ),
        max_priority=(
            "final_master_priority",
            "max"
        ),
    )
    .reset_index()
)


# ============================================================
# SAVE MASTER
# ============================================================

master.to_csv(
    MASTER_FILE,
    index=False
)

target_summary.to_csv(
    SUMMARY_FILE,
    index=False
)

# ============================================================
# QC TABLE
# ============================================================

qc = pd.DataFrame(
    [
        {
            "metric": "final_rows",
            "value": len(master)
        },
        {
            "metric": "unique_features",
            "value": master["feature"].nunique()
        },
        {
            "metric": "targets",
            "value": master["target"].nunique()
        },
        {
            "metric": "fdr_significant_rows",
            "value": int(
                master["final_fdr_significant"].sum()
            )
        },
        {
            "metric": "subject_robustness_available",
            "value": int(
                master[
                    "subject_robustness_available"
                ].sum()
            )
        },
        {
            "metric": "nan_numeric_cells",
            "value": nan_numeric
        },
        {
            "metric": "inf_numeric_cells",
            "value": inf_numeric
        },
        {
            "metric": "duplicate_target_feature",
            "value": int(
                master
                .duplicated(
                    ["target", "feature"]
                )
                .sum()
            )
        },
        {
            "metric": "statistical_features",
            "value": len(stat_features)
        },
        {
            "metric": "robustness_features",
            "value": len(robust_features)
        },
        {
            "metric": "evidence_features",
            "value": len(evidence_features)
        },
    ]
)

qc.to_csv(
    QC_FILE,
    index=False
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 90)
print("SAVED")
print("=" * 90)

print(MASTER_FILE)
print(SUMMARY_FILE)
print(QC_FILE)

print("\n" + "=" * 90)
print("FINAL EEG PERTURBATION / WHAT-IF MASTER INTEGRATION V1 COMPLETE")
print("=" * 90)

if (
    len(master) == 110
    and master["target"].nunique() == 2
    and master["feature"].nunique() == 55
    and master.duplicated(
        ["target", "feature"]
    ).sum() == 0
    and inf_numeric == 0
):

    print(
        "STATUS: PASS - FINAL PERTURBATION MASTER CREATED"
    )

else:

    print(
        "STATUS: REVIEW_REQUIRED - "
        "CHECK FINAL MASTER QC"
    )