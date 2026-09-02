# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =============================================================================
# ML TARGET LEAKAGE AUDIT
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

DATASET = (
    BASE
    / "features"
    / "ml_ready"
    / "ml_ready_dataset.csv"
)

SPLIT = (
    BASE
    / "features"
    / "ml_ready"
    / "split"
    / "subject_level_split.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "ml_results"
    / "leakage_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_AUDIT_FILE = OUTPUT_DIR / "feature_target_leakage_audit.csv"
SUMMARY_FILE = OUTPUT_DIR / "target_leakage_audit_summary.csv"


# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("ML TARGET LEAKAGE AUDIT")
print("=" * 80)

df = pd.read_csv(DATASET)
split_df = pd.read_csv(SPLIT)

print()
print(f"Dataset rows: {len(df):,}")
print(f"Subjects:      {df['subject'].nunique()}")
print(f"Features file: {DATASET}")


# =============================================================================
# NORMALIZE
# =============================================================================

df["subject"] = df["subject"].astype(str).str.strip()

split_df["subject"] = split_df["subject"].astype(str).str.strip()
split_df["split"] = (
    split_df["split"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# =============================================================================
# TARGET NORMALIZATION
# =============================================================================

def normalize_binary(series):

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce"
        )

    return (
        series
        .astype(str)
        .str.lower()
        .str.strip()
        .map({
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0
        })
    )


df["target_remember"] = normalize_binary(
    df["target_remember"]
)

df["target_correct"] = normalize_binary(
    df["target_correct"]
)


# =============================================================================
# SPLIT MERGE
# =============================================================================

subject_split = (
    split_df[["subject", "split"]]
    .drop_duplicates()
)

if (
    subject_split
    .groupby("subject")["split"]
    .nunique()
    .gt(1)
    .any()
):
    raise RuntimeError(
        "A subject appears in multiple splits."
    )

df = df.drop(columns=["split"], errors="ignore")

df = df.merge(
    subject_split,
    on="subject",
    how="left",
    validate="many_to_one"
)

if df["split"].isna().any():
    raise RuntimeError(
        "Some subjects do not have split assignments."
    )


# =============================================================================
# DEFINE METADATA
# =============================================================================

metadata = {
    "file",
    "subject",
    "run",
    "epoch",
    "trial",
    "split",

    "event_code",
    "event_name",
    "event_source",

    "feedback",
    "response_type",
    "complete_trial",

    "memory_cond",
    "remember_count",
    "ignore_count",
    "remember_letters",
    "ignore_letters",

    "probe_type",
    "probe_letter",

    "behavior_outcome",
    "behavior_label",
    "alignment_status",

    "target_label",
    "target_remember",
    "target_correct",
}


# =============================================================================
# NUMERIC FEATURES
# =============================================================================

candidate_features = [
    c for c in df.columns
    if c not in metadata
]

numeric_features = [
    c for c in candidate_features
    if pd.api.types.is_numeric_dtype(df[c])
]

if not numeric_features:
    raise RuntimeError(
        "No numeric feature columns found."
    )

print()
print("=" * 80)
print("FEATURE INVENTORY")
print("=" * 80)

print(f"Candidate features: {len(candidate_features)}")
print(f"Numeric features:   {len(numeric_features)}")

print()
print(numeric_features)


# =============================================================================
# BASIC TARGET CORRELATION AUDIT
# =============================================================================

print()
print("=" * 80)
print("UNIVARIATE TARGET LEAKAGE AUDIT")
print("=" * 80)

audit_rows = []

targets = {
    "remember": "target_remember",
    "correct": "target_correct",
}

for target_name, target_col in targets.items():

    print()
    print(f"TARGET: {target_name.upper()}")
    print("-" * 80)

    valid = df[target_col].notna()

    work = df.loc[valid].copy()

    y = work[target_col].astype(int)

    for feature in numeric_features:

        x = pd.to_numeric(
            work[feature],
            errors="coerce"
        )

        valid_xy = x.notna() & y.notna()

        x_valid = x.loc[valid_xy]
        y_valid = y.loc[valid_xy]

        n = len(x_valid)

        if n < 20 or y_valid.nunique() < 2:

            audit_rows.append({
                "target": target_name,
                "feature": feature,
                "n": n,
                "correlation": np.nan,
                "auc": np.nan,
                "flag": "INSUFFICIENT_DATA"
            })

            continue


        # -------------------------------------------------------------
        # CORRELATION
        # -------------------------------------------------------------

        try:
            correlation = np.corrcoef(
                x_valid.to_numpy(dtype=float),
                y_valid.to_numpy(dtype=float)
            )[0, 1]
        except Exception:
            correlation = np.nan


        # -------------------------------------------------------------
        # UNIVARIATE AUC
        # -------------------------------------------------------------

        try:

            auc_forward = roc_auc_score(
                y_valid,
                x_valid
            )

            auc_reverse = roc_auc_score(
                y_valid,
                -x_valid
            )

            auc = max(
                auc_forward,
                auc_reverse
            )

        except Exception:
            auc = np.nan


        # -------------------------------------------------------------
        # FLAG
        # -------------------------------------------------------------

        if np.isfinite(auc) and auc >= 0.995:
            flag = "VERY_HIGH_AUC"

        elif np.isfinite(auc) and auc >= 0.95:
            flag = "HIGH_AUC"

        elif (
            np.isfinite(correlation)
            and abs(correlation) >= 0.95
        ):
            flag = "VERY_HIGH_CORRELATION"

        elif (
            np.isfinite(correlation)
            and abs(correlation) >= 0.80
        ):
            flag = "HIGH_CORRELATION"

        else:
            flag = "NORMAL"


        audit_rows.append({
            "target": target_name,
            "feature": feature,
            "n": n,
            "correlation": correlation,
            "auc": auc,
            "flag": flag
        })


audit_df = pd.DataFrame(audit_rows)


# =============================================================================
# DISPLAY TOP SUSPICIOUS FEATURES
# =============================================================================

for target_name in targets:

    subset = audit_df[
        audit_df["target"] == target_name
    ].copy()

    subset = subset.sort_values(
        "auc",
        ascending=False
    )

    print()
    print(f"TOP FEATURES FOR {target_name.upper()}")
    print("-" * 80)

    print(
        subset[
            [
                "feature",
                "correlation",
                "auc",
                "flag"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


# =============================================================================
# DIRECT DUPLICATE / TARGET-LIKE FEATURE CHECK
# =============================================================================

print()
print("=" * 80)
print("TARGET-LIKE FEATURE NAME AUDIT")
print("=" * 80)

target_keywords = [
    "target",
    "remember",
    "correct",
    "incorrect",
    "ignore",
    "probe",
    "behavior",
    "feedback",
    "response",
    "label",
    "accuracy",
]

suspicious_name_features = []

for feature in numeric_features:

    name_lower = feature.lower()

    hits = [
        k for k in target_keywords
        if k in name_lower
    ]

    if hits:

        suspicious_name_features.append({
            "feature": feature,
            "keywords": "|".join(hits)
        })


if suspicious_name_features:

    name_df = pd.DataFrame(
        suspicious_name_features
    )

    print(
        name_df.to_string(index=False)
    )

else:

    print(
        "No target-like keywords found in numeric feature names."
    )


# =============================================================================
# EXACT TARGET EQUALITY CHECK
# =============================================================================

print()
print("=" * 80)
print("EXACT TARGET EQUALITY CHECK")
print("=" * 80)

equality_rows = []

for target_name, target_col in targets.items():

    y = df[target_col]

    for feature in numeric_features:

        x = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        valid = x.notna() & y.notna()

        if valid.sum() == 0:
            continue

        x_valid = x.loc[valid]
        y_valid = y.loc[valid]

        equal_direct = np.array_equal(
            x_valid.to_numpy(),
            y_valid.to_numpy()
        )

        equal_inverse = np.array_equal(
            x_valid.to_numpy(),
            1 - y_valid.to_numpy()
        )

        if equal_direct or equal_inverse:

            equality_rows.append({
                "target": target_name,
                "feature": feature,
                "exact_direct_match": equal_direct,
                "exact_inverse_match": equal_inverse
            })


if equality_rows:

    print(
        pd.DataFrame(
            equality_rows
        ).to_string(index=False)
    )

else:

    print(
        "No exact target-equivalent numeric feature found."
    )


# =============================================================================
# SUBJECT-LEVEL CROSS-VALIDATION SANITY CHECK
# =============================================================================

print()
print("=" * 80)
print("GROUPED CROSS-VALIDATION SANITY CHECK")
print("=" * 80)

# This is NOT the final ML model.
# It is only used to determine whether the suspiciously perfect
# performance persists when every subject is kept inside one fold.

group_cv = GroupKFold(n_splits=5)

cv_rows = []

for target_name, target_col in targets.items():

    print()
    print(f"TARGET: {target_name.upper()}")

    valid = df[target_col].notna()

    work = df.loc[valid].copy()

    X_cv = (
        work[numeric_features]
        .replace([np.inf, -np.inf], np.nan)
    )

    y_cv = work[target_col].astype(int)

    groups = work["subject"]

    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        group_cv.split(
            X_cv,
            y_cv,
            groups
        ),
        start=1
    ):

        X_train = X_cv.iloc[train_idx]
        X_test = X_cv.iloc[test_idx]

        y_train = y_cv.iloc[train_idx]
        y_test = y_cv.iloc[test_idx]

        if y_train.nunique() < 2:
            continue

        if y_test.nunique() < 2:
            continue

        model = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ])

        model.fit(
            X_train,
            y_train
        )

        prob = model.predict_proba(
            X_test
        )[:, 1]

        auc = roc_auc_score(
            y_test,
            prob
        )

        test_subjects = sorted(
            work.iloc[test_idx]["subject"]
            .unique()
        )

        fold_scores.append(auc)

        cv_rows.append({
            "target": target_name,
            "fold": fold,
            "test_subject_count": len(test_subjects),
            "test_subjects": "|".join(test_subjects),
            "roc_auc": auc
        })

        print(
            f"Fold {fold}: ROC-AUC = {auc:.4f}"
        )


# =============================================================================
# SAVE AUDIT
# =============================================================================

audit_df.to_csv(
    FEATURE_AUDIT_FILE,
    index=False
)

cv_df = pd.DataFrame(cv_rows)

# =============================================================================
# FINAL DECISION
# =============================================================================

summary_rows = []

for target_name in targets:

    subset = audit_df[
        audit_df["target"] == target_name
    ]

    very_high_auc = int(
        (
            subset["auc"] >= 0.995
        ).sum()
    )

    high_auc = int(
        (
            subset["auc"] >= 0.95
        ).sum()
    )

    cv_subset = cv_df[
        cv_df["target"] == target_name
    ]

    if len(cv_subset):

        cv_mean = cv_subset["roc_auc"].mean()
        cv_min = cv_subset["roc_auc"].min()
        cv_max = cv_subset["roc_auc"].max()

    else:

        cv_mean = np.nan
        cv_min = np.nan
        cv_max = np.nan


    if very_high_auc > 0:

        status = "REVIEW_REQUIRED"

    elif (
        np.isfinite(cv_mean)
        and cv_mean >= 0.995
    ):

        status = "STRONG_LEAKAGE_SUSPECTED"

    elif (
        np.isfinite(cv_mean)
        and cv_mean >= 0.90
    ):

        status = "HIGH_PERFORMANCE_REQUIRES_REVIEW"

    else:

        status = "NO_OBVIOUS_LEAKAGE"


    summary_rows.append({
        "target": target_name,
        "features_auc_ge_0.995": very_high_auc,
        "features_auc_ge_0.95": high_auc,
        "group_cv_mean_auc": cv_mean,
        "group_cv_min_auc": cv_min,
        "group_cv_max_auc": cv_max,
        "status": status
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print()
print("=" * 80)
print("TARGET LEAKAGE AUDIT COMPLETE")
print("=" * 80)

print()
print(summary_df.to_string(index=False))

print()
print("SAVED:")
print(FEATURE_AUDIT_FILE)
print(SUMMARY_FILE)

print()
print("=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    "Do NOT perform hyperparameter tuning, SHAP, or final ML "
    "until this audit is reviewed."
)

print()
print("STATUS: REVIEW REQUIRED")
print("=" * 80)