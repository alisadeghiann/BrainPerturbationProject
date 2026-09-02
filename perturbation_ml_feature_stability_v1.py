from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance


# =============================================================================
# PERTURBATION / WHAT-IF ML FEATURE STABILITY V1
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "ml_ready_v2"
    / "ml_ready_dataset_v2.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "ml_feature_stability_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "ml_feature_stability_v1_results.csv"
FOLD_FILE = OUTPUT_DIR / "ml_feature_stability_v1_fold_results.csv"
QC_FILE = OUTPUT_DIR / "ml_feature_stability_v1_qc.csv"


print("=" * 90)
print("PERTURBATION / WHAT-IF ML FEATURE STABILITY V1")
print("=" * 90)

print(f"Project root: {BASE}")
print(f"Input:        {INPUT}")

if not INPUT.exists():
    raise FileNotFoundError(f"Input dataset not found:\n{INPUT}")

df = pd.read_csv(INPUT)

print("\n" + "=" * 90)
print("DATASET SUMMARY")
print("=" * 90)

print(f"Rows:      {len(df):,}")
print(f"Columns:   {len(df.columns)}")

if "subject" not in df.columns:
    raise ValueError("Required column 'subject' not found.")

TARGETS = {
    "remember": "target_remember",
    "correct": "target_correct",
}

for name, col in TARGETS.items():
    if col not in df.columns:
        raise ValueError(f"Required target column not found: {col}")


# =============================================================================
# REMOVE NON-FEATURE COLUMNS
# =============================================================================

NON_FEATURE_COLUMNS = {
    "file",
    "subject",
    "run",
    "epoch",
    "trial",
    "memory_cond",
    "probe_type",
    "probe_letter",
    "target_label",
    "target_remember",
    "target_correct",
}

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

feature_columns = [
    c for c in numeric_columns
    if c not in NON_FEATURE_COLUMNS
]

if not feature_columns:
    raise ValueError("No numeric EEG feature columns found.")

print("\n" + "=" * 90)
print("FEATURE SET")
print("=" * 90)

print(f"Numeric EEG features: {len(feature_columns)}")


# =============================================================================
# REMOVE CONSTANT FEATURES
# =============================================================================

constant_features = []

for feature in feature_columns:
    if df[feature].nunique(dropna=False) <= 1:
        constant_features.append(feature)

analysis_features = [
    f for f in feature_columns
    if f not in constant_features
]

print(f"Constant features removed: {len(constant_features)}")
print(f"Features analyzed:          {len(analysis_features)}")

if constant_features:
    print("\nConstant features:")
    for f in constant_features:
        print(" ", f)


# =============================================================================
# BASIC NUMERIC QC
# =============================================================================

numeric_analysis = df[analysis_features]

nan_cells = int(numeric_analysis.isna().sum().sum())
inf_cells = int(np.isinf(numeric_analysis.to_numpy()).sum())

print("\n" + "=" * 90)
print("NUMERIC QC")
print("=" * 90)

print(f"NaN numeric cells: {nan_cells}")
print(f"Inf numeric cells: {inf_cells}")

if nan_cells > 0:
    raise ValueError("NaN values found in EEG features.")

if inf_cells > 0:
    raise ValueError("Inf values found in EEG features.")


# =============================================================================
# SUBJECT-AWARE CV
# =============================================================================

groups = df["subject"].astype(str)

N_SPLITS = 5

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42,
)


# =============================================================================
# STORAGE
# =============================================================================

fold_records = []
importance_records = []


# =============================================================================
# TARGET LOOP
# =============================================================================

for target_name, target_column in TARGETS.items():

    print("\n" + "=" * 90)
    print(f"TARGET: {target_name.upper()}")
    print("=" * 90)

    work = df[
        analysis_features + [target_column, "subject"]
    ].copy()

    work = work.dropna(
        subset=analysis_features + [target_column]
    )

    X = work[analysis_features]
    y = work[target_column].astype(int)
    g = work["subject"].astype(str)

    print(f"Rows used:     {len(work):,}")
    print(f"Subjects used: {g.nunique()}")

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    if y.nunique() < 2:
        raise ValueError(
            f"Target {target_name} does not contain two classes."
        )

    # -------------------------------------------------------------------------
    # FOLD LOOP
    # -------------------------------------------------------------------------

    for fold_id, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups=g),
        start=1
    ):

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        train_subjects = set(g.iloc[train_idx])
        test_subjects = set(g.iloc[test_idx])

        overlap = train_subjects.intersection(test_subjects)

        if overlap:
            raise RuntimeError(
                f"Subject leakage detected in fold {fold_id}: {overlap}"
            )

        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42
                    )
                ),
            ]
        )

        model.fit(X_train, y_train)

        # ---------------------------------------------------------------------
        # COEFFICIENT IMPORTANCE
        # ---------------------------------------------------------------------

        coefficients = (
            model.named_steps["model"]
            .coef_[0]
        )

        abs_coefficients = np.abs(coefficients)

        # ---------------------------------------------------------------------
        # PERMUTATION IMPORTANCE ON TEST SUBJECTS
        # ---------------------------------------------------------------------

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=10,
            random_state=42 + fold_id,
            n_jobs=-1,
        )

        permutation_mean = perm.importances_mean
        permutation_std = perm.importances_std

        for feature_idx, feature in enumerate(analysis_features):

            importance_records.append(
                {
                    "target": target_name,
                    "fold": fold_id,
                    "feature": feature,
                    "coefficient": float(
                        coefficients[feature_idx]
                    ),
                    "abs_coefficient": float(
                        abs_coefficients[feature_idx]
                    ),
                    "permutation_importance": float(
                        permutation_mean[feature_idx]
                    ),
                    "permutation_importance_std": float(
                        permutation_std[feature_idx]
                    ),
                    "train_rows": len(train_idx),
                    "test_rows": len(test_idx),
                    "train_subjects": len(train_subjects),
                    "test_subjects": len(test_subjects),
                    "subject_overlap": len(overlap),
                }
            )

        fold_records.append(
            {
                "target": target_name,
                "fold": fold_id,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_subjects": len(train_subjects),
                "test_subjects": len(test_subjects),
                "subject_overlap": len(overlap),
            }
        )

        print(
            f"Fold {fold_id}: "
            f"train={len(train_idx):,}, "
            f"test={len(test_idx):,}, "
            f"train_subjects={len(train_subjects)}, "
            f"test_subjects={len(test_subjects)}"
        )


# =============================================================================
# AGGREGATE FEATURE STABILITY
# =============================================================================

importance_df = pd.DataFrame(importance_records)

if importance_df.empty:
    raise RuntimeError("No feature importance results were generated.")


def safe_cv_mean(series):
    return float(series.mean())


def safe_cv_std(series):
    return float(series.std(ddof=1)) if len(series) > 1 else 0.0


summary_records = []

for (target, feature), group in importance_df.groupby(
    ["target", "feature"]
):

    coef_values = group["coefficient"].to_numpy()
    abs_coef_values = group["abs_coefficient"].to_numpy()
    perm_values = group["permutation_importance"].to_numpy()

    coefficient_sign_consistency = float(
        np.mean(np.sign(coef_values) == np.sign(np.mean(coef_values)))
    )

    positive_fraction = float(
        np.mean(coef_values > 0)
    )

    negative_fraction = float(
        np.mean(coef_values < 0)
    )

    summary_records.append(
        {
            "target": target,
            "feature": feature,

            "mean_coefficient":
                safe_cv_mean(group["coefficient"]),

            "std_coefficient":
                safe_cv_std(group["coefficient"]),

            "mean_abs_coefficient":
                safe_cv_mean(group["abs_coefficient"]),

            "std_abs_coefficient":
                safe_cv_std(group["abs_coefficient"]),

            "mean_permutation_importance":
                safe_cv_mean(group["permutation_importance"]),

            "std_permutation_importance":
                safe_cv_std(group["permutation_importance"]),

            "mean_permutation_importance_std":
                safe_cv_mean(
                    group["permutation_importance_std"]
                ),

            "coefficient_sign_consistency":
                coefficient_sign_consistency,

            "positive_fraction":
                positive_fraction,

            "negative_fraction":
                negative_fraction,

            "folds_available":
                int(len(group)),

            "stable_direction":
                bool(
                    coefficient_sign_consistency >= 0.8
                ),
        }
    )


summary_df = pd.DataFrame(summary_records)


# =============================================================================
# RANK FEATURES
# =============================================================================

summary_df["mean_abs_permutation_importance"] = (
    summary_df["mean_permutation_importance"].abs()
)

summary_df["stability_score"] = (
    summary_df["mean_abs_permutation_importance"]
    * summary_df["coefficient_sign_consistency"]
)

summary_df = summary_df.sort_values(
    ["target", "stability_score"],
    ascending=[True, False]
).reset_index(drop=True)

summary_df["scientific_ml_rank"] = (
    summary_df.groupby("target")
    .cumcount()
    + 1
)


# =============================================================================
# QC
# =============================================================================

fold_df = pd.DataFrame(fold_records)

numeric_summary = summary_df.select_dtypes(
    include=[np.number]
)

nan_summary = int(
    numeric_summary.isna().sum().sum()
)

inf_summary = int(
    np.isinf(numeric_summary.to_numpy()).sum()
)

duplicate_target_feature = int(
    summary_df.duplicated(
        subset=["target", "feature"]
    ).sum()
)

subject_overlap_total = int(
    fold_df["subject_overlap"].sum()
)


# =============================================================================
# PRINT TOP FEATURES
# =============================================================================

print("\n" + "=" * 90)
print("TOP STABLE ML FEATURES")
print("=" * 90)

for target_name in TARGETS.keys():

    subset = summary_df[
        summary_df["target"] == target_name
    ].head(20)

    print(f"\nTARGET: {target_name.upper()}")

    print(
        subset[
            [
                "scientific_ml_rank",
                "feature",
                "mean_abs_permutation_importance",
                "coefficient_sign_consistency",
                "mean_coefficient",
                "stability_score",
            ]
        ].to_string(index=False)
    )


# =============================================================================
# FINAL QC
# =============================================================================

print("\n" + "=" * 90)
print("FINAL ML FEATURE STABILITY QC")
print("=" * 90)

print(f"Targets analyzed:          {summary_df['target'].nunique()}")
print(f"Features analyzed:         {summary_df['feature'].nunique()}")
print(f"Total target-feature rows: {len(summary_df)}")
print(f"CV folds:                  {N_SPLITS}")
print(f"NaN numeric values:        {nan_summary}")
print(f"Inf numeric values:        {inf_summary}")
print(f"Duplicate target-feature:  {duplicate_target_feature}")
print(f"Subject train/test overlap:{subject_overlap_total}")


if nan_summary != 0:
    raise RuntimeError(
        "NaN numeric values detected in final ML stability results."
    )

if inf_summary != 0:
    raise RuntimeError(
        "Inf numeric values detected in final ML stability results."
    )

if duplicate_target_feature != 0:
    raise RuntimeError(
        "Duplicate target-feature rows detected."
    )

if subject_overlap_total != 0:
    raise RuntimeError(
        "Subject overlap detected across train/test folds."
    )


# =============================================================================
# SAVE
# =============================================================================

summary_df.to_csv(
    RESULTS_FILE,
    index=False
)

importance_df.to_csv(
    FOLD_FILE,
    index=False
)

qc_df = pd.DataFrame(
    [
        {
            "targets_analyzed":
                summary_df["target"].nunique(),

            "features_analyzed":
                summary_df["feature"].nunique(),

            "target_feature_rows":
                len(summary_df),

            "cv_folds":
                N_SPLITS,

            "subjects":
                df["subject"].nunique(),

            "nan_numeric_values":
                nan_summary,

            "inf_numeric_values":
                inf_summary,

            "duplicate_target_feature":
                duplicate_target_feature,

            "subject_train_test_overlap":
                subject_overlap_total,

            "constant_features_removed":
                len(constant_features),
        }
    ]
)

qc_df.to_csv(
    QC_FILE,
    index=False
)


print("\n" + "=" * 90)
print("SAVED")
print("=" * 90)

print(f"Results:")
print(RESULTS_FILE)

print(f"\nFold results:")
print(FOLD_FILE)

print(f"\nQC:")
print(QC_FILE)

print("\n" + "=" * 90)
print("PERTURBATION / WHAT-IF ML FEATURE STABILITY V1 COMPLETE")
print("=" * 90)

print("STATUS: PASS - ML FEATURE STABILITY CREATED")