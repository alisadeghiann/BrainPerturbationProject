# ============================================================
# PERTURBATION / WHAT-IF
# SUBJECT-LEVEL GENERALIZATION ML VALIDATION V1
# ============================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

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
    / "ml_subject_generalization_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = (
    OUTPUT_DIR
    / "subject_generalization_v1_results.csv"
)

FOLD_FILE = (
    OUTPUT_DIR
    / "subject_generalization_v1_fold_results.csv"
)

SUBJECT_FILE = (
    OUTPUT_DIR
    / "subject_generalization_v1_subject_results.csv"
)

FEATURE_FILE = (
    OUTPUT_DIR
    / "subject_generalization_v1_feature_list.csv"
)

QC_FILE = (
    OUTPUT_DIR
    / "subject_generalization_v1_qc.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("PERTURBATION / WHAT-IF ML SUBJECT-LEVEL GENERALIZATION V1")
print("=" * 90)
print(f"Project root: {BASE}")
print()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("LOADING ML DATASET")
print("=" * 90)

if not INPUT.exists():
    raise FileNotFoundError(
        f"ML-ready dataset not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

print(f"Input file: {INPUT}")
print(f"Rows:       {len(df):,}")
print(f"Columns:    {len(df.columns):,}")
print()


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "subject",
    "target_remember",
    "target_correct",
]

missing_required = [
    c for c in required_columns
    if c not in df.columns
]

if missing_required:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_required)
    )

print("=" * 90)
print("DATASET VALIDATION")
print("=" * 90)

print(f"Subjects: {df['subject'].nunique()}")
print(f"Rows:     {len(df):,}")

print()
print("Target distributions:")

print()
print("target_remember")
print(df["target_remember"].value_counts(dropna=False).sort_index())

print()
print("target_correct")
print(df["target_correct"].value_counts(dropna=False).sort_index())

print()


# ============================================================
# SUBJECT STRUCTURE
# ============================================================

print("=" * 90)
print("SUBJECT STRUCTURE")
print("=" * 90)

subject_counts = (
    df.groupby("subject")
    .size()
    .reset_index(name="n_trials")
)

print(
    f"Minimum trials/subject: {subject_counts['n_trials'].min()}"
)
print(
    f"Maximum trials/subject: {subject_counts['n_trials'].max()}"
)
print(
    f"Mean trials/subject:    {subject_counts['n_trials'].mean():.2f}"
)

print()


# ============================================================
# FEATURE SELECTION
# ============================================================

print("=" * 90)
print("FEATURE SELECTION")
print("=" * 90)

excluded_columns = {
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

# Explicitly exclude known constant technical columns.
constant_columns = [
    "sfreq",
    "n_channels",
    "n_timepoints",
]

excluded_columns.update(constant_columns)

candidate_features = [
    c for c in df.columns
    if c not in excluded_columns
]

numeric_features = []

for col in candidate_features:
    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_features.append(col)

if len(numeric_features) == 0:
    raise ValueError(
        "No numeric ML features were found."
    )

print(f"Candidate numeric features: {len(numeric_features)}")
print()

print("Excluded columns:")
for c in sorted(excluded_columns):
    if c in df.columns:
        print(f"  {c}")

print()
print(f"Final ML features: {len(numeric_features)}")
print()


# ============================================================
# REMOVE REMAINING CONSTANT FEATURES
# ============================================================

constant_detected = []

for col in numeric_features:
    if df[col].nunique(dropna=False) <= 1:
        constant_detected.append(col)

if constant_detected:
    print("=" * 90)
    print("ADDITIONAL CONSTANT FEATURES DETECTED")
    print("=" * 90)

    for c in constant_detected:
        print(f"  {c}")

    numeric_features = [
        c for c in numeric_features
        if c not in constant_detected
    ]

    print()

print(f"Final non-constant features: {len(numeric_features)}")
print()


# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_df = pd.DataFrame({
    "feature": numeric_features
})

feature_df.to_csv(
    FEATURE_FILE,
    index=False
)


# ============================================================
# PREPARE X / SUBJECT GROUPS
# ============================================================

X = df[numeric_features].copy()

groups = df["subject"].astype(str)

# Replace infinite values.
X = X.replace([np.inf, -np.inf], np.nan)

nan_count = int(X.isna().sum().sum())

print("=" * 90)
print("FEATURE MATRIX QC")
print("=" * 90)

print(f"Rows:              {len(X):,}")
print(f"Features:          {X.shape[1]}")
print(f"NaN cells:         {nan_count:,}")
print(f"Inf cells:         0")
print()


# ============================================================
# MODELS
# ============================================================

models = {

    "logistic_regression": Pipeline([
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
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "random_forest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
                max_features="sqrt"
            )
        )
    ])
}


# ============================================================
# TARGETS
# ============================================================

targets = {
    "remember": "target_remember",
    "correct": "target_correct",
}


# ============================================================
# CV CONFIGURATION
# ============================================================

N_SPLITS = 5

print("=" * 90)
print("GROUP-AWARE CROSS-VALIDATION")
print("=" * 90)

print(f"CV method: StratifiedGroupKFold")
print(f"Splits:    {N_SPLITS}")
print("Grouping:  subject")
print()

print(
    "IMPORTANT: No subject can appear in both "
    "training and test data within a fold."
)

print()


# ============================================================
# RESULT CONTAINERS
# ============================================================

summary_results = []
fold_results = []
subject_results = []


# ============================================================
# MAIN VALIDATION LOOP
# ============================================================

for target_name, target_column in targets.items():

    print("=" * 90)
    print(f"TARGET: {target_name.upper()}")
    print("=" * 90)

    y = df[target_column].copy()

    valid_mask = y.notna() & groups.notna()

    X_target = X.loc[valid_mask].copy()
    y_target = y.loc[valid_mask].astype(int)
    groups_target = groups.loc[valid_mask]

    print(f"Rows used:      {len(X_target):,}")
    print(
        f"Subjects used:  {groups_target.nunique()}"
    )

    print()
    print("Class distribution:")
    print(y_target.value_counts().sort_index())
    print()

    # --------------------------------------------------------
    # Stratified Group K-Fold
    # --------------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    for model_name, model in models.items():

        print("-" * 80)
        print(f"MODEL: {model_name}")
        print("-" * 80)

        fold_metrics = []

        all_true = []
        all_pred = []
        all_prob = []

        for fold_number, (train_idx, test_idx) in enumerate(
            cv.split(
                X_target,
                y_target,
                groups_target
            ),
            start=1
        ):

            X_train = X_target.iloc[train_idx]
            X_test = X_target.iloc[test_idx]

            y_train = y_target.iloc[train_idx]
            y_test = y_target.iloc[test_idx]

            train_subjects = set(
                groups_target.iloc[train_idx]
            )

            test_subjects = set(
                groups_target.iloc[test_idx]
            )

            overlap = train_subjects.intersection(
                test_subjects
            )

            if overlap:
                raise RuntimeError(
                    f"SUBJECT LEAKAGE DETECTED in fold "
                    f"{fold_number}: {overlap}"
                )

            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(
                    X_test
                )[:, 1]
            else:
                y_prob = None

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            balanced_accuracy = balanced_accuracy_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            if (
                y_prob is not None
                and y_test.nunique() == 2
            ):
                roc_auc = roc_auc_score(
                    y_test,
                    y_prob
                )
            else:
                roc_auc = np.nan

            fold_metrics.append({
                "fold": fold_number,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1": f1,
                "roc_auc": roc_auc,
            })

            # ------------------------------------------------
            # Fold result
            # ------------------------------------------------

            fold_results.append({
                "target": target_name,
                "model": model_name,
                "fold": fold_number,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_subjects": len(train_subjects),
                "test_subjects": len(test_subjects),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1": f1,
                "roc_auc": roc_auc,
                "subject_overlap": len(overlap),
            })

            # ------------------------------------------------
            # Store global predictions
            # ------------------------------------------------

            all_true.extend(
                y_test.tolist()
            )

            all_pred.extend(
                y_pred.tolist()
            )

            if y_prob is not None:
                all_prob.extend(
                    y_prob.tolist()
                )

            # ------------------------------------------------
            # Subject-level performance inside test fold
            # ------------------------------------------------

            fold_test_df = df.loc[
                X_test.index,
                ["subject"]
            ].copy()

            fold_test_df["y_true"] = y_test.values
            fold_test_df["y_pred"] = y_pred

            if y_prob is not None:
                fold_test_df["y_prob"] = y_prob

            for subject, sdf in fold_test_df.groupby(
                "subject"
            ):

                subject_accuracy = accuracy_score(
                    sdf["y_true"],
                    sdf["y_pred"]
                )

                subject_balanced_accuracy = (
                    balanced_accuracy_score(
                        sdf["y_true"],
                        sdf["y_pred"]
                    )
                    if sdf["y_true"].nunique() == 2
                    else np.nan
                )

                subject_f1 = f1_score(
                    sdf["y_true"],
                    sdf["y_pred"],
                    zero_division=0
                )

                if (
                    "y_prob" in sdf.columns
                    and sdf["y_true"].nunique() == 2
                ):
                    subject_auc = roc_auc_score(
                        sdf["y_true"],
                        sdf["y_prob"]
                    )
                else:
                    subject_auc = np.nan

                subject_results.append({
                    "target": target_name,
                    "model": model_name,
                    "fold": fold_number,
                    "subject": subject,
                    "n_trials": len(sdf),
                    "accuracy": subject_accuracy,
                    "balanced_accuracy": subject_balanced_accuracy,
                    "f1": subject_f1,
                    "roc_auc": subject_auc,
                })

        # ----------------------------------------------------
        # Aggregate fold metrics
        # ----------------------------------------------------

        fold_df = pd.DataFrame(fold_metrics)

        accuracy_mean = fold_df["accuracy"].mean()
        accuracy_std = fold_df["accuracy"].std(ddof=1)

        balanced_mean = (
            fold_df["balanced_accuracy"].mean()
        )
        balanced_std = (
            fold_df["balanced_accuracy"].std(ddof=1)
        )

        f1_mean = fold_df["f1"].mean()
        f1_std = fold_df["f1"].std(ddof=1)

        auc_mean = fold_df["roc_auc"].mean()
        auc_std = fold_df["roc_auc"].std(ddof=1)

        # Global out-of-fold metrics
        global_accuracy = accuracy_score(
            all_true,
            all_pred
        )

        global_balanced = balanced_accuracy_score(
            all_true,
            all_pred
        )

        global_f1 = f1_score(
            all_true,
            all_pred,
            zero_division=0
        )

        if (
            len(all_prob) == len(all_true)
            and len(np.unique(all_true)) == 2
        ):
            global_auc = roc_auc_score(
                all_true,
                all_prob
            )
        else:
            global_auc = np.nan

        print(
            f"Accuracy:          "
            f"{accuracy_mean:.4f} +/- {accuracy_std:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{balanced_mean:.4f} +/- {balanced_std:.4f}"
        )

        print(
            f"F1:                "
            f"{f1_mean:.4f} +/- {f1_std:.4f}"
        )

        print(
            f"ROC-AUC:           "
            f"{auc_mean:.4f} +/- {auc_std:.4f}"
        )

        print()

        print(
            f"OOF Accuracy:          {global_accuracy:.4f}"
        )
        print(
            f"OOF Balanced Accuracy: {global_balanced:.4f}"
        )
        print(
            f"OOF F1:                {global_f1:.4f}"
        )
        print(
            f"OOF ROC-AUC:           {global_auc:.4f}"
        )

        print()

        summary_results.append({
            "target": target_name,
            "model": model_name,
            "n_rows": len(X_target),
            "n_subjects": groups_target.nunique(),
            "n_features": len(numeric_features),

            "accuracy_mean": accuracy_mean,
            "accuracy_std": accuracy_std,

            "balanced_accuracy_mean": balanced_mean,
            "balanced_accuracy_std": balanced_std,

            "f1_mean": f1_mean,
            "f1_std": f1_std,

            "roc_auc_mean": auc_mean,
            "roc_auc_std": auc_std,

            "oof_accuracy": global_accuracy,
            "oof_balanced_accuracy": global_balanced,
            "oof_f1": global_f1,
            "oof_roc_auc": global_auc,
        })


# ============================================================
# SAVE RESULTS
# ============================================================

print("=" * 90)
print("SAVING SUBJECT-GENERALIZATION RESULTS")
print("=" * 90)

summary_df = pd.DataFrame(
    summary_results
)

fold_df = pd.DataFrame(
    fold_results
)

subject_df = pd.DataFrame(
    subject_results
)

summary_df.to_csv(
    RESULTS_FILE,
    index=False
)

fold_df.to_csv(
    FOLD_FILE,
    index=False
)

subject_df.to_csv(
    SUBJECT_FILE,
    index=False
)


# ============================================================
# FINAL QC
# ============================================================

print("=" * 90)
print("FINAL SUBJECT GENERALIZATION QC")
print("=" * 90)

numeric_summary = summary_df.select_dtypes(
    include=[np.number]
)

nan_numeric = int(
    numeric_summary.isna().sum().sum()
)

inf_numeric = int(
    np.isinf(
        numeric_summary.to_numpy()
    ).sum()
)

duplicate_fold_rows = int(
    fold_df.duplicated(
        subset=[
            "target",
            "model",
            "fold"
        ]
    ).sum()
)

subject_overlap_count = int(
    fold_df["subject_overlap"].sum()
)

print(
    f"Targets analyzed:          "
    f"{summary_df['target'].nunique()}"
)

print(
    f"Models analyzed:           "
    f"{summary_df['model'].nunique()}"
)

print(
    f"Features analyzed:         "
    f"{len(numeric_features)}"
)

print(
    f"Subjects:                  "
    f"{df['subject'].nunique()}"
)

print(
    f"CV folds:                  "
    f"{N_SPLITS}"
)

print(
    f"NaN numeric values:        "
    f"{nan_numeric}"
)

print(
    f"Inf numeric values:        "
    f"{inf_numeric}"
)

print(
    f"Duplicate fold rows:       "
    f"{duplicate_fold_rows}"
)

print(
    f"Subject train/test overlap:"
    f" {subject_overlap_count}"
)

print()


# ============================================================
# QC TABLE
# ============================================================

qc = pd.DataFrame([{
    "input_file": str(INPUT),
    "rows": len(df),
    "subjects": df["subject"].nunique(),
    "features": len(numeric_features),
    "targets": summary_df["target"].nunique(),
    "models": summary_df["model"].nunique(),
    "cv_folds": N_SPLITS,
    "cv_method": "StratifiedGroupKFold",
    "group_variable": "subject",
    "nan_numeric_values": nan_numeric,
    "inf_numeric_values": inf_numeric,
    "duplicate_fold_rows": duplicate_fold_rows,
    "subject_train_test_overlap": subject_overlap_count,
}])

qc.to_csv(
    QC_FILE,
    index=False
)


# ============================================================
# FINAL STATUS
# ============================================================

if (
    nan_numeric == 0
    and inf_numeric == 0
    and duplicate_fold_rows == 0
    and subject_overlap_count == 0
):

    status = "PASS"

else:

    status = "REVIEW_REQUIRED"


print("=" * 90)
print("SAVED")
print("=" * 90)

print(f"Results:")
print(RESULTS_FILE)

print()

print(f"Fold results:")
print(FOLD_FILE)

print()

print(f"Subject results:")
print(SUBJECT_FILE)

print()

print(f"Feature list:")
print(FEATURE_FILE)

print()

print(f"QC:")
print(QC_FILE)

print()

print("=" * 90)
print("PERTURBATION / WHAT-IF SUBJECT-LEVEL GENERALIZATION V1 COMPLETE")
print("=" * 90)

print(
    f"STATUS: {status} - "
    "SUBJECT-AWARE GENERALIZATION VALIDATION"
)