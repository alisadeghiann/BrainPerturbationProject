# ============================================================
# PERTURBATION ML VALIDATION V2
# Subject-Level Cross-Validated ML Validation
# ============================================================

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "ml_validation_v2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# TOP SCIENTIFIC PERTURBATION FEATURES
# ============================================================

TOP_FEATURES = [
    "gamma_temporal",
    "beta_temporal",
    "delta_rel",
    "theta_rel",
    "alpha_beta_central_ratio",
    "beta_parietal",
    "alpha_central",
    "theta_central",
    "theta_frontoparietal_ratio",
    "delta_frontoparietal_diff",
    "delta_central",
    "delta_abs",
    "alpha_beta_frontal_ratio",
    "beta_frontoparietal_diff",
    "alpha_rel",
    "gamma_parietal",
    "theta_abs",
    "gamma_occipital",
    "delta_parietal",
    "alpha_beta_parietal_ratio",
]

TARGETS = {
    "remember": "target_remember",
    "correct": "target_correct",
}

# ============================================================
# SEARCH FOR ML DATASET
# ============================================================

print("=" * 90)
print("PERTURBATION ML VALIDATION V2")
print("=" * 90)

print()
print("Project root:")
print(BASE)

print()
print("=" * 90)
print("SEARCHING FOR EEG FEATURE DATASET")
print("=" * 90)

candidate_files = [
    BASE / "features" / "perturbation_analysis" / "perturbation_subject_effects.csv",
    BASE / "features" / "perturbation_analysis" / "perturbation_features.csv",
    BASE / "features" / "final_features.csv",
    BASE / "features" / "all_features.csv",
]

INPUT = None

for f in candidate_files:
    if f.exists():
        try:
            tmp = pd.read_csv(f, nrows=5)

            required_check = {
                "subject",
                "target_remember",
                "target_correct"
            }

            if required_check.issubset(set(tmp.columns)):
                INPUT = f
                break

        except Exception:
            pass

# Recursive fallback search
if INPUT is None:

    print("Candidate files did not contain the required columns.")
    print("Searching recursively...")

    for f in BASE.rglob("*.csv"):

        # Avoid searching generated ML outputs
        if "ml_validation" in str(f):
            continue

        try:
            tmp = pd.read_csv(f, nrows=5)

            required_check = {
                "subject",
                "target_remember",
                "target_correct"
            }

            if required_check.issubset(set(tmp.columns)):
                INPUT = f
                break

        except Exception:
            continue

if INPUT is None:
    raise FileNotFoundError(
        "No EEG dataset containing subject, target_remember "
        "and target_correct was found."
    )

print()
print("INPUT FILE FOUND:")
print(INPUT)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)

print()
print("=" * 90)
print("INPUT VALIDATION")
print("=" * 90)

print(f"Rows:       {len(df):,}")
print(f"Columns:    {len(df.columns):,}")
print(f"Subjects:   {df['subject'].nunique():,}")

missing_features = [
    f for f in TOP_FEATURES
    if f not in df.columns
]

if missing_features:
    raise ValueError(
        "Missing scientific features:\n"
        + "\n".join(missing_features)
    )

for target_name, target_col in TARGETS.items():

    if target_col not in df.columns:
        raise ValueError(
            f"Required target column not found: {target_col}"
        )

print()
print("Scientific features:")
for i, feature in enumerate(TOP_FEATURES, 1):
    print(f"{i:02d}. {feature}")

# ============================================================
# CLEAN TARGETS
# ============================================================

print()
print("=" * 90)
print("TARGET VALIDATION")
print("=" * 90)

for target_name, target_col in TARGETS.items():

    values = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    print()
    print(f"{target_name}: {target_col}")
    print(values.value_counts(dropna=False).sort_index())

# ============================================================
# REMOVE INVALID TARGET ROWS
# ============================================================

df_ml = df.copy()

for target_col in TARGETS.values():

    df_ml[target_col] = pd.to_numeric(
        df_ml[target_col],
        errors="coerce"
    )

# Subject must exist
df_ml = df_ml.dropna(subset=["subject"])

# ============================================================
# FEATURES
# ============================================================

X = df_ml[TOP_FEATURES].copy()

# Convert everything to numeric
for feature in TOP_FEATURES:
    X[feature] = pd.to_numeric(
        X[feature],
        errors="coerce"
    )

# Infinite values -> NaN
X = X.replace([np.inf, -np.inf], np.nan)

print()
print("=" * 90)
print("FEATURE MATRIX")
print("=" * 90)

print(f"Rows:       {len(X):,}")
print(f"Features:   {X.shape[1]:,}")
print(f"NaN cells:  {X.isna().sum().sum():,}")

# ============================================================
# GROUPS
# ============================================================

groups = df_ml["subject"].astype(str)

print()
print("=" * 90)
print("SUBJECT-LEVEL GROUP STRUCTURE")
print("=" * 90)

print(f"Unique subjects: {groups.nunique()}")

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
                max_depth=6,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])
}

# ============================================================
# GROUPED CROSS VALIDATION
# ============================================================

N_SPLITS = min(5, groups.nunique())

if N_SPLITS < 3:
    raise ValueError(
        "At least 3 subjects are required for subject-level CV."
    )

cv = GroupKFold(
    n_splits=N_SPLITS
)

all_results = []
prediction_results = []

# ============================================================
# TARGET LOOP
# ============================================================

for target_name, target_col in TARGETS.items():

    print()
    print("=" * 90)
    print(f"TARGET: {target_name.upper()}")
    print("=" * 90)

    y = df_ml[target_col].copy()

    valid = y.notna()

    X_target = X.loc[valid].copy()
    y_target = y.loc[valid].astype(int)
    groups_target = groups.loc[valid]

    print(f"Rows:       {len(y_target):,}")
    print(f"Subjects:   {groups_target.nunique():,}")
    print("Class distribution:")
    print(y_target.value_counts().sort_index())

    if y_target.nunique() < 2:
        print(
            f"WARNING: {target_name} contains only one class. "
            "Skipping."
        )
        continue

    for model_name, model in models.items():

        print()
        print("-" * 90)
        print(f"MODEL: {model_name}")
        print("-" * 90)

        fold_rows = []

        for fold, (train_idx, test_idx) in enumerate(
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

            group_train = groups_target.iloc[train_idx]
            group_test = groups_target.iloc[test_idx]

            train_subjects = set(group_train)
            test_subjects = set(group_test)

            overlap = train_subjects.intersection(
                test_subjects
            )

            if overlap:
                raise RuntimeError(
                    f"DATA LEAKAGE DETECTED in fold {fold}: "
                    f"{overlap}"
                )

            model.fit(
                X_train,
                y_train
            )

            pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_test)[:, 1]
            else:
                prob = pred.astype(float)

            acc = accuracy_score(
                y_test,
                pred
            )

            bal_acc = balanced_accuracy_score(
                y_test,
                pred
            )

            precision = precision_score(
                y_test,
                pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                pred,
                zero_division=0
            )

            if y_test.nunique() == 2:
                auc = roc_auc_score(
                    y_test,
                    prob
                )
            else:
                auc = np.nan

            cm = confusion_matrix(
                y_test,
                pred,
                labels=[0, 1]
            )

            tn, fp, fn, tp = cm.ravel()

            fold_result = {
                "target": target_name,
                "model": model_name,
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_subjects": len(train_subjects),
                "test_subjects": len(test_subjects),
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "roc_auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "subject_overlap": len(overlap)
            }

            fold_rows.append(
                fold_result
            )

            prediction_results.extend(
                pd.DataFrame({
                    "target": target_name,
                    "model": model_name,
                    "fold": fold,
                    "subject": group_test.values,
                    "true_label": y_test.values,
                    "predicted_label": pred,
                    "predicted_probability": prob
                }).to_dict("records")
            )

            print(
                f"Fold {fold}: "
                f"AUC={auc:.4f} | "
                f"BalancedAcc={bal_acc:.4f} | "
                f"F1={f1:.4f} | "
                f"Subjects={len(test_subjects)}"
            )

        # ----------------------------------------------------
        # MODEL SUMMARY
        # ----------------------------------------------------

        fold_df = pd.DataFrame(
            fold_rows
        )

        summary = {
            "target": target_name,
            "model": model_name,
            "n_folds": len(fold_df),
            "subjects": groups_target.nunique(),
            "rows": len(y_target),

            "accuracy_mean":
                fold_df["accuracy"].mean(),

            "accuracy_std":
                fold_df["accuracy"].std(),

            "balanced_accuracy_mean":
                fold_df["balanced_accuracy"].mean(),

            "balanced_accuracy_std":
                fold_df["balanced_accuracy"].std(),

            "roc_auc_mean":
                fold_df["roc_auc"].mean(),

            "roc_auc_std":
                fold_df["roc_auc"].std(),

            "precision_mean":
                fold_df["precision"].mean(),

            "recall_mean":
                fold_df["recall"].mean(),

            "f1_mean":
                fold_df["f1"].mean(),

            "leakage_subject_overlap_total":
                fold_df["subject_overlap"].sum()
        }

        all_results.append(summary)

        print()
        print(
            f"SUMMARY | "
            f"AUC={summary['roc_auc_mean']:.4f} ± "
            f"{summary['roc_auc_std']:.4f} | "
            f"BalancedAcc="
            f"{summary['balanced_accuracy_mean']:.4f} ± "
            f"{summary['balanced_accuracy_std']:.4f}"
        )

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.DataFrame(
    prediction_results
)

results_path = (
    OUTPUT_DIR
    / "perturbation_ml_validation_results_v2.csv"
)

predictions_path = (
    OUTPUT_DIR
    / "perturbation_ml_validation_predictions_v2.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 90)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 90)

importance_rows = []

for target_name, target_col in TARGETS.items():

    y = df_ml[target_col].copy()
    valid = y.notna()

    X_target = X.loc[valid].copy()
    y_target = y.loc[valid].astype(int)
    groups_target = groups.loc[valid]

    if y_target.nunique() < 2:
        continue

    for model_name in [
        "logistic_regression",
        "random_forest"
    ]:

        model = models[model_name]

        # Fit only on complete target dataset for
        # descriptive feature importance.
        model.fit(
            X_target,
            y_target
        )

        fitted_model = model.named_steps["model"]

        if model_name == "logistic_regression":

            coefficients = (
                fitted_model.coef_[0]
            )

            for feature, coefficient in zip(
                TOP_FEATURES,
                coefficients
            ):

                importance_rows.append({
                    "target": target_name,
                    "model": model_name,
                    "feature": feature,
                    "importance":
                        abs(coefficient),
                    "signed_coefficient":
                        coefficient
                })

        elif model_name == "random_forest":

            importances = (
                fitted_model.feature_importances_
            )

            for feature, importance in zip(
                TOP_FEATURES,
                importances
            ):

                importance_rows.append({
                    "target": target_name,
                    "model": model_name,
                    "feature": feature,
                    "importance":
                        importance,
                    "signed_coefficient":
                        np.nan
                })

importance_df = pd.DataFrame(
    importance_rows
)

importance_path = (
    OUTPUT_DIR
    / "perturbation_ml_feature_importance_v2.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

# ============================================================
# FINAL QC
# ============================================================

print()
print("=" * 90)
print("FINAL ML VALIDATION QC")
print("=" * 90)

if len(results_df) > 0:

    print(
        f"Targets validated:       "
        f"{results_df['target'].nunique()}"
    )

    print(
        f"Models validated:        "
        f"{results_df['model'].nunique()}"
    )

    print(
        f"Validation rows:         "
        f"{len(results_df)}"
    )

    print(
        f"Subjects:                "
        f"{groups.nunique()}"
    )

    print(
        f"Scientific features:     "
        f"{len(TOP_FEATURES)}"
    )

    print(
        f"NaN numeric values:      "
        f"{results_df.select_dtypes(include=np.number).isna().sum().sum()}"
    )

    print(
        f"Inf numeric values:      "
        f"{np.isinf(
            results_df.select_dtypes(include=np.number)
        ).sum().sum()}"
    )

    print(
        f"Subject leakage overlap: "
        f"{results_df['leakage_subject_overlap_total'].sum()}"
    )

# ============================================================
# TOP RESULTS
# ============================================================

print()
print("=" * 90)
print("FINAL ML RESULTS")
print("=" * 90)

if len(results_df) > 0:

    display_cols = [
        "target",
        "model",
        "roc_auc_mean",
        "roc_auc_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_mean"
    ]

    print(
        results_df[
            display_cols
        ].to_string(
            index=False
        )
    )

# ============================================================
# SAVE QC
# ============================================================

qc = {
    "input_file": str(INPUT),
    "subjects": int(groups.nunique()),
    "rows": int(len(df_ml)),
    "scientific_features": len(TOP_FEATURES),
    "targets_validated":
        int(results_df["target"].nunique())
        if len(results_df) else 0,
    "models_validated":
        int(results_df["model"].nunique())
        if len(results_df) else 0,
    "result_rows": int(len(results_df)),
    "prediction_rows": int(len(predictions_df)),
    "nan_feature_cells":
        int(X.isna().sum().sum()),
    "subject_leakage_overlap":
        int(
            results_df[
                "leakage_subject_overlap_total"
            ].sum()
        )
        if len(results_df) else 0
}

qc_df = pd.DataFrame(
    [qc]
)

qc_path = (
    OUTPUT_DIR
    / "perturbation_ml_validation_qc_v2.csv"
)

qc_df.to_csv(
    qc_path,
    index=False
)

# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 90)
print("SUBJECT-LEVEL PERTURBATION ML VALIDATION V2 COMPLETE")
print("=" * 90)

print()
print("Results:")
print(results_path)

print()
print("Predictions:")
print(predictions_path)

print()
print("Feature importance:")
print(importance_path)

print()
print("QC:")
print(qc_path)

print()
print("=" * 90)
print("STATUS: PASS - SUBJECT-LEVEL ML VALIDATION CREATED")
print("=" * 90)