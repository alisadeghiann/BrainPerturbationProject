# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
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
    / "baseline"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("ML BASELINE MODELING")
print("=" * 80)

print(f"Dataset: {INPUT}")
print(f"Split:   {SPLIT}")

df = pd.read_csv(INPUT)
split_df = pd.read_csv(SPLIT)

print()
print("DATASET")
print("-" * 80)
print(f"Rows:       {len(df):,}")
print(f"Subjects:   {df['subject'].nunique()}")
print(f"Runs:       {df['run'].nunique()}")


# =============================================================================
# NORMALIZE IDENTIFIERS
# =============================================================================

df["subject"] = df["subject"].astype(str).str.strip()

split_df["subject"] = split_df["subject"].astype(str).str.strip()
split_df["split"] = split_df["split"].astype(str).str.strip().str.lower()


# =============================================================================
# CHECK REQUIRED COLUMNS
# =============================================================================

required_dataset = [
    "subject",
    "target_remember",
    "target_correct",
]

required_split = [
    "subject",
    "split",
]

missing_dataset = [
    c for c in required_dataset
    if c not in df.columns
]

missing_split = [
    c for c in required_split
    if c not in split_df.columns
]

if missing_dataset:
    raise RuntimeError(
        f"Missing required dataset columns: {missing_dataset}"
    )

if missing_split:
    raise RuntimeError(
        f"Missing required split columns: {missing_split}"
    )


# =============================================================================
# CHECK SUBJECT SPLIT
# =============================================================================

subject_split = (
    split_df[["subject", "split"]]
    .drop_duplicates()
)

duplicate_subjects = (
    subject_split
    .groupby("subject")["split"]
    .nunique()
)

bad_subjects = duplicate_subjects[
    duplicate_subjects > 1
]

if len(bad_subjects) > 0:
    raise RuntimeError(
        "Some subjects belong to multiple splits."
    )


valid_splits = {"train", "validation", "test"}

unknown_splits = set(subject_split["split"]) - valid_splits

if unknown_splits:
    raise RuntimeError(
        f"Unknown split labels: {unknown_splits}"
    )


# =============================================================================
# MERGE SPLIT INFORMATION
# =============================================================================

df = df.drop(columns=["split"], errors="ignore")

df = df.merge(
    subject_split,
    on="subject",
    how="left",
    validate="many_to_one",
)

if df["split"].isna().any():
    missing_subjects = sorted(
        df.loc[df["split"].isna(), "subject"].unique()
    )

    raise RuntimeError(
        "Some dataset subjects have no split assignment: "
        f"{missing_subjects}"
    )


# =============================================================================
# SUBJECT-LEVEL LEAKAGE CHECK
# =============================================================================

train_subjects = set(
    df.loc[df["split"] == "train", "subject"]
)

validation_subjects = set(
    df.loc[df["split"] == "validation", "subject"]
)

test_subjects = set(
    df.loc[df["split"] == "test", "subject"]
)

if train_subjects & validation_subjects:
    raise RuntimeError("TRAIN / VALIDATION leakage detected.")

if train_subjects & test_subjects:
    raise RuntimeError("TRAIN / TEST leakage detected.")

if validation_subjects & test_subjects:
    raise RuntimeError("VALIDATION / TEST leakage detected.")

print()
print("=" * 80)
print("SUBJECT-LEVEL LEAKAGE CHECK")
print("=" * 80)

print(f"Train subjects:      {len(train_subjects)}")
print(f"Validation subjects: {len(validation_subjects)}")
print(f"Test subjects:       {len(test_subjects)}")

print("Train ∩ Validation:", len(train_subjects & validation_subjects))
print("Train ∩ Test:       ", len(train_subjects & test_subjects))
print("Validation ∩ Test:  ", len(validation_subjects & test_subjects))

print()
print("LEAKAGE STATUS: PASS")


# =============================================================================
# FEATURE SELECTION
# =============================================================================

metadata_columns = {
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

candidate_features = [
    c for c in df.columns
    if c not in metadata_columns
]


# =============================================================================
# KEEP ONLY NUMERIC FEATURES
# =============================================================================

numeric_features = []

for col in candidate_features:
    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_features.append(col)

if len(numeric_features) == 0:
    raise RuntimeError(
        "No numeric EEG feature columns were found."
    )

X = df[numeric_features].copy()

print()
print("=" * 80)
print("FEATURE SET")
print("=" * 80)

print(f"Numeric features: {len(numeric_features)}")

print()
print("First features:")
print(numeric_features[:20])


# =============================================================================
# NUMERIC QC
# =============================================================================

X = X.replace([np.inf, -np.inf], np.nan)

nan_count = int(X.isna().sum().sum())

print()
print("=" * 80)
print("FEATURE QC")
print("=" * 80)

print(f"NaN values: {nan_count:,}")
print("Inf values: 0")

# We do not delete rows.
# Imputation is performed inside the ML pipeline using training data.


# =============================================================================
# TARGET PREPARATION
# =============================================================================

targets = {
    "remember": "target_remember",
    "correct": "target_correct",
}


def normalize_binary_target(series, name):

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")

        unique_values = set(
            values.dropna().unique().tolist()
        )

        if not unique_values.issubset({0, 1}):
            raise RuntimeError(
                f"{name} contains values other than 0/1: "
                f"{sorted(unique_values)}"
            )

        return values.astype("Int64")

    mapped = (
        series.astype(str)
        .str.lower()
        .str.strip()
        .map({
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        })
    )

    return mapped.astype("Int64")


for target_name, target_col in targets.items():

    df[target_col] = normalize_binary_target(
        df[target_col],
        target_col,
    )


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

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
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )
        ),
    ]),

    "random_forest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        ),
    ]),
}


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(y_true, y_pred, y_prob):

    result = {}

    result["accuracy"] = accuracy_score(
        y_true,
        y_pred,
    )

    result["balanced_accuracy"] = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    result["precision"] = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    result["recall"] = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    result["f1"] = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    try:
        result["roc_auc"] = roc_auc_score(
            y_true,
            y_prob,
        )
    except Exception:
        result["roc_auc"] = np.nan

    return result


# =============================================================================
# RUN MODELS
# =============================================================================

all_results = []
confusion_results = []

for target_name, target_col in targets.items():

    print()
    print("=" * 80)
    print(f"TARGET: {target_name.upper()}")
    print("=" * 80)

    valid_mask = df[target_col].notna()

    target_df = df.loc[valid_mask].copy()

    X_target = X.loc[valid_mask].copy()

    y = target_df[target_col].astype(int)

    train_mask = target_df["split"].eq("train")
    validation_mask = target_df["split"].eq("validation")
    test_mask = target_df["split"].eq("test")

    X_train = X_target.loc[train_mask]
    X_validation = X_target.loc[validation_mask]
    X_test = X_target.loc[test_mask]

    y_train = y.loc[train_mask]
    y_validation = y.loc[validation_mask]
    y_test = y.loc[test_mask]

    print()
    print("ROWS")
    print("-" * 80)
    print(f"Train:      {len(X_train):,}")
    print(f"Validation: {len(X_validation):,}")
    print(f"Test:       {len(X_test):,}")

    print()
    print("TARGET DISTRIBUTION")
    print("-" * 80)

    print(
        "Train:",
        y_train.value_counts().sort_index().to_dict()
    )

    print(
        "Validation:",
        y_validation.value_counts().sort_index().to_dict()
    )

    print(
        "Test:",
        y_test.value_counts().sort_index().to_dict()
    )

    if y_train.nunique() < 2:
        raise RuntimeError(
            f"Training target '{target_name}' has only one class."
        )

    if y_validation.nunique() < 2:
        raise RuntimeError(
            f"Validation target '{target_name}' has only one class."
        )

    if y_test.nunique() < 2:
        raise RuntimeError(
            f"Test target '{target_name}' has only one class."
        )


    # -------------------------------------------------------------------------
    # EACH MODEL
    # -------------------------------------------------------------------------

    for model_name, model in models.items():

        print()
        print("-" * 80)
        print(f"MODEL: {model_name}")
        print("-" * 80)

        # -------------------------------------------------------------
        # TRAIN
        # -------------------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # -------------------------------------------------------------
        # VALIDATION
        # -------------------------------------------------------------

        validation_pred = model.predict(
            X_validation
        )

        validation_prob = model.predict_proba(
            X_validation
        )[:, 1]

        validation_metrics = calculate_metrics(
            y_validation,
            validation_pred,
            validation_prob,
        )

        # -------------------------------------------------------------
        # TEST
        # -------------------------------------------------------------

        test_pred = model.predict(
            X_test
        )

        test_prob = model.predict_proba(
            X_test
        )[:, 1]

        test_metrics = calculate_metrics(
            y_test,
            test_pred,
            test_prob,
        )

        # -------------------------------------------------------------
        # SAVE RESULTS
        # -------------------------------------------------------------

        for metric_name, value in validation_metrics.items():

            all_results.append({
                "target": target_name,
                "target_column": target_col,
                "model": model_name,
                "split": "validation",
                "metric": metric_name,
                "value": value,
                "n_rows": len(X_validation),
                "n_subjects": validation_subjects.__len__(),
            })

        for metric_name, value in test_metrics.items():

            all_results.append({
                "target": target_name,
                "target_column": target_col,
                "model": model_name,
                "split": "test",
                "metric": metric_name,
                "value": value,
                "n_rows": len(X_test),
                "n_subjects": test_subjects.__len__(),
            })


        # -------------------------------------------------------------
        # CONFUSION MATRICES
        # -------------------------------------------------------------

        val_cm = confusion_matrix(
            y_validation,
            validation_pred,
            labels=[0, 1],
        )

        test_cm = confusion_matrix(
            y_test,
            test_pred,
            labels=[0, 1],
        )

        confusion_results.extend([
            {
                "target": target_name,
                "model": model_name,
                "split": "validation",
                "TN": int(val_cm[0, 0]),
                "FP": int(val_cm[0, 1]),
                "FN": int(val_cm[1, 0]),
                "TP": int(val_cm[1, 1]),
            },
            {
                "target": target_name,
                "model": model_name,
                "split": "test",
                "TN": int(test_cm[0, 0]),
                "FP": int(test_cm[0, 1]),
                "FN": int(test_cm[1, 0]),
                "TP": int(test_cm[1, 1]),
            },
        ])


        # -------------------------------------------------------------
        # PRINT
        # -------------------------------------------------------------

        print()
        print("VALIDATION")
        print(
            f"Accuracy:           {validation_metrics['accuracy']:.4f}"
        )
        print(
            f"Balanced Accuracy:  {validation_metrics['balanced_accuracy']:.4f}"
        )
        print(
            f"Precision:          {validation_metrics['precision']:.4f}"
        )
        print(
            f"Recall:             {validation_metrics['recall']:.4f}"
        )
        print(
            f"F1:                 {validation_metrics['f1']:.4f}"
        )
        print(
            f"ROC-AUC:            {validation_metrics['roc_auc']:.4f}"
        )

        print()
        print("TEST")
        print(
            f"Accuracy:           {test_metrics['accuracy']:.4f}"
        )
        print(
            f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}"
        )
        print(
            f"Precision:          {test_metrics['precision']:.4f}"
        )
        print(
            f"Recall:             {test_metrics['recall']:.4f}"
        )
        print(
            f"F1:                 {test_metrics['f1']:.4f}"
        )
        print(
            f"ROC-AUC:            {test_metrics['roc_auc']:.4f}"
        )


# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df = pd.DataFrame(all_results)

confusion_df = pd.DataFrame(confusion_results)

results_path = (
    OUTPUT_DIR
    / "baseline_model_metrics.csv"
)

confusion_path = (
    OUTPUT_DIR
    / "baseline_confusion_matrices.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)

confusion_df.to_csv(
    confusion_path,
    index=False,
)


# =============================================================================
# BEST MODEL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("BASELINE MODELING COMPLETE")
print("=" * 80)

if not results_df.empty:

    test_results = results_df[
        results_df["split"] == "test"
    ].copy()

    best = (
        test_results[
            test_results["metric"] == "roc_auc"
        ]
        .sort_values(
            "value",
            ascending=False,
        )
        .head(1)
    )

    if not best.empty:

        row = best.iloc[0]

        print()
        print("BEST TEST ROC-AUC")
        print("-" * 80)
        print(f"Target:  {row['target']}")
        print(f"Model:   {row['model']}")
        print(f"ROC-AUC: {row['value']:.4f}")


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(results_path)
print(confusion_path)

print()
print("=" * 80)
print("STATUS: PASS - BASELINE ML MODELS COMPLETED")
print("=" * 80)