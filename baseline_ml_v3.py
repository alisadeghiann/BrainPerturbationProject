# =============================================================================
# BASELINE ML V3
# Scientific EEG Features - Selected 55 Features
#
# PURPOSE:
#   Final clean baseline before What-if / Perturbation Analysis
#
# TARGETS:
#   1) remember
#   2) correct
#
# MODELS:
#   - Logistic Regression
#   - Random Forest
#
# IMPORTANT:
#   - Subject-level split
#   - No target-derived predictors
#   - No hyperparameter tuning
#   - No SHAP
#   - No perturbation yet
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "ml_ready_v2"
    / "feature_selection"
    / "ml_ready_dataset_v2_selected.csv"
)

SPLIT_FILE = (
    BASE
    / "features"
    / "ml_ready_v2"
    / "split"
    / "subject_level_split_v2.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "ml_results"
    / "baseline_v3"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_OUT = OUTPUT_DIR / "baseline_v3_metrics.csv"
CM_OUT = OUTPUT_DIR / "baseline_v3_confusion_matrices.csv"


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("BASELINE ML V3")
print("=" * 80)

print(f"Input dataset:")
print(INPUT)

print(f"Split file:")
print(SPLIT_FILE)

if not INPUT.exists():
    raise FileNotFoundError(f"Input dataset not found:\n{INPUT}")

if not SPLIT_FILE.exists():
    raise FileNotFoundError(f"Split file not found:\n{SPLIT_FILE}")

df = pd.read_csv(INPUT)
split_df = pd.read_csv(SPLIT_FILE)

print()
print("=" * 80)
print("DATASET")
print("=" * 80)

print(f"Rows:       {len(df):,}")
print(f"Columns:    {len(df.columns)}")
print(f"Subjects:   {df['subject'].nunique()}")
print(f"Runs:       {df['run'].nunique()}")


# =============================================================================
# CHECK REQUIRED COLUMNS
# =============================================================================

required_columns = [
    "subject",
    "run",
    "epoch",
    "target_remember",
    "target_correct",
    "target_label",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# =============================================================================
# DETECT SCIENTIFIC FEATURES
# =============================================================================

NON_FEATURE_COLUMNS = {
    "subject",
    "run",
    "epoch",
    "file",
    "trial",
    "target_label",
    "target_remember",
    "target_correct",

    # target-derived / behavioral variables
    "is_correct",
    "is_remembered",
    "is_ignored",

    "behavior_label",
    "behavior_outcome",
    "probe_type",
    "probe_letter",
    "feedback",
    "response_type",
    "memory_cond",
    "event_name",
    "event_code",
    "event_sequence",
    "event_source",
    "alignment_status",
}


feature_candidates = []

for col in df.columns:

    if col in NON_FEATURE_COLUMNS:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        feature_candidates.append(col)


# =============================================================================
# TARGET-LEAKAGE PROTECTION
# =============================================================================

TARGET_KEYWORDS = [
    "target",
    "remember",
    "ignore",
    "correct",
    "incorrect",
    "behavior",
    "response",
    "feedback",
    "probe",
]

target_like_features = []

for col in feature_candidates:

    name = col.lower()

    if any(keyword in name for keyword in TARGET_KEYWORDS):
        target_like_features.append(col)


if target_like_features:

    print()
    print("=" * 80)
    print("TARGET-LIKE FEATURES DETECTED")
    print("=" * 80)

    for col in target_like_features:
        print(col)

    raise RuntimeError(
        "STOP: target-like variables detected among predictors."
    )


FEATURES = feature_candidates


# =============================================================================
# FEATURE SUMMARY
# =============================================================================

print()
print("=" * 80)
print("SCIENTIFIC FEATURES")
print("=" * 80)

print(f"Scientific predictors: {len(FEATURES)}")

print()
print(FEATURES)


# =============================================================================
# EXPECTED FEATURE COUNT
# =============================================================================

if len(FEATURES) != 55:

    raise RuntimeError(
        f"Expected exactly 55 selected scientific features, "
        f"but found {len(FEATURES)}."
    )


# =============================================================================
# BASIC FEATURE QC
# =============================================================================

X_all = df[FEATURES]

nan_count = int(X_all.isna().sum().sum())

numeric_array = X_all.to_numpy(dtype=np.float64)

inf_count = int(np.isinf(numeric_array).sum())

duplicate_keys = int(
    df.duplicated(
        subset=["subject", "run", "epoch"]
    ).sum()
)

print()
print("=" * 80)
print("FEATURE QC")
print("=" * 80)

print(f"NaN values:       {nan_count}")
print(f"Inf values:       {inf_count}")
print(f"Duplicate keys:   {duplicate_keys}")


if nan_count != 0:
    raise RuntimeError("NaN values detected.")

if inf_count != 0:
    raise RuntimeError("Inf values detected.")

if duplicate_keys != 0:
    raise RuntimeError("Duplicate subject/run/epoch keys detected.")


# =============================================================================
# LOAD SUBJECT-LEVEL SPLIT
# =============================================================================

if "split" not in split_df.columns:

    raise RuntimeError(
        "Split file does not contain a 'split' column."
    )

split_columns = [
    "subject",
    "split",
]

missing_split_cols = [
    c for c in split_columns
    if c not in split_df.columns
]

if missing_split_cols:

    raise RuntimeError(
        "Missing columns in split file:\n"
        + "\n".join(missing_split_cols)
    )


# Keep only unique subject -> split assignments
subject_split = (
    split_df[["subject", "split"]]
    .drop_duplicates()
)


# Check every subject has exactly one split
split_counts = (
    subject_split
    .groupby("subject")["split"]
    .nunique()
)

if (split_counts > 1).any():

    bad_subjects = split_counts[
        split_counts > 1
    ].index.tolist()

    raise RuntimeError(
        "Some subjects occur in multiple splits:\n"
        + str(bad_subjects)
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

    missing_subjects = (
        df.loc[
            df["split"].isna(),
            "subject"
        ]
        .unique()
        .tolist()
    )

    raise RuntimeError(
        "Subjects without split assignment:\n"
        + str(missing_subjects)
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

print()
print("=" * 80)
print("SUBJECT-LEVEL LEAKAGE CHECK")
print("=" * 80)

print(f"Train subjects:       {len(train_subjects)}")
print(f"Validation subjects:  {len(validation_subjects)}")
print(f"Test subjects:        {len(test_subjects)}")

print(
    f"Train ∩ Validation:   "
    f"{len(train_subjects & validation_subjects)}"
)

print(
    f"Train ∩ Test:         "
    f"{len(train_subjects & test_subjects)}"
)

print(
    f"Validation ∩ Test:    "
    f"{len(validation_subjects & test_subjects)}"
)


if train_subjects & validation_subjects:
    raise RuntimeError("Subject leakage: train/validation overlap.")

if train_subjects & test_subjects:
    raise RuntimeError("Subject leakage: train/test overlap.")

if validation_subjects & test_subjects:
    raise RuntimeError("Subject leakage: validation/test overlap.")


print()
print("LEAKAGE STATUS: PASS")


# =============================================================================
# ROW SPLIT
# =============================================================================

train = df[df["split"] == "train"].copy()
validation = df[df["split"] == "validation"].copy()
test = df[df["split"] == "test"].copy()

print()
print("=" * 80)
print("ROW DISTRIBUTION")
print("=" * 80)

print(
    df["split"]
    .value_counts()
)


# =============================================================================
# TARGET DEFINITIONS
# =============================================================================

TARGETS = {
    "remember": "target_remember",
    "correct": "target_correct",
}


# =============================================================================
# MODELS
# =============================================================================

models = {

    "logistic_regression": Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42
                )
            ),
        ]
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),
}


# =============================================================================
# STORAGE
# =============================================================================

metrics_rows = []
confusion_rows = []


# =============================================================================
# MODEL EVALUATION FUNCTION
# =============================================================================

def evaluate_model(
    model,
    X_train,
    y_train,
    X_eval,
    y_eval,
):

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_eval
    )

    probabilities = model.predict_proba(
        X_eval
    )[:, 1]

    accuracy = accuracy_score(
        y_eval,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_eval,
        predictions
    )

    precision = precision_score(
        y_eval,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_eval,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_eval,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_eval,
        probabilities
    )

    cm = confusion_matrix(
        y_eval,
        predictions,
        labels=[0, 1]
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


# =============================================================================
# RUN MODELS
# =============================================================================

for target_name, target_column in TARGETS.items():

    print()
    print("=" * 80)
    print(f"TARGET: {target_name.upper()}")
    print("=" * 80)

    y_train = train[target_column].astype(int)
    y_validation = validation[target_column].astype(int)
    y_test = test[target_column].astype(int)

    X_train = train[FEATURES]
    X_validation = validation[FEATURES]
    X_test = test[FEATURES]

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

    for model_name, model in models.items():

        print()
        print("-" * 80)
        print(f"MODEL: {model_name}")
        print("-" * 80)

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------

        validation_result = evaluate_model(
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )

        print()
        print("VALIDATION")

        print(
            f"Accuracy            "
            f"{validation_result['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy   "
            f"{validation_result['balanced_accuracy']:.4f}"
        )

        print(
            f"Precision           "
            f"{validation_result['precision']:.4f}"
        )

        print(
            f"Recall              "
            f"{validation_result['recall']:.4f}"
        )

        print(
            f"F1                  "
            f"{validation_result['f1']:.4f}"
        )

        print(
            f"ROC-AUC             "
            f"{validation_result['roc_auc']:.4f}"
        )

        metrics_rows.append({
            "target": target_name,
            "model": model_name,
            "split": "validation",
            "accuracy": validation_result["accuracy"],
            "balanced_accuracy": validation_result["balanced_accuracy"],
            "precision": validation_result["precision"],
            "recall": validation_result["recall"],
            "f1": validation_result["f1"],
            "roc_auc": validation_result["roc_auc"],
        })

        cm = validation_result["confusion_matrix"]

        confusion_rows.extend([
            {
                "target": target_name,
                "model": model_name,
                "split": "validation",
                "actual": 0,
                "predicted": 0,
                "count": int(cm[0, 0]),
            },
            {
                "target": target_name,
                "model": model_name,
                "split": "validation",
                "actual": 0,
                "predicted": 1,
                "count": int(cm[0, 1]),
            },
            {
                "target": target_name,
                "model": model_name,
                "split": "validation",
                "actual": 1,
                "predicted": 0,
                "count": int(cm[1, 0]),
            },
            {
                "target": target_name,
                "model": model_name,
                "split": "validation",
                "actual": 1,
                "predicted": 1,
                "count": int(cm[1, 1]),
            },
        ])

        # ---------------------------------------------------------------------
        # TEST
        # ---------------------------------------------------------------------

        test_result = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        print()
        print("TEST")

        print(
            f"Accuracy            "
            f"{test_result['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy   "
            f"{test_result['balanced_accuracy']:.4f}"
        )

        print(
            f"Precision           "
            f"{test_result['precision']:.4f}"
        )

        print(
            f"Recall              "
            f"{test_result['recall']:.4f}"
        )

        print(
            f"F1                  "
            f"{test_result['f1']:.4f}"
        )

        print(
            f"ROC-AUC             "
            f"{test_result['roc_auc']:.4f}"
        )

        metrics_rows.append({
            "target": target_name,
            "model": model_name,
            "split": "test",
            "accuracy": test_result["accuracy"],
            "balanced_accuracy": test_result["balanced_accuracy"],
            "precision": test_result["precision"],
            "recall": test_result["recall"],
            "f1": test_result["f1"],
            "roc_auc": test_result["roc_auc"],
        })

        cm = test_result["confusion_matrix"]

        confusion_rows.extend([
            {
                "target": target_name,
                "model": model_name,
                "split": "test",
                "actual": 0,
                "predicted": 0,
                "count": int(cm[0, 0]),
            },
            {
                "target": target_name,
                "model": model_name,
                "split": "test",
                "actual": 0,
                "predicted": 1,
                "count": int(cm[0, 1]),
            },
            {
                "target": target_name,
                "model": model_name,
                "split": "test",
                "actual": 1,
                "predicted": 0,
                "count": int(cm[1, 0]),
            },
            {
                "target": target_name,
                "model": model_name,
                "split": "test",
                "actual": 1,
                "predicted": 1,
                "count": int(cm[1, 1]),
            },
        ])


# =============================================================================
# SAVE RESULTS
# =============================================================================

metrics_df = pd.DataFrame(
    metrics_rows
)

confusion_df = pd.DataFrame(
    confusion_rows
)

metrics_df.to_csv(
    METRICS_OUT,
    index=False
)

confusion_df.to_csv(
    CM_OUT,
    index=False
)


# =============================================================================
# BEST TEST RESULTS
# =============================================================================

best_test = (
    metrics_df[
        metrics_df["split"] == "test"
    ]
    .sort_values(
        "roc_auc",
        ascending=False
    )
)

print()
print("=" * 80)
print("BEST TEST RESULTS")
print("=" * 80)

print(
    best_test[
        [
            "target",
            "model",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "roc_auc",
        ]
    ].to_string(index=False)
)


# =============================================================================
# FINAL STATUS
# =============================================================================

print()
print("=" * 80)
print("BASELINE ML V3 COMPLETE")
print("=" * 80)

print()
print("Scientific predictors:", len(FEATURES))
print("Rows:", len(df))
print("Subjects:", df["subject"].nunique())

print()
print("Saved:")
print(METRICS_OUT)
print(CM_OUT)

print()
print("=" * 80)
print("STATUS: PASS - FINAL CLEAN BASELINE CREATED")
print("=" * 80)