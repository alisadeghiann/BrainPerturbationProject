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


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

INPUT = (
    BASE
    / "features"
    / "ml_ready_v2"
    / "split"
    / "subject_level_split_v2.csv"
)

OUT_DIR = (
    BASE
    / "features"
    / "ml_results"
    / "baseline_v2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_OUTPUT = (
    OUT_DIR
    / "baseline_v2_metrics.csv"
)

CM_OUTPUT = (
    OUT_DIR
    / "baseline_v2_confusion_matrices.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("BASELINE ML V2")
print("=" * 80)

df = pd.read_csv(INPUT)

print(f"Input rows: {len(df):,}")
print(f"Subjects:    {df['subject'].nunique()}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "subject",
    "split",
    "target_remember",
    "target_correct",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# IDENTIFY NUMERIC FEATURES
# ============================================================

EXCLUDE = {
    "target_remember",
    "target_correct",
    "subject",
    "split",
    "file",
    "run",
    "trial",
    "epoch",
    "target_label",
    "behavior_label",
    "probe_type",
    "probe_letter",
    "behavior_outcome",
    "response_type",
    "event_name",
    "event_code",
    "alignment_status",
    "event_source",
    "memory_cond",
    "remember_letters",
    "ignore_letters",
    "event_sequence",
}


numeric_features = []

for col in df.columns:

    if col in EXCLUDE:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_features.append(col)


if len(numeric_features) == 0:
    raise RuntimeError(
        "No numeric predictor features found."
    )


print()
print("=" * 80)
print("FEATURE SET")
print("=" * 80)

print(f"Numeric features: {len(numeric_features)}")

print()
print("Features:")
print(numeric_features)


# ============================================================
# TARGET-DERIVED FEATURE PROTECTION
# ============================================================

for forbidden in [
    "is_correct",
    "is_remembered",
    "is_ignored",
]:

    if forbidden in numeric_features:
        raise RuntimeError(
            f"TARGET-DERIVED FEATURE FOUND: {forbidden}"
        )


print()
print("TARGET-DERIVED FEATURE CHECK: PASS")


# ============================================================
# NUMERIC QC
# ============================================================

X_all = df[numeric_features].copy()

nan_count = X_all.isna().sum().sum()

if nan_count != 0:
    raise RuntimeError(
        f"NaN values detected: {nan_count}"
    )

inf_count = (
    np.isinf(
        X_all.to_numpy(dtype=float)
    ).sum()
)

if inf_count != 0:
    raise RuntimeError(
        f"Inf values detected: {inf_count}"
    )


print()
print("=" * 80)
print("FEATURE QC")
print("=" * 80)

print(f"NaN values: {nan_count}")
print(f"Inf values: {inf_count}")


# ============================================================
# SPLIT DATA
# ============================================================

train = df[df["split"] == "train"].copy()
validation = df[df["split"] == "validation"].copy()
test = df[df["split"] == "test"].copy()


print()
print("=" * 80)
print("DATA SPLITS")
print("=" * 80)

print(f"Train:       {len(train):,}")
print(f"Validation:  {len(validation):,}")
print(f"Test:        {len(test):,}")


# ============================================================
# SUBJECT LEAKAGE CHECK
# ============================================================

train_subjects = set(
    train["subject"].unique()
)

validation_subjects = set(
    validation["subject"].unique()
)

test_subjects = set(
    test["subject"].unique()
)

if train_subjects & validation_subjects:
    raise RuntimeError(
        "Train/Validation subject leakage detected."
    )

if train_subjects & test_subjects:
    raise RuntimeError(
        "Train/Test subject leakage detected."
    )

if validation_subjects & test_subjects:
    raise RuntimeError(
        "Validation/Test subject leakage detected."
    )

print()
print("SUBJECT LEAKAGE CHECK: PASS")


# ============================================================
# MODELS
# ============================================================

models = {

    "logistic_regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                random_state=42
            )
        ),
    ]),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),
}


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
):

    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)[:, 1]

    else:

        probabilities = model.decision_function(X)

    result = {

        "accuracy":
            accuracy_score(
                y,
                predictions
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                predictions
            ),

        "precision":
            precision_score(
                y,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                predictions,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y,
                probabilities
            ),
    }

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    return result, cm


# ============================================================
# RUN BASELINES
# ============================================================

targets = {

    "remember": "target_remember",

    "correct": "target_correct",
}


metrics_records = []
cm_records = []


for target_name, target_column in targets.items():

    print()
    print("=" * 80)
    print(f"TARGET: {target_name.upper()}")
    print("=" * 80)

    y_train = (
        train[target_column]
        .astype(int)
        .to_numpy()
    )

    y_validation = (
        validation[target_column]
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test[target_column]
        .astype(int)
        .to_numpy()
    )

    X_train = train[numeric_features]
    X_validation = validation[numeric_features]
    X_test = test[numeric_features]

    print()
    print("TARGET DISTRIBUTION")

    print(
        "Train:",
        pd.Series(y_train).value_counts().sort_index().to_dict()
    )

    print(
        "Validation:",
        pd.Series(y_validation).value_counts().sort_index().to_dict()
    )

    print(
        "Test:",
        pd.Series(y_test).value_counts().sort_index().to_dict()
    )


    for model_name, model in models.items():

        print()
        print("-" * 80)
        print(f"MODEL: {model_name}")
        print("-" * 80)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_metrics, val_cm = evaluate_model(
            model,
            X_validation,
            y_validation
        )

        print()
        print("VALIDATION")

        for key, value in val_metrics.items():

            print(
                f"{key.replace('_', ' ').title():20s}"
                f"{value:.4f}"
            )


        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        test_metrics, test_cm = evaluate_model(
            model,
            X_test,
            y_test
        )

        print()
        print("TEST")

        for key, value in test_metrics.items():

            print(
                f"{key.replace('_', ' ').title():20s}"
                f"{value:.4f}"
            )


        # ----------------------------------------------------
        # SAVE METRICS
        # ----------------------------------------------------

        val_record = {

            "target":
                target_name,

            "model":
                model_name,

            "split":
                "validation",
        }

        val_record.update(
            val_metrics
        )

        metrics_records.append(
            val_record
        )


        test_record = {

            "target":
                target_name,

            "model":
                model_name,

            "split":
                "test",
        }

        test_record.update(
            test_metrics
        )

        metrics_records.append(
            test_record
        )


        # ----------------------------------------------------
        # SAVE CONFUSION MATRICES
        # ----------------------------------------------------

        for split_name, cm in [
            ("validation", val_cm),
            ("test", test_cm),
        ]:

            cm_records.append({

                "target":
                    target_name,

                "model":
                    model_name,

                "split":
                    split_name,

                "TN":
                    int(cm[0, 0]),

                "FP":
                    int(cm[0, 1]),

                "FN":
                    int(cm[1, 0]),

                "TP":
                    int(cm[1, 1]),
            })


# ============================================================
# SAVE RESULTS
# ============================================================

metrics_df = pd.DataFrame(
    metrics_records
)

cm_df = pd.DataFrame(
    cm_records
)


metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False
)

cm_df.to_csv(
    CM_OUTPUT,
    index=False
)


# ============================================================
# BEST TEST RESULTS
# ============================================================

print()
print("=" * 80)
print("BEST TEST RESULTS")
print("=" * 80)

test_results = (
    metrics_df[
        metrics_df["split"] == "test"
    ]
    .sort_values(
        "roc_auc",
        ascending=False
    )
)

print(
    test_results[
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


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("BASELINE ML V2 COMPLETE")
print("=" * 80)

print()
print("SAVED:")
print(METRICS_OUTPUT)
print(CM_OUTPUT)

print()
print("=" * 80)
print("STATUS: PASS - BASELINE V2 COMPLETED")
print("=" * 80)