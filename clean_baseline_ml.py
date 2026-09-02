from pathlib import Path
import pandas as pd
import numpy as np

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
# CLEAN BASELINE ML
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "ml_ready"
    / "clean"
    / "ml_ready_clean_dataset.csv"
)

SPLIT_FILE = (
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
    / "clean_baseline"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = OUTPUT_DIR / "clean_baseline_metrics.csv"
CONFUSION_FILE = OUTPUT_DIR / "clean_baseline_confusion_matrices.csv"


print("=" * 80)
print("CLEAN BASELINE ML - NO TARGET LEAKAGE")
print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)
split_df = pd.read_csv(SPLIT_FILE)

print(f"Dataset rows: {len(df):,}")
print(f"Split rows:    {len(split_df):,}")
print(f"Subjects:      {df['subject'].nunique()}")


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_targets = [
    "target_remember",
    "target_correct",
]

for col in required_targets:
    if col not in df.columns:
        raise RuntimeError(
            f"Missing required target column: {col}"
        )


if "split" not in split_df.columns:
    raise RuntimeError(
        "Split file does not contain 'split' column."
    )


# ============================================================
# CREATE UNIQUE KEY
# ============================================================

def make_key(data):
    return (
        data["subject"].astype(str)
        + "|"
        + data["run"].astype(str)
        + "|"
        + data["epoch"].astype(str)
    )


df["_key"] = make_key(df)
split_df["_key"] = make_key(split_df)


# ============================================================
# CHECK SPLIT OVERLAP
# ============================================================

print()
print("=" * 80)
print("SUBJECT-LEVEL SPLIT CHECK")
print("=" * 80)

train_subjects = set(
    split_df.loc[
        split_df["split"] == "train",
        "subject"
    ]
)

validation_subjects = set(
    split_df.loc[
        split_df["split"] == "validation",
        "subject"
    ]
)

test_subjects = set(
    split_df.loc[
        split_df["split"] == "test",
        "subject"
    ]
)

print(f"Train subjects:       {len(train_subjects)}")
print(f"Validation subjects:  {len(validation_subjects)}")
print(f"Test subjects:        {len(test_subjects)}")

print(
    f"Train ∩ Validation:  "
    f"{len(train_subjects & validation_subjects)}"
)

print(
    f"Train ∩ Test:        "
    f"{len(train_subjects & test_subjects)}"
)

print(
    f"Validation ∩ Test:   "
    f"{len(validation_subjects & test_subjects)}"
)

if (
    train_subjects & validation_subjects
    or train_subjects & test_subjects
    or validation_subjects & test_subjects
):
    raise RuntimeError(
        "SUBJECT LEAKAGE DETECTED. STOP."
    )

print("LEAKAGE STATUS: PASS")


# ============================================================
# MERGE SPLIT INFORMATION
# ============================================================

split_info = split_df[
    ["_key", "split"]
].drop_duplicates("_key")

df = df.merge(
    split_info,
    on="_key",
    how="inner",
    validate="one_to_one",
)

print()
print(f"Rows after split merge: {len(df):,}")


if len(df) == 0:
    raise RuntimeError(
        "No rows remained after merging split information."
    )


# ============================================================
# IDENTIFY CLEAN EEG FEATURES
# ============================================================

metadata_columns = {
    "file",
    "subject",
    "run",
    "epoch",
    "trial",
    "target_label",
    "target_remember",
    "target_correct",
    "split",
    "_key",
}

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()

feature_columns = [
    c
    for c in numeric_columns
    if c not in metadata_columns
]


# ============================================================
# ABSOLUTE LEAKAGE PROTECTION
# ============================================================

forbidden = [
    "is_correct",
    "is_remembered",
    "is_ignored",
]

remaining_forbidden = [
    c for c in forbidden
    if c in feature_columns
]

if remaining_forbidden:
    raise RuntimeError(
        "LEAKAGE FEATURES STILL PRESENT: "
        + str(remaining_forbidden)
    )


print()
print("=" * 80)
print("CLEAN FEATURE SET")
print("=" * 80)

print(f"Number of features: {len(feature_columns)}")

for feature in feature_columns:
    print(feature)


# ============================================================
# NUMERIC QC
# ============================================================

X_all = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

nan_count = int(
    X_all.isna().sum().sum()
)

inf_count = int(
    np.isinf(
        X_all.to_numpy(dtype=np.float64)
    ).sum()
)

print()
print("=" * 80)
print("FEATURE QC")
print("=" * 80)

print(f"NaN: {nan_count}")
print(f"Inf: {inf_count}")

if nan_count > 0 or inf_count > 0:
    raise RuntimeError(
        "NaN or Inf found in feature matrix."
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X_train,
    y_train,
    X_eval,
    y_eval,
    target_name,
    model_name,
    split_name,
):

    model.fit(X_train, y_train)

    predictions = model.predict(X_eval)

    probabilities = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "target": target_name,
        "model": model_name,
        "split": split_name,
        "rows": len(y_eval),
        "accuracy": accuracy_score(
            y_eval,
            predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_eval,
            predictions
        ),
        "precision": precision_score(
            y_eval,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_eval,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_eval,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_eval,
            probabilities
        ),
    }

    cm = confusion_matrix(
        y_eval,
        predictions,
        labels=[0, 1]
    )

    confusion_record = {
        "target": target_name,
        "model": model_name,
        "split": split_name,
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }

    return metrics, confusion_record


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
                max_iter=2000,
                class_weight="balanced",
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
# TARGETS
# ============================================================

targets = {
    "remember": "target_remember",
    "correct": "target_correct",
}


all_metrics = []
all_confusions = []


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

for target_name, target_column in targets.items():

    print()
    print("=" * 80)
    print(f"TARGET: {target_name.upper()}")
    print("=" * 80)

    train_mask = df["split"] == "train"
    validation_mask = df["split"] == "validation"
    test_mask = df["split"] == "test"

    X_train = X_all.loc[train_mask]
    X_validation = X_all.loc[validation_mask]
    X_test = X_all.loc[test_mask]

    y_train = df.loc[
        train_mask,
        target_column
    ].astype(int)

    y_validation = df.loc[
        validation_mask,
        target_column
    ].astype(int)

    y_test = df.loc[
        test_mask,
        target_column
    ].astype(int)

    print()
    print("ROWS")
    print("-" * 80)

    print(f"Train:       {len(y_train):,}")
    print(f"Validation:  {len(y_validation):,}")
    print(f"Test:        {len(y_test):,}")

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


    # --------------------------------------------------------
    # EACH MODEL
    # --------------------------------------------------------

    for model_name, model in models.items():

        print()
        print("-" * 80)
        print(f"MODEL: {model_name}")
        print("-" * 80)

        # Validation
        validation_metrics, validation_cm = evaluate_model(
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            target_name,
            model_name,
            "validation",
        )

        # Refit on TRAIN + VALIDATION
        X_train_full = pd.concat(
            [X_train, X_validation],
            axis=0
        )

        y_train_full = pd.concat(
            [y_train, y_validation],
            axis=0
        )

        test_metrics, test_cm = evaluate_model(
            model,
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            target_name,
            model_name,
            "test",
        )

        all_metrics.append(
            validation_metrics
        )

        all_metrics.append(
            test_metrics
        )

        all_confusions.append(
            validation_cm
        )

        all_confusions.append(
            test_cm
        )

        print()
        print("VALIDATION")
        print(
            f"Accuracy:           "
            f"{validation_metrics['accuracy']:.4f}"
        )
        print(
            f"Balanced Accuracy:  "
            f"{validation_metrics['balanced_accuracy']:.4f}"
        )
        print(
            f"Precision:          "
            f"{validation_metrics['precision']:.4f}"
        )
        print(
            f"Recall:             "
            f"{validation_metrics['recall']:.4f}"
        )
        print(
            f"F1:                 "
            f"{validation_metrics['f1']:.4f}"
        )
        print(
            f"ROC-AUC:            "
            f"{validation_metrics['roc_auc']:.4f}"
        )

        print()
        print("TEST")
        print(
            f"Accuracy:           "
            f"{test_metrics['accuracy']:.4f}"
        )
        print(
            f"Balanced Accuracy:  "
            f"{test_metrics['balanced_accuracy']:.4f}"
        )
        print(
            f"Precision:          "
            f"{test_metrics['precision']:.4f}"
        )
        print(
            f"Recall:             "
            f"{test_metrics['recall']:.4f}"
        )
        print(
            f"F1:                 "
            f"{test_metrics['f1']:.4f}"
        )
        print(
            f"ROC-AUC:            "
            f"{test_metrics['roc_auc']:.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

metrics_df = pd.DataFrame(all_metrics)

confusion_df = pd.DataFrame(
    all_confusions
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)

confusion_df.to_csv(
    CONFUSION_FILE,
    index=False
)


# ============================================================
# BEST RESULTS
# ============================================================

print()
print("=" * 80)
print("CLEAN BASELINE ML COMPLETE")
print("=" * 80)

print()
print("BEST TEST RESULTS")
print("-" * 80)

best_test = (
    metrics_df[
        metrics_df["split"] == "test"
    ]
    .sort_values(
        "roc_auc",
        ascending=False
    )
)

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


# ============================================================
# SAVE
# ============================================================

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(METRICS_FILE)
print(CONFUSION_FILE)

print()
print("=" * 80)
print("STATUS: PASS - CLEAN BASELINE ML COMPLETED")
print("=" * 80)