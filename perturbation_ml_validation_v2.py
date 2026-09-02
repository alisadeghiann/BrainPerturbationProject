from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

# ============================================================
# PERTURBATION / WHAT-IF ML VALIDATION V2
# SUBJECT-AWARE VALIDATION
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "ml_validation_v2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("PERTURBATION / WHAT-IF ML VALIDATION V2")
print("=" * 90)
print(f"Project root: {BASE}")

# ============================================================
# SEARCH FOR SUBJECT-LEVEL / EPOCH-LEVEL DATA
# ============================================================

print("\n" + "=" * 90)
print("SEARCHING FOR SUBJECT / EPOCH-LEVEL DATA")
print("=" * 90)

CANDIDATES = [
    BASE
    / "features"
    / "perturbation_analysis"
    / "perturbation_subject_effects.csv",

    BASE
    / "features"
    / "perturbation_analysis"
    / "final_scientific_synthesis_v1"
    / "final_scientific_feature_synthesis_v1.csv",
]

input_file = None

for candidate in CANDIDATES:
    if not candidate.exists():
        continue

    try:
        cols = pd.read_csv(candidate, nrows=2).columns.tolist()

        if (
            "subject" in cols
            and (
                "target_remember" in cols
                or "target_correct" in cols
            )
        ):
            input_file = candidate
            break

    except Exception:
        pass

# Recursive fallback search
if input_file is None:

    print("Primary candidates not suitable.")
    print("Running recursive search...")

    for csv_file in BASE.rglob("*.csv"):

        try:
            cols = pd.read_csv(
                csv_file,
                nrows=2
            ).columns.tolist()

            if (
                "subject" in cols
                and (
                    "target_remember" in cols
                    or "target_correct" in cols
                )
            ):
                input_file = csv_file
                break

        except Exception:
            continue

if input_file is None:
    raise FileNotFoundError(
        "No subject/epoch-level dataset containing "
        "'subject' and target_remember/target_correct "
        "was found."
    )

print("\nINPUT FILE FOUND:")
print(input_file)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(input_file)

print("\n" + "=" * 90)
print("INPUT VALIDATION")
print("=" * 90)

print(f"Rows:       {len(df):,}")
print(f"Columns:    {len(df.columns)}")
print(f"Subjects:   {df['subject'].nunique()}")

# ============================================================
# TARGET DETECTION
# ============================================================

targets = []

if "target_remember" in df.columns:
    targets.append("remember")

if "target_correct" in df.columns:
    targets.append("correct")

print("\nAvailable targets:")
print(targets)

if not targets:
    raise ValueError(
        "Neither target_remember nor target_correct "
        "was found."
    )

# ============================================================
# FEATURE DETECTION
# ============================================================

META_COLUMNS = {
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

FEATURE_COLUMNS = []

for column in df.columns:

    if column in META_COLUMNS:
        continue

    if not pd.api.types.is_numeric_dtype(df[column]):
        continue

    if df[column].nunique(dropna=True) <= 1:
        continue

    FEATURE_COLUMNS.append(column)

print("\n" + "=" * 90)
print("FEATURE VALIDATION")
print("=" * 90)

print(f"Scientific numeric features: {len(FEATURE_COLUMNS)}")

for i, feature in enumerate(FEATURE_COLUMNS, 1):
    print(f"{i:02d}. {feature}")

if len(FEATURE_COLUMNS) == 0:
    raise ValueError(
        "No usable numeric scientific features found."
    )

# ============================================================
# SUBJECT VALIDATION
# ============================================================

subjects = df["subject"].astype(str)

print("\n" + "=" * 90)
print("SUBJECT-LEVEL VALIDATION")
print("=" * 90)

print(f"Unique subjects: {subjects.nunique()}")

subject_counts = subjects.value_counts()

print(
    f"Minimum rows per subject: {subject_counts.min()}"
)

print(
    f"Maximum rows per subject: {subject_counts.max()}"
)

if subjects.nunique() < 5:
    raise ValueError(
        "At least 5 subjects are required "
        "for subject-aware cross-validation."
    )

# ============================================================
# MODELS
# ============================================================

models = {

    "logistic_regression": Pipeline(
        steps=[
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
        ]
    ),

    "random_forest": Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    ),
}

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "balanced_accuracy": make_scorer(
        balanced_accuracy_score
    ),
    "f1": make_scorer(f1_score),
    "roc_auc": "roc_auc",
}

# ============================================================
# RESULTS
# ============================================================

RESULTS = []

# ============================================================
# TARGET LOOP
# ============================================================

for target in targets:

    print("\n" + "=" * 90)
    print(f"TARGET: {target.upper()}")
    print("=" * 90)

    if target == "remember":
        target_column = "target_remember"
    else:
        target_column = "target_correct"

    work = df[
        FEATURE_COLUMNS
        + [target_column, "subject"]
    ].copy()

    work[target_column] = pd.to_numeric(
        work[target_column],
        errors="coerce"
    )

    work["subject"] = work["subject"].astype(str)

    work = work.dropna(
        subset=[
            target_column,
            "subject"
        ]
    )

    X = work[FEATURE_COLUMNS]
    y = work[target_column].astype(int)
    groups = work["subject"]

    print(f"Rows used:     {len(work):,}")
    print(f"Subjects used: {groups.nunique()}")

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    if y.nunique() < 2:
        print(
            "WARNING: Target contains only one class. "
            "Skipping."
        )
        continue

    n_splits = min(
        5,
        groups.nunique()
    )

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    # ========================================================
    # MODEL LOOP
    # ========================================================

    for model_name, model in models.items():

        print("\n" + "-" * 80)
        print(f"MODEL: {model_name}")
        print("-" * 80)

        try:

            scores = cross_validate(
                model,
                X,
                y,
                groups=groups,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                return_train_score=False,
                error_score="raise"
            )

            result = {
                "target": target,
                "model": model_name,
                "n_rows": len(work),
                "n_subjects": groups.nunique(),
                "n_features": len(FEATURE_COLUMNS),
                "cv_folds": n_splits,

                "accuracy_mean":
                    np.mean(
                        scores["test_accuracy"]
                    ),

                "accuracy_std":
                    np.std(
                        scores["test_accuracy"]
                    ),

                "balanced_accuracy_mean":
                    np.mean(
                        scores["test_balanced_accuracy"]
                    ),

                "balanced_accuracy_std":
                    np.std(
                        scores["test_balanced_accuracy"]
                    ),

                "f1_mean":
                    np.mean(
                        scores["test_f1"]
                    ),

                "f1_std":
                    np.std(
                        scores["test_f1"]
                    ),

                "roc_auc_mean":
                    np.mean(
                        scores["test_roc_auc"]
                    ),

                "roc_auc_std":
                    np.std(
                        scores["test_roc_auc"]
                    ),
            }

            RESULTS.append(result)

            print(
                f"Accuracy:          "
                f"{result['accuracy_mean']:.4f} "
                f"+/- {result['accuracy_std']:.4f}"
            )

            print(
                f"Balanced Accuracy: "
                f"{result['balanced_accuracy_mean']:.4f} "
                f"+/- {result['balanced_accuracy_std']:.4f}"
            )

            print(
                f"F1:                "
                f"{result['f1_mean']:.4f} "
                f"+/- {result['f1_std']:.4f}"
            )

            print(
                f"ROC-AUC:           "
                f"{result['roc_auc_mean']:.4f} "
                f"+/- {result['roc_auc_std']:.4f}"
            )

        except Exception as e:

            print(
                f"MODEL FAILED: {model_name}"
            )

            print(
                f"Reason: {type(e).__name__}: {e}"
            )

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(RESULTS)

results_file = (
    OUTPUT_DIR
    / "perturbation_ml_validation_v2_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_file = (
    OUTPUT_DIR
    / "ml_validation_v2_feature_list.csv"
)

pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS
    }
).to_csv(
    feature_file,
    index=False
)

# ============================================================
# QC
# ============================================================

print("\n" + "=" * 90)
print("FINAL ML VALIDATION QC")
print("=" * 90)

print(
    f"Targets analyzed:       {len(targets)}"
)

print(
    f"Features analyzed:      {len(FEATURE_COLUMNS)}"
)

print(
    f"Subjects:               {df['subject'].nunique()}"
)

print(
    f"Result rows:            {len(results_df)}"
)

if len(results_df) > 0:

    numeric_results = results_df.select_dtypes(
        include=np.number
    )

    nan_count = int(
        numeric_results.isna().sum().sum()
    )

    inf_count = int(
        np.isinf(numeric_results).sum().sum()
    )

else:

    nan_count = 0
    inf_count = 0

print(
    f"NaN numeric values:     {nan_count}"
)

print(
    f"Inf numeric values:     {inf_count}"
)

print("\n" + "=" * 90)
print("SAVED")
print("=" * 90)

print(results_file)
print(feature_file)

print("\n" + "=" * 90)
print("PERTURBATION / WHAT-IF ML VALIDATION V2 COMPLETE")
print("=" * 90)

if len(results_df) > 0:
    print(
        "STATUS: PASS - SUBJECT-AWARE ML VALIDATION CREATED"
    )
else:
    print(
        "STATUS: REVIEW_REQUIRED - NO VALID ML RESULTS"
    )