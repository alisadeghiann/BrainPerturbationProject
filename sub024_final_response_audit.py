import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "qc"
    / "trial_anomaly_inspection"
    / "sub024_final_behavior_review"
    / "behavior_response_audit"
    / "sub024_behavior_response_audit_full.csv"
)

OUTPUT = (
    BASE
    / "qc"
    / "trial_anomaly_inspection"
    / "sub024_final_behavior_review"
    / "behavior_response_audit"
    / "final_response_check"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("SUB-024 FINAL RESPONSE LOGIC AUDIT")
print("=" * 90)

df = pd.read_csv(INPUT)

print("\nRows:", len(df))

# ================================================================
# 1. IMPORTANT COLUMNS
# ================================================================

important = [
    "subject",
    "source_file",
    "trial",
    "memory_cond",
    "response",
    "feedback",
    "accuracy",
    "has_probe_target",
    "expected_14_events"
]

print("\n" + "=" * 90)
print("1. IMPORTANT COLUMNS")
print("=" * 90)

for c in important:
    if c in df.columns:
        print(f"\n--- {c} ---")
        print("dtype:", df[c].dtype)
        print("unique:", df[c].nunique(dropna=False))
        print(
            df[c]
            .value_counts(dropna=False)
            .head(30)
            .to_string()
        )
    else:
        print("\nMISSING:", c)

# ================================================================
# 2. RESPONSE VALUES
# ================================================================

print("\n" + "=" * 90)
print("2. RESPONSE VALUES")
print("=" * 90)

if "response" in df.columns:
    print(
        df["response"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

# ================================================================
# 3. RESPONSE × FEEDBACK
# ================================================================

print("\n" + "=" * 90)
print("3. RESPONSE × FEEDBACK")
print("=" * 90)

if "response" in df.columns and "feedback" in df.columns:

    table = pd.crosstab(
        df["response"],
        df["feedback"],
        dropna=False
    )

    print(table.to_string())

    table.to_csv(
        OUTPUT / "response_x_feedback.csv"
    )

# ================================================================
# 4. RESPONSE × ACCURACY
# ================================================================

print("\n" + "=" * 90)
print("4. RESPONSE × ACCURACY")
print("=" * 90)

if "response" in df.columns and "accuracy" in df.columns:

    table = pd.crosstab(
        df["response"],
        df["accuracy"],
        dropna=False
    )

    print(table.to_string())

    table.to_csv(
        OUTPUT / "response_x_accuracy.csv"
    )

# ================================================================
# 5. PROBE TARGET
# ================================================================

print("\n" + "=" * 90)
print("5. PROBE TARGET / RESPONSE COLUMNS")
print("=" * 90)

for c in ["has_probe_target", "expected_14_events"]:

    if c in df.columns:

        print("\n---", c, "---")

        print(
            df[c]
            .value_counts(dropna=False)
            .to_string()
        )

# ================================================================
# 6. HAS PROBE TARGET × FEEDBACK
# ================================================================

print("\n" + "=" * 90)
print("6. HAS_PROBE_TARGET × FEEDBACK")
print("=" * 90)

if "has_probe_target" in df.columns and "feedback" in df.columns:

    table = pd.crosstab(
        df["has_probe_target"],
        df["feedback"],
        dropna=False
    )

    print(table.to_string())

# ================================================================
# 7. MEMORY × RESPONSE × FEEDBACK
# ================================================================

print("\n" + "=" * 90)
print("7. MEMORY × RESPONSE × FEEDBACK")
print("=" * 90)

if (
    "memory_cond" in df.columns
    and "response" in df.columns
    and "feedback" in df.columns
):

    table = pd.crosstab(
        [
            df["memory_cond"],
            df["response"]
        ],
        df["feedback"],
        dropna=False
    )

    print(table.to_string())

    table.to_csv(
        OUTPUT / "memory_response_feedback.csv"
    )

# ================================================================
# 8. CHECK FEEDBACK VS NUMERIC ACCURACY
# ================================================================

print("\n" + "=" * 90)
print("8. FEEDBACK VS NUMERIC ACCURACY CONSISTENCY")
print("=" * 90)

if "feedback" in df.columns and "accuracy" in df.columns:

    df["_feedback_expected_accuracy"] = (
        df["feedback"]
        .astype(str)
        .str.lower()
        .map({
            "correct": 1.0,
            "wrong": 0.0
        })
    )

    df["_accuracy_match"] = (
        df["accuracy"]
        == df["_feedback_expected_accuracy"]
    )

    print(
        df["_accuracy_match"]
        .value_counts(dropna=False)
        .to_string()
    )

    mismatches = df[
        df["_accuracy_match"] == False
    ].copy()

    print("\nNumber of mismatches:", len(mismatches))

    if len(mismatches) > 0:
        print("\nMISMATCHES:")
        print(
            mismatches[
                [
                    c for c in [
                        "source_file",
                        "trial",
                        "memory_cond",
                        "response",
                        "feedback",
                        "accuracy"
                    ]
                    if c in mismatches.columns
                ]
            ].to_string(index=False)
        )

        mismatches.to_csv(
            OUTPUT / "feedback_accuracy_mismatches.csv",
            index=False
        )

# ================================================================
# 9. COMPLETE BEHAVIOR TABLE
# ================================================================

print("\n" + "=" * 90)
print("9. FIRST 20 TRIALS")
print("=" * 90)

cols = [
    c for c in [
        "source_file",
        "trial",
        "memory_cond",
        "response",
        "feedback",
        "accuracy",
        "has_probe_target",
        "expected_14_events"
    ]
    if c in df.columns
]

print(
    df[cols]
    .head(20)
    .to_string(index=False)
)

# ================================================================
# 10. FINAL INTERPRETATION
# ================================================================

print("\n" + "=" * 90)
print("10. FINAL NUMERIC SUMMARY")
print("=" * 90)

correct = (
    pd.to_numeric(df["accuracy"], errors="coerce") == 1
).sum()

wrong = (
    pd.to_numeric(df["accuracy"], errors="coerce") == 0
).sum()

print("Trials:", len(df))
print("Correct:", correct)
print("Wrong:", wrong)
print("Accuracy:", round(correct / len(df) * 100, 2), "%")

if "feedback" in df.columns:

    feedback_correct = (
        df["feedback"]
        .astype(str)
        .str.lower()
        .eq("correct")
        .sum()
    )

    feedback_wrong = (
        df["feedback"]
        .astype(str)
        .str.lower()
        .eq("wrong")
        .sum()
    )

    print("Feedback correct:", feedback_correct)
    print("Feedback wrong:", feedback_wrong)

print("\nOutput:")
print(OUTPUT)

print("\nNO EEG FILES MODIFIED.")
print("NO EPOCHS DELETED.")
print("NO SUBJECTS DELETED.")
print("NO DATA EXCLUDED.")

