import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = BASE / "qc" / "trial_table" / "TRIAL_LEVEL_TABLE.csv"
OUTPUT_DIR = BASE / "qc" / "trial_anomaly_inspection" / "sub024_final_behavior_review"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SUB-024 FINAL BEHAVIORAL REVIEW")
print("=" * 80)

print("\nReading:")
print(INPUT)

df = pd.read_csv(INPUT)

# ------------------------------------------------------------------
# FIND SUBJECT COLUMN
# ------------------------------------------------------------------

if "subject" not in df.columns:
    raise RuntimeError(
        "Column 'subject' was not found.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )

sub = df[
    df["subject"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("sub-024")
].copy()

print("\nTotal SUB-024 rows:", len(sub))

if len(sub) == 0:
    raise RuntimeError("No rows found for sub-024.")

# ------------------------------------------------------------------
# SHOW COLUMNS
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("COLUMNS")
print("=" * 80)

for i, c in enumerate(sub.columns, 1):
    print(f"{i:03d} | {c}")

# ------------------------------------------------------------------
# FIND RELEVANT COLUMNS AUTOMATICALLY
# ------------------------------------------------------------------

def find_columns(words):
    return [
        c for c in sub.columns
        if any(w in c.lower() for w in words)
    ]

behavior_cols = find_columns([
    "correct",
    "accuracy",
    "response",
    "answer",
    "result",
    "memory"
])

event_cols = find_columns([
    "event",
    "count"
])

run_cols = find_columns([
    "run"
])

trial_cols = find_columns([
    "trial"
])

print("\n" + "=" * 80)
print("RELEVANT COLUMNS")
print("=" * 80)

print("Behavior-related:")
print(behavior_cols)

print("\nEvent-related:")
print(event_cols)

print("\nRun-related:")
print(run_cols)

print("\nTrial-related:")
print(trial_cols)

# ------------------------------------------------------------------
# SAVE ALL SUB-024 TRIALS
# ------------------------------------------------------------------

all_path = OUTPUT_DIR / "sub024_all_75_trials.csv"
sub.to_csv(all_path, index=False)

print("\nSaved:")
print(all_path)

# ------------------------------------------------------------------
# CORRECT / WRONG DETECTION
# ------------------------------------------------------------------

result_col = None

priority = [
    "correct",
    "accuracy",
    "result",
    "response_correct"
]

for candidate in priority:
    matches = [
        c for c in sub.columns
        if c.lower() == candidate
    ]
    if matches:
        result_col = matches[0]
        break

if result_col is None:
    for c in sub.columns:
        vals = (
            sub[c]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )

        if (
            "correct" in vals
            and "wrong" in vals
        ):
            result_col = c
            break

print("\nDetected result column:")
print(result_col)

if result_col is None:
    print("\nWARNING:")
    print("Could not automatically identify correct/wrong column.")
else:

    # Normalize result
    result = (
        sub[result_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print("\n" + "=" * 80)
    print("BEHAVIORAL RESULT COUNTS")
    print("=" * 80)

    print(result.value_counts(dropna=False).to_string())

    correct = (result == "correct").sum()
    wrong = (result == "wrong").sum()

    total_valid = correct + wrong

    if total_valid > 0:
        accuracy = correct / total_valid
    else:
        accuracy = float("nan")

    print("\nCorrect:", correct)
    print("Wrong:  ", wrong)
    print("Valid:  ", total_valid)

    print(
        "Accuracy:",
        accuracy,
        f"({accuracy * 100:.2f}%)"
    )

    # --------------------------------------------------------------
    # SAVE CORRECT / WRONG TRIALS
    # --------------------------------------------------------------

    correct_path = OUTPUT_DIR / "sub024_correct_trials.csv"
    wrong_path = OUTPUT_DIR / "sub024_wrong_trials.csv"

    sub[result == "correct"].to_csv(correct_path, index=False)
    sub[result == "wrong"].to_csv(wrong_path, index=False)

    print("\nSaved:")
    print(correct_path)
    print(wrong_path)

# ------------------------------------------------------------------
# RUN-LEVEL ACCURACY
# ------------------------------------------------------------------

if run_cols:
    run_col = run_cols[0]

    print("\n" + "=" * 80)
    print("RUN-LEVEL ANALYSIS")
    print("=" * 80)

    if result_col is not None:

        tmp = sub.copy()

        tmp["_result"] = (
            tmp[result_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        run_summary = (
            tmp.groupby(run_col)["_result"]
            .agg(
                trials="size",
                correct=lambda x: (x == "correct").sum(),
                wrong=lambda x: (x == "wrong").sum()
            )
            .reset_index()
        )

        run_summary["accuracy_percent"] = (
            run_summary["correct"]
            /
            (run_summary["correct"] + run_summary["wrong"])
            * 100
        )

        print(run_summary.to_string(index=False))

        run_path = OUTPUT_DIR / "sub024_run_accuracy.csv"
        run_summary.to_csv(run_path, index=False)

        print("\nSaved:")
        print(run_path)

# ------------------------------------------------------------------
# MEMORY CONDITION × BEHAVIOR
# ------------------------------------------------------------------

memory_cols = [
    c for c in sub.columns
    if "memory" in c.lower()
]

if memory_cols and result_col is not None:

    memory_col = memory_cols[0]

    print("\n" + "=" * 80)
    print("MEMORY CONDITION × BEHAVIOR")
    print("=" * 80)

    tmp = sub.copy()

    tmp["_result"] = (
        tmp[result_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    memory_summary = (
        tmp.groupby(memory_col)["_result"]
        .agg(
            trials="size",
            correct=lambda x: (x == "correct").sum(),
            wrong=lambda x: (x == "wrong").sum()
        )
        .reset_index()
    )

    memory_summary["accuracy_percent"] = (
        memory_summary["correct"]
        /
        (memory_summary["correct"] + memory_summary["wrong"])
        * 100
    )

    print(memory_summary.to_string(index=False))

    memory_path = OUTPUT_DIR / "sub024_memory_accuracy.csv"
    memory_summary.to_csv(memory_path, index=False)

    print("\nSaved:")
    print(memory_path)

# ------------------------------------------------------------------
# EVENT COUNT CHECK
# ------------------------------------------------------------------

count_cols = [
    c for c in sub.columns
    if "event_count" in c.lower()
    or c.lower() == "eventcount"
]

if count_cols:
    count_col = count_cols[0]

    print("\n" + "=" * 80)
    print("EVENT COUNT")
    print("=" * 80)

    print(
        sub[count_col]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

# ------------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL SUB-024 REVIEW SUMMARY")
print("=" * 80)

print("Trials:", len(sub))

if result_col is not None:
    print("Result column:", result_col)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nNO EEG FILES WERE MODIFIED.")
print("NO EPOCHS WERE DELETED.")
print("NO SUBJECTS WERE DELETED.")
