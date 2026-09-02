import os
import pandas as pd
import numpy as np

# ============================================================
# SUB-024 BEHAVIORAL QC - READ ONLY
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "qc",
    "trial_anomaly_inspection",
    "sub024_final_behavior_review",
    "sub024_all_75_trials.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "qc",
    "trial_anomaly_inspection",
    "sub024_final_behavior_review_READ_ONLY"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SUB-024 BEHAVIORAL QC - READ ONLY")
print("=" * 80)

print("\nInput CSV:")
print(INPUT_CSV)

print("\nOutput directory:")
print(OUTPUT_DIR)

# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Input CSV was not found:\n{INPUT_CSV}"
    )

print("\nInput file exists: TRUE")

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("\n" + "=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)

print("Rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "memory_cond",
    "has_work_memory",
    "accuracy",
    "response",
    "event_count",
    "expected_14_events",
    "run",
    "trial",
    "bad_trial"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n" +
        "\n".join(missing_columns)
    )

print("\nRequired columns check: PASSED")

# ============================================================
# RELEVANT COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("RELEVANT COLUMNS")
print("=" * 80)

print("\nBehavior-related:")
print([
    "memory_cond",
    "has_work_memory",
    "accuracy",
    "response"
])

print("\nEvent-related:")
print([
    "event_count",
    "expected_14_events"
])

print("\nRun-related:")
print([
    "run"
])

print("\nTrial-related:")
print([
    "trial",
    "bad_trial"
])

# ============================================================
# RESULT COLUMN
# ============================================================

result_col = "accuracy"

print("\nDetected result column:")
print(result_col)

# ============================================================
# IN-MEMORY TYPE CONVERSION
# ============================================================
# IMPORTANT:
# These conversions happen only in memory.
# The original CSV is NOT modified.

df[result_col] = pd.to_numeric(
    df[result_col],
    errors="coerce"
)

df["run"] = pd.to_numeric(
    df["run"],
    errors="coerce"
)

df["memory_cond"] = pd.to_numeric(
    df["memory_cond"],
    errors="coerce"
)

df["event_count"] = pd.to_numeric(
    df["event_count"],
    errors="coerce"
)

# ============================================================
# BEHAVIORAL RESULT COUNTS
# ============================================================

print("\n" + "=" * 80)
print("BEHAVIORAL RESULT COUNTS")
print("=" * 80)

print("\nRaw accuracy distribution:")

print(
    df[result_col]
    .value_counts(dropna=False)
    .sort_index()
)

# 1 = Correct
# 0 = Wrong
# Everything else = Invalid

valid_mask = df[result_col].isin([0, 1])

valid_df = df[
    valid_mask
].copy()

correct_df = valid_df[
    valid_df[result_col] == 1
].copy()

wrong_df = valid_df[
    valid_df[result_col] == 0
].copy()

invalid_df = df[
    ~valid_mask
].copy()

correct = len(correct_df)
wrong = len(wrong_df)
valid = len(valid_df)
invalid = len(invalid_df)

if valid > 0:
    accuracy_percent = (
        100.0 * correct / valid
    )
else:
    accuracy_percent = np.nan

print("\nBehavioral summary:")
print("Correct:", correct)
print("Wrong:", wrong)
print("Valid:", valid)
print("Invalid/NaN:", invalid)

if np.isnan(accuracy_percent):
    print("Accuracy: NaN")
else:
    print(
        "Accuracy:",
        round(accuracy_percent, 2),
        "%"
    )

# ============================================================
# SAVE BEHAVIORAL REPORTS
# ============================================================

correct_path = os.path.join(
    OUTPUT_DIR,
    "sub024_correct_trials.csv"
)

wrong_path = os.path.join(
    OUTPUT_DIR,
    "sub024_wrong_trials.csv"
)

invalid_path = os.path.join(
    OUTPUT_DIR,
    "sub024_invalid_behavior_trials.csv"
)

all_copy_path = os.path.join(
    OUTPUT_DIR,
    "sub024_all_75_trials_QC_copy.csv"
)

correct_df.to_csv(
    correct_path,
    index=False
)

wrong_df.to_csv(
    wrong_path,
    index=False
)

invalid_df.to_csv(
    invalid_path,
    index=False
)

df.to_csv(
    all_copy_path,
    index=False
)

print("\nSaved behavioral reports:")
print(correct_path)
print(wrong_path)
print(invalid_path)
print(all_copy_path)

# ============================================================
# RUN-LEVEL ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("RUN-LEVEL ANALYSIS")
print("=" * 80)

run_rows = []

for run_value, group in valid_df.groupby(
    "run",
    dropna=False
):

    trials = len(group)

    correct_n = int(
        (group[result_col] == 1).sum()
    )

    wrong_n = int(
        (group[result_col] == 0).sum()
    )

    if trials > 0:
        acc = (
            100.0 * correct_n / trials
        )
    else:
        acc = np.nan

    run_rows.append({
        "run": run_value,
        "trials": trials,
        "correct": correct_n,
        "wrong": wrong_n,
        "accuracy_percent": round(acc, 2)
    })

run_accuracy_df = pd.DataFrame(
    run_rows
)

if len(run_accuracy_df) > 0:

    run_accuracy_df = (
        run_accuracy_df
        .sort_values("run")
        .reset_index(drop=True)
    )

    print(
        run_accuracy_df.to_string(
            index=False
        )
    )

else:

    print(
        "No valid run-level behavioral data found."
    )

run_path = os.path.join(
    OUTPUT_DIR,
    "sub024_run_accuracy.csv"
)

run_accuracy_df.to_csv(
    run_path,
    index=False
)

print("\nSaved:")
print(run_path)

# ============================================================
# MEMORY CONDITION × BEHAVIOR
# ============================================================

print("\n" + "=" * 80)
print("MEMORY CONDITION × BEHAVIOR")
print("=" * 80)

memory_rows = []

for memory_value, group in valid_df.groupby(
    "memory_cond",
    dropna=False
):

    trials = len(group)

    correct_n = int(
        (group[result_col] == 1).sum()
    )

    wrong_n = int(
        (group[result_col] == 0).sum()
    )

    if trials > 0:

        acc = (
            100.0 * correct_n / trials
        )

    else:

        acc = np.nan

    memory_rows.append({
        "memory_cond": memory_value,
        "trials": trials,
        "correct": correct_n,
        "wrong": wrong_n,
        "accuracy_percent": round(acc, 2)
    })

memory_accuracy_df = pd.DataFrame(
    memory_rows
)

if len(memory_accuracy_df) > 0:

    memory_accuracy_df = (
        memory_accuracy_df
        .sort_values("memory_cond")
        .reset_index(drop=True)
    )

    print(
        memory_accuracy_df.to_string(
            index=False
        )
    )

else:

    print(
        "No valid memory-condition behavioral data found."
    )

memory_path = os.path.join(
    OUTPUT_DIR,
    "sub024_memory_accuracy.csv"
)

memory_accuracy_df.to_csv(
    memory_path,
    index=False
)

print("\nSaved:")
print(memory_path)

# ============================================================
# EVENT COUNT QC
# ============================================================
#
# IMPORTANT:
#
# expected_14_events is a BOOLEAN FLAG.
#
# True means:
# "This trial is expected to contain 14 events."
#
# Therefore we DO NOT compare:
#
# event_count == expected_14_events
#
# because that would compare:
#
# 14 == True
#
# Instead we check:
#
# event_count == 14
# AND
# expected_14_events == True
#
# ============================================================

print("\n" + "=" * 80)
print("EVENT COUNT QC")
print("=" * 80)

print("\nEvent count distribution:")

print(
    df["event_count"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nExpected-14-events flag distribution:")

print(
    df["expected_14_events"]
    .value_counts(dropna=False)
)

event_qc = df.copy()

event_qc["event_count_ok"] = (
    (event_qc["event_count"] == 14)
    &
    (event_qc["expected_14_events"] == True)
)

print("\nEvent-count QC:")

print(
    event_qc["event_count_ok"]
    .value_counts(dropna=False)
)

event_pass = int(
    event_qc["event_count_ok"].sum()
)

event_fail = (
    len(event_qc) - event_pass
)

print("\nEvent QC summary:")
print("Total trials:", len(event_qc))
print("Passed:", event_pass)
print("Failed:", event_fail)

event_path = os.path.join(
    OUTPUT_DIR,
    "sub024_event_count_qc.csv"
)

event_qc.to_csv(
    event_path,
    index=False
)

print("\nSaved:")
print(event_path)

# ============================================================
# BAD TRIAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("BAD TRIAL SUMMARY")
print("=" * 80)

print(
    df["bad_trial"]
    .value_counts(dropna=False)
    .sort_index()
)

bad_trial_summary = (
    df["bad_trial"]
    .value_counts(dropna=False)
    .rename_axis("bad_trial")
    .reset_index(name="count")
)

bad_trial_path = os.path.join(
    OUTPUT_DIR,
    "sub024_bad_trial_summary.csv"
)

bad_trial_summary.to_csv(
    bad_trial_path,
    index=False
)

print("\nSaved:")
print(bad_trial_path)

# ============================================================
# TRIAL-LEVEL QC SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TRIAL-LEVEL QC SUMMARY")
print("=" * 80)

trial_qc = df.copy()

trial_qc["behavior_valid"] = (
    trial_qc[result_col].isin([0, 1])
)

trial_qc["event_count_ok"] = (
    (trial_qc["event_count"] == 14)
    &
    (trial_qc["expected_14_events"] == True)
)

trial_qc["bad_trial_flag"] = (
    trial_qc["bad_trial"].astype(bool)
)

trial_qc_path = os.path.join(
    OUTPUT_DIR,
    "sub024_trial_level_qc_summary.csv"
)

trial_qc.to_csv(
    trial_qc_path,
    index=False
)

print("Saved:")
print(trial_qc_path)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUB-024 REVIEW SUMMARY")
print("=" * 80)

print("\nTrials:")
print("Total:", len(df))

print("\nBehavior:")
print("Correct:", correct)
print("Wrong:", wrong)
print("Valid:", valid)
print("Invalid:", invalid)

if np.isnan(accuracy_percent):
    print("Accuracy: NaN")
else:
    print(
        "Accuracy:",
        round(accuracy_percent, 2),
        "%"
    )

print("\nMemory conditions:")

print(
    df["memory_cond"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nEvent count:")

print(
    df["event_count"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nEvent QC:")
print("Passed:", event_pass)
print("Failed:", event_fail)

print("\nBad trial:")

print(
    df["bad_trial"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nOutput directory:")
print(OUTPUT_DIR)

# ============================================================
# SAFETY CONFIRMATION
# ============================================================

print("\n" + "=" * 80)
print("SAFETY CHECK")
print("=" * 80)

print("NO EEG FILES WERE MODIFIED.")
print("NO EEG FILES WERE DELETED.")
print("NO EPOCHS WERE DELETED.")
print("NO EPOCHS WERE MODIFIED.")
print("NO SUBJECTS WERE DELETED.")
print("ORIGINAL INPUT CSV WAS NOT MODIFIED.")
print("ALL PROCESSING WAS READ-ONLY.")

print("=" * 80)
print("QC COMPLETE")
print("=" * 80)