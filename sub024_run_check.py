import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = BASE / "qc" / "trial_table" / "TRIAL_LEVEL_TABLE.csv"

OUTPUT = (
    BASE
    / "qc"
    / "trial_anomaly_inspection"
    / "sub024_final_behavior_review"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SUB-024 RUN IDENTIFICATION FROM SOURCE FILE")
print("=" * 80)

df = pd.read_csv(INPUT)

# ---------------------------------------------------------------
# SUB-024
# ---------------------------------------------------------------

sub = df[
    df["subject"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("sub-024")
].copy()

print("\nSUB-024 rows:", len(sub))

# ---------------------------------------------------------------
# SHOW SOURCE FILE DISTRIBUTION
# ---------------------------------------------------------------

print("\n" + "=" * 80)
print("SOURCE FILE DISTRIBUTION")
print("=" * 80)

print(
    sub["source_file"]
    .value_counts(dropna=False)
    .to_string()
)

# ---------------------------------------------------------------
# EXTRACT RUN NUMBER FROM SOURCE FILE
# ---------------------------------------------------------------

sub["run_from_file"] = (
    sub["source_file"]
    .astype(str)
    .str.extract(r"_run-(\d+)_", expand=False)
)

print("\n" + "=" * 80)
print("RUN IDENTIFICATION")
print("=" * 80)

print(
    sub["run_from_file"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

# ---------------------------------------------------------------
# CHECK UNIDENTIFIED RUNS
# ---------------------------------------------------------------

missing_run = sub["run_from_file"].isna().sum()

print("\nUnidentified runs:", missing_run)

if missing_run > 0:
    print("\nSOURCE FILES WITH UNIDENTIFIED RUN:")
    print(
        sub.loc[
            sub["run_from_file"].isna(),
            "source_file"
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

# ---------------------------------------------------------------
# RUN-LEVEL ACCURACY
# ---------------------------------------------------------------

print("\n" + "=" * 80)
print("RUN-LEVEL ACCURACY")
print("=" * 80)

sub["accuracy_numeric"] = pd.to_numeric(
    sub["accuracy"],
    errors="coerce"
)

run_summary = (
    sub.groupby("run_from_file")
    .agg(
        trials=("accuracy_numeric", "size"),
        correct=("accuracy_numeric", lambda x: (x == 1).sum()),
        wrong=("accuracy_numeric", lambda x: (x == 0).sum())
    )
    .reset_index()
)

run_summary["accuracy_percent"] = (
    run_summary["correct"]
    /
    run_summary["trials"]
    * 100
)

print(
    run_summary
    .sort_values("run_from_file")
    .to_string(index=False)
)

run_summary.to_csv(
    OUTPUT / "sub024_run_accuracy_from_source.csv",
    index=False
)

# ---------------------------------------------------------------
# RUN × MEMORY CONDITION
# ---------------------------------------------------------------

print("\n" + "=" * 80)
print("RUN × MEMORY CONDITION")
print("=" * 80)

run_memory = (
    sub.groupby(
        ["run_from_file", "memory_cond"]
    )
    .agg(
        trials=("accuracy_numeric", "size"),
        correct=("accuracy_numeric", lambda x: (x == 1).sum()),
        wrong=("accuracy_numeric", lambda x: (x == 0).sum())
    )
    .reset_index()
)

run_memory["accuracy_percent"] = (
    run_memory["correct"]
    /
    run_memory["trials"]
    * 100
)

print(
    run_memory
    .sort_values(["run_from_file", "memory_cond"])
    .to_string(index=False)
)

run_memory.to_csv(
    OUTPUT / "sub024_run_memory_accuracy_from_source.csv",
    index=False
)

# ---------------------------------------------------------------
# RUN × FEEDBACK
# ---------------------------------------------------------------

print("\n" + "=" * 80)
print("RUN × FEEDBACK")
print("=" * 80)

feedback = (
    sub.groupby(
        ["run_from_file", "feedback"]
    )
    .size()
    .reset_index(name="count")
)

print(
    feedback
    .sort_values(["run_from_file", "feedback"])
    .to_string(index=False)
)

feedback.to_csv(
    OUTPUT / "sub024_run_feedback_from_source.csv",
    index=False
)

# ---------------------------------------------------------------
# TRIAL ORDER
# ---------------------------------------------------------------

print("\n" + "=" * 80)
print("TRIAL DISTRIBUTION BY RUN")
print("=" * 80)

if "trial" in sub.columns:

    trial_summary = (
        sub.groupby("run_from_file")["trial"]
        .agg(
            trial_count="count",
            min_trial="min",
            max_trial="max"
        )
        .reset_index()
    )

    print(
        trial_summary
        .sort_values("run_from_file")
        .to_string(index=False)
    )

# ---------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------

sub.to_csv(
    OUTPUT / "sub024_trials_with_run_from_source.csv",
    index=False
)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print("\nFiles saved in:")
print(OUTPUT)

print("\nNO EEG FILES MODIFIED.")
print("NO EPOCHS DELETED.")
print("NO SUBJECTS DELETED.")
print("NO DATA EXCLUDED.")
