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
print("SUB-024 BEHAVIORAL ACCURACY - CORRECTED ANALYSIS")
print("=" * 80)

df = pd.read_csv(INPUT)

sub = df[
    df["subject"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("sub-024")
].copy()

print("\nSUB-024 trials:", len(sub))

# ================================================================
# IDENTIFY COLUMNS
# ================================================================

print("\nCOLUMNS:")
print(sub.columns.tolist())

print("\n" + "=" * 80)
print("RAW ACCURACY VALUES")
print("=" * 80)

print(sub["accuracy"].value_counts(dropna=False).sort_index())

# ================================================================
# BASIC ACCURACY
# ================================================================

sub["accuracy_numeric"] = pd.to_numeric(
    sub["accuracy"],
    errors="coerce"
)

valid = sub["accuracy_numeric"].dropna()

correct = (valid == 1).sum()
wrong = (valid == 0).sum()
other = len(valid) - correct - wrong

print("\n" + "=" * 80)
print("GLOBAL BEHAVIOR")
print("=" * 80)

print("Total trials:", len(sub))
print("Valid accuracy:", len(valid))
print("Correct:", correct)
print("Wrong:", wrong)
print("Other:", other)

if len(valid) > 0:
    accuracy_percent = valid.mean() * 100
else:
    accuracy_percent = float("nan")

print(f"Accuracy: {accuracy_percent:.2f}%")

# ================================================================
# CHECK TEXT RESULT COLUMN
# ================================================================

text_result_cols = []

for c in sub.columns:
    vals = (
        sub[c]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    if "correct" in vals or "wrong" in vals:
        text_result_cols.append(c)

print("\n" + "=" * 80)
print("TEXT RESULT COLUMNS")
print("=" * 80)

print(text_result_cols)

for c in text_result_cols:
    print(f"\n--- {c} ---")
    print(
        sub[c]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts(dropna=False)
        .to_string()
    )

# ================================================================
# RUN ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("RUN-LEVEL ACCURACY")
print("=" * 80)

if "run" in sub.columns:

    run_summary = (
        sub.groupby("run")
        .agg(
            trials=("accuracy_numeric", "size"),
            correct=("accuracy_numeric", lambda x: (x == 1).sum()),
            wrong=("accuracy_numeric", lambda x: (x == 0).sum())
        )
        .reset_index()
    )

    run_summary["accuracy_percent"] = (
        run_summary["correct"]
        / run_summary["trials"]
        * 100
    )

    print(run_summary.to_string(index=False))

    run_summary.to_csv(
        OUTPUT / "sub024_run_accuracy_corrected.csv",
        index=False
    )

else:
    print("NO RUN COLUMN FOUND")

# ================================================================
# MEMORY CONDITION ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("MEMORY CONDITION × ACCURACY")
print("=" * 80)

if "memory_cond" in sub.columns:

    memory_summary = (
        sub.groupby("memory_cond")
        .agg(
            trials=("accuracy_numeric", "size"),
            correct=("accuracy_numeric", lambda x: (x == 1).sum()),
            wrong=("accuracy_numeric", lambda x: (x == 0).sum())
        )
        .reset_index()
    )

    memory_summary["accuracy_percent"] = (
        memory_summary["correct"]
        / memory_summary["trials"]
        * 100
    )

    print(memory_summary.to_string(index=False))

    memory_summary.to_csv(
        OUTPUT / "sub024_memory_accuracy_corrected.csv",
        index=False
    )

else:
    print("NO memory_cond COLUMN FOUND")

# ================================================================
# MEMORY × RUN
# ================================================================

if "memory_cond" in sub.columns and "run" in sub.columns:

    print("\n" + "=" * 80)
    print("MEMORY × RUN ACCURACY")
    print("=" * 80)

    memory_run = (
        sub.groupby(["run", "memory_cond"])
        .agg(
            trials=("accuracy_numeric", "size"),
            correct=("accuracy_numeric", lambda x: (x == 1).sum()),
            wrong=("accuracy_numeric", lambda x: (x == 0).sum())
        )
        .reset_index()
    )

    memory_run["accuracy_percent"] = (
        memory_run["correct"]
        / memory_run["trials"]
        * 100
    )

    print(memory_run.to_string(index=False))

    memory_run.to_csv(
        OUTPUT / "sub024_memory_by_run_accuracy.csv",
        index=False
    )

# ================================================================
# FULL TRIAL TABLE WITH NORMALIZED RESULT
# ================================================================

sub["behavior_result"] = sub["accuracy_numeric"].map({
    1: "correct",
    0: "wrong"
})

trial_output = OUTPUT / "sub024_all_trials_corrected.csv"

sub.to_csv(
    trial_output,
    index=False
)

print("\nSaved:")
print(trial_output)

# ================================================================
# FINAL
# ================================================================

print("\n" + "=" * 80)
print("FINAL")
print("=" * 80)

print("SUB-024 trials:", len(sub))
print("Correct:", correct)
print("Wrong:", wrong)
print(f"Accuracy: {accuracy_percent:.2f}%")

print("\nOutput directory:")
print(OUTPUT)

print("\nNO EEG FILES MODIFIED.")
print("NO EPOCHS DELETED.")
print("NO SUBJECTS DELETED.")
print("NO DATA EXCLUDED.")
