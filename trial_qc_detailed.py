import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

INPUT_FILE = (
    BASE_DIR
    / "qc"
    / "trial_table"
    / "TRIAL_LEVEL_TABLE.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "qc"
    / "trial_table"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("DETAILED TRIAL QC")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

print("\nLoaded trials:", len(df))

# ============================================================
# NON-14 EVENT TRIALS
# ============================================================

non14 = df[df["event_count"] != 14].copy()

print("\n" + "=" * 80)
print("NON-14 EVENT TRIALS")
print("=" * 80)

print("Count:", len(non14))

print("\nDistribution:")

print(
    non14["event_count"]
    .value_counts()
    .sort_index()
)

# ============================================================
# BAD TRIALS
# ============================================================

print("\n" + "=" * 80)
print("BAD TRIALS")
print("=" * 80)

bad = df[df["bad_trial"] == True].copy()

print("Bad trials:", len(bad))

if len(bad) > 0:

    print("\nBad trial event counts:")

    print(
        bad["event_count"]
        .value_counts()
        .sort_index()
    )

    print("\nBad trial memory conditions:")

    print(
        bad["memory_cond"]
        .value_counts(dropna=False)
        .sort_index()
    )

# ============================================================
# NON-14 BY MEMORY CONDITION
# ============================================================

print("\n" + "=" * 80)
print("NON-14 × MEMORY CONDITION")
print("=" * 80)

print(
    pd.crosstab(
        non14["event_count"],
        non14["memory_cond"],
        dropna=False
    )
)

# ============================================================
# NON-14 BY SUBJECT
# ============================================================

print("\n" + "=" * 80)
print("NON-14 × SUBJECT")
print("=" * 80)

subject_non14 = (
    non14
    .groupby("subject")
    .agg(
        non14_trials=("trial", "count"),
        total_trials=("trial", "size")
    )
    .reset_index()
)

subject_total = (
    df
    .groupby("subject")
    .size()
    .reset_index(name="all_trials")
)

subject_non14 = subject_non14.merge(
    subject_total,
    on="subject",
    how="right"
)

subject_non14["non14_trials"] = (
    subject_non14["non14_trials"]
    .fillna(0)
)

subject_non14["non14_percent"] = (
    subject_non14["non14_trials"]
    / subject_non14["all_trials"]
    * 100
)

subject_non14 = subject_non14.sort_values(
    "non14_trials",
    ascending=False
)

print(subject_non14.to_string(index=False))

# ============================================================
# NON-14 BY FILE
# ============================================================

print("\n" + "=" * 80)
print("NON-14 × FILE")
print("=" * 80)

file_non14 = (
    non14
    .groupby(
        ["subject", "run", "source_file"]
    )
    .agg(
        non14_trials=("trial", "count"),
        event_counts=(
            "event_count",
            lambda x: ",".join(
                sorted(
                    x.astype(str).unique()
                )
            )
        )
    )
    .reset_index()
)

file_total = (
    df
    .groupby(
        ["subject", "run", "source_file"]
    )
    .size()
    .reset_index(
        name="total_trials"
    )
)

file_non14 = file_non14.merge(
    file_total,
    on=[
        "subject",
        "run",
        "source_file"
    ],
    how="right"
)

file_non14["non14_trials"] = (
    file_non14["non14_trials"]
    .fillna(0)
)

file_non14["non14_percent"] = (
    file_non14["non14_trials"]
    / file_non14["total_trials"]
    * 100
)

file_non14 = file_non14.sort_values(
    "non14_trials",
    ascending=False
)

print(
    file_non14[
        [
            "subject",
            "run",
            "non14_trials",
            "total_trials",
            "non14_percent",
            "event_counts"
        ]
    ]
    .to_string(index=False)
)

# ============================================================
# ACCURACY BY MEMORY CONDITION
# ============================================================

print("\n" + "=" * 80)
print("ACCURACY × MEMORY CONDITION")
print("=" * 80)

accuracy_memory = (
    df
    .groupby("memory_cond")
    .agg(
        trials=("trial", "count"),
        correct=("accuracy", lambda x: (x == 1).sum()),
        incorrect=("accuracy", lambda x: (x == 0).sum()),
        missing=("accuracy", lambda x: x.isna().sum())
    )
    .reset_index()
)

accuracy_memory["accuracy_percent"] = (
    accuracy_memory["correct"]
    /
    (
        accuracy_memory["correct"]
        + accuracy_memory["incorrect"]
    )
    * 100
)

print(
    accuracy_memory.to_string(
        index=False
    )
)

# ============================================================
# BAD TRIALS BY SUBJECT
# ============================================================

bad_subject = (
    bad
    .groupby("subject")
    .size()
    .reset_index(
        name="bad_trials"
    )
    .sort_values(
        "bad_trials",
        ascending=False
    )
)

print("\n" + "=" * 80)
print("BAD TRIALS × SUBJECT")
print("=" * 80)

print(
    bad_subject.to_string(
        index=False
    )
)

# ============================================================
# SAVE OUTPUTS
# ============================================================

non14.to_csv(
    OUTPUT_DIR / "NON14_TRIALS.csv",
    index=False
)

subject_non14.to_csv(
    OUTPUT_DIR / "NON14_BY_SUBJECT.csv",
    index=False
)

file_non14.to_csv(
    OUTPUT_DIR / "NON14_BY_FILE.csv",
    index=False
)

accuracy_memory.to_csv(
    OUTPUT_DIR / "ACCURACY_BY_MEMORY.csv",
    index=False
)

bad.to_csv(
    OUTPUT_DIR / "BAD_TRIALS.csv",
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("TRIAL QC COMPLETE")
print("=" * 80)

print("\nCreated:")

print("  NON14_TRIALS.csv")
print("  NON14_BY_SUBJECT.csv")
print("  NON14_BY_FILE.csv")
print("  ACCURACY_BY_MEMORY.csv")
print("  BAD_TRIALS.csv")

print("\nIMPORTANT:")
print("No trials were deleted.")
print("No subjects were deleted.")
print("No EEG data were modified.")

print("\nNEXT STEP:")
print("Inspect trial anomalies before EEG preprocessing.")

print("=" * 80)