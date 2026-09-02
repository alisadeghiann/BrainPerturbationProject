import pandas as pd
import numpy as np
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
    / "events"
    / "ALL_EVENTS_83_RUNS.csv"
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
print("BUILDING TRIAL-LEVEL TABLE")
print("=" * 80)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("\nRows:", len(df))
print("\nColumns:")
for c in df.columns:
    print(" ", c)

# ============================================================
# NORMALIZE COLUMNS
# ============================================================

df.columns = [c.strip().lower() for c in df.columns]

# Normalize subject
df["subject"] = df["subject"].astype(str).str.strip()

# Normalize run
df["run"] = pd.to_numeric(df["run"], errors="coerce")

# Normalize trial
df["trial"] = pd.to_numeric(df["trial"], errors="coerce")

# Normalize memory condition
df["memory_cond"] = pd.to_numeric(
    df["memory_cond"],
    errors="coerce"
)

# Normalize strings
for col in [
    "type",
    "task_role",
    "value",
    "letter",
    "source_file"
]:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

# ============================================================
# BASIC CHECK
# ============================================================

print("\n" + "=" * 80)
print("BASIC CHECK")
print("=" * 80)

print("Unique EEG files:", df["source_file"].nunique())
print("Unique subjects:", df["subject"].nunique())

print(
    "Unique subject × run:",
    df[["subject", "run"]].drop_duplicates().shape[0]
)

print(
    "Unique subject × run × trial:",
    df[["subject", "run", "trial"]]
    .drop_duplicates()
    .shape[0]
)

# ============================================================
# REMOVE ONLY EXACT DUPLICATE EVENT ROWS
# ============================================================

print("\n" + "=" * 80)
print("DUPLICATE CHECK")
print("=" * 80)

duplicate_mask = df.duplicated(keep=False)

duplicates = df[duplicate_mask].copy()

print("Duplicate rows:", len(duplicates))

if len(duplicates) > 0:
    duplicates.to_csv(
        OUTPUT_DIR / "duplicate_events.csv",
        index=False
    )

    print(
        "Saved:",
        OUTPUT_DIR / "duplicate_events.csv"
    )

# Keep first exact duplicate
df = df.drop_duplicates().copy()

print("Rows after exact duplicate removal:", len(df))

# ============================================================
# BUILD TRIAL TABLE
# ============================================================

print("\n" + "=" * 80)
print("BUILDING TRIALS")
print("=" * 80)

group_cols = [
    "source_file",
    "subject",
    "run",
    "trial"
]

trial_rows = []

for keys, g in df.groupby(group_cols, dropna=False):

    source_file, subject, run, trial = keys

    g = g.sort_values("sample")

    row = {
        "source_file": source_file,
        "subject": subject,
        "run": run,
        "trial": trial,
    }

    # --------------------------------------------------------
    # MEMORY CONDITION
    # --------------------------------------------------------

    mem_values = (
        g["memory_cond"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(mem_values) > 0:
        row["memory_cond"] = mem_values[0]
    else:
        row["memory_cond"] = np.nan

    # --------------------------------------------------------
    # EVENT COUNT
    # --------------------------------------------------------

    row["event_count"] = len(g)

    # --------------------------------------------------------
    # EVENT TYPES
    # --------------------------------------------------------

    types = g["type"].dropna().tolist()

    row["has_fixation"] = (
        "show_cross" in types
    )

    row["has_work_memory"] = (
        "show_dash" in types
    )

    row["has_probe"] = (
        "right_click" in types
        or "left_click" in types
    )

    row["has_feedback"] = (
        "sound_beep" in types
        or "sound_buzz" in types
    )

    # --------------------------------------------------------
    # TASK ROLES
    # --------------------------------------------------------

    roles = set(
        g["task_role"]
        .dropna()
        .tolist()
    )

    row["has_to_remember"] = (
        "to_remember" in roles
    )

    row["has_to_ignore"] = (
        "to_ignore" in roles
    )

    row["has_probe_target"] = (
        "probe_target" in roles
    )

    row["has_probe_not_shown"] = (
        "probe_not_shown" in roles
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    if "feedback_correct" in roles:
        row["accuracy"] = 1

    elif "feedback_incorrect" in roles:
        row["accuracy"] = 0

    elif "remembered_correct" in roles:
        row["accuracy"] = 1

    elif "remembered_incorrect" in roles:
        row["accuracy"] = 0

    elif "ignored_correct" in roles:
        row["accuracy"] = 1

    elif "ignored_incorrect" in roles:
        row["accuracy"] = 0

    else:
        row["accuracy"] = np.nan

    # --------------------------------------------------------
    # LETTERS
    # --------------------------------------------------------

    letters = (
        g.loc[
            g["letter"].notna(),
            "letter"
        ]
        .astype(str)
        .tolist()
    )

    letters = [
        x for x in letters
        if x not in ["nan", "n/a", "+", "-"]
    ]

    row["letters_presented"] = " ".join(letters)

    # --------------------------------------------------------
    # STIMULUS COUNT
    # --------------------------------------------------------

    row["n_letters"] = len(letters)

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    response_values = (
        g.loc[
            g["type"].isin(
                ["left_click", "right_click"]
            ),
            "value"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    if response_values:
        row["response"] = response_values[-1]
    else:
        row["response"] = np.nan

    # --------------------------------------------------------
    # FEEDBACK
    # --------------------------------------------------------

    feedback_values = (
        g.loc[
            g["type"].isin(
                ["sound_beep", "sound_buzz"]
            ),
            "value"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    if feedback_values:
        row["feedback"] = feedback_values[-1]
    else:
        row["feedback"] = np.nan

    # --------------------------------------------------------
    # START / END SAMPLE
    # --------------------------------------------------------

    samples = pd.to_numeric(
        g["sample"],
        errors="coerce"
    ).dropna()

    if len(samples) > 0:

        row["start_sample"] = int(samples.min())
        row["end_sample"] = int(samples.max())

    else:

        row["start_sample"] = np.nan
        row["end_sample"] = np.nan

    # --------------------------------------------------------
    # START / END LATENCY
    # --------------------------------------------------------

    latency = pd.to_numeric(
        g["latency"],
        errors="coerce"
    ).dropna()

    if len(latency) > 0:

        row["start_latency"] = latency.min()
        row["end_latency"] = latency.max()

    else:

        row["start_latency"] = np.nan
        row["end_latency"] = np.nan

    # --------------------------------------------------------
    # TRIAL QUALITY
    # --------------------------------------------------------

    row["expected_14_events"] = (
        row["event_count"] == 14
    )

    row["bad_trial"] = (
        "bad_trial" in roles
    )

    trial_rows.append(row)

# ============================================================
# DATAFRAME
# ============================================================

trials = pd.DataFrame(trial_rows)

# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR
    / "TRIAL_LEVEL_TABLE.csv"
)

trials.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TRIAL SUMMARY")
print("=" * 80)

print(
    "\nTotal trials:",
    len(trials)
)

print(
    "Subjects:",
    trials["subject"].nunique()
)

print(
    "Runs:",
    trials[["subject", "run"]]
    .drop_duplicates()
    .shape[0]
)

print(
    "Files:",
    trials["source_file"].nunique()
)

print("\nMemory conditions:")

print(
    trials["memory_cond"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nAccuracy:")

print(
    trials["accuracy"]
    .value_counts(dropna=False)
)

print("\nEvent count per trial:")

print(
    trials["event_count"]
    .value_counts()
    .sort_index()
)

print("\nBad trials:")

print(
    trials["bad_trial"].value_counts()
)

print("\nTrials with exactly 14 events:")

print(
    trials["expected_14_events"].value_counts()
)

# ============================================================
# FILE SUMMARY
# ============================================================

file_summary = (
    trials
    .groupby(
        ["subject", "run", "source_file"],
        dropna=False
    )
    .agg(
        trials=("trial", "nunique"),
        mean_events=("event_count", "mean"),
        bad_trials=("bad_trial", "sum"),
        valid_14_event_trials=(
            "expected_14_events",
            "sum"
        ),
        accuracy=("accuracy", "mean")
    )
    .reset_index()
)

file_summary.to_csv(
    OUTPUT_DIR / "TRIAL_FILE_SUMMARY.csv",
    index=False
)

# ============================================================
# SUBJECT SUMMARY
# ============================================================

subject_summary = (
    trials
    .groupby("subject", dropna=False)
    .agg(
        files=("source_file", "nunique"),
        runs=("run", "nunique"),
        trials=("trial", "nunique"),
        bad_trials=("bad_trial", "sum"),
        accuracy=("accuracy", "mean")
    )
    .reset_index()
)

subject_summary.to_csv(
    OUTPUT_DIR / "TRIAL_SUBJECT_SUMMARY.csv",
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("TRIAL TABLE COMPLETE")
print("=" * 80)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nFILES CREATED:")
print("  TRIAL_LEVEL_TABLE.csv")
print("  TRIAL_FILE_SUMMARY.csv")
print("  TRIAL_SUBJECT_SUMMARY.csv")
print("  duplicate_events.csv")

print("\nNEXT STEP:")
print("EEG PREPROCESSING + ARTIFACT QC")

print("=" * 80)