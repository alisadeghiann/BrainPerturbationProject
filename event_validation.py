import pandas as pd
import numpy as np
import re
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

EVENT_FILE = (
    BASE_DIR
    / "qc"
    / "events"
    / "ALL_EVENTS_83_RUNS.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "qc"
    / "event_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_subject(value):
    """
    Extract subject ID such as sub-001 from filename/text.
    """
    match = re.search(
        r"(sub-\d+)",
        str(value)
    )

    if match:
        return match.group(1)

    return "UNKNOWN"


def extract_run(value):
    """
    Extract run number from filename/text.
    """
    match = re.search(
        r"run-(\d+)",
        str(value)
    )

    if match:
        return int(match.group(1))

    return np.nan


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("EVENT VALIDATION - 83 EEG RUNS")
print("=" * 80)

print()
print("Event file:")
print(EVENT_FILE)


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not EVENT_FILE.exists():

    raise FileNotFoundError(
        f"\nEVENT FILE NOT FOUND:\n{EVENT_FILE}"
    )


# ============================================================
# LOAD CSV
# ============================================================

print_section("LOADING EVENTS")

df = pd.read_csv(
    EVENT_FILE
)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print()
print("Columns found:")

for column in df.columns:
    print("  ", repr(column))


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

df.columns = [
    str(column)
    .strip()
    .lower()
    .replace(" ", "_")
    for column in df.columns
]

print()
print("Normalized columns:")
print(df.columns.tolist())


# ============================================================
# IDENTIFY FILE COLUMN
# ============================================================

file_candidates = [
    "file",
    "filename",
    "file_name",
    "eeg_file",
    "set_file",
    "source_file"
]

file_column = None

for candidate in file_candidates:

    if candidate in df.columns:

        file_column = candidate
        break


if file_column is None:

    raise ValueError(
        "\nCould not identify the EEG filename column.\n\n"
        f"Available columns:\n{df.columns.tolist()}\n\n"
        "Expected one of:\n"
        f"{file_candidates}"
    )


print()
print("EEG file column detected:")
print(" ", file_column)


# ============================================================
# STANDARDIZED FILE COLUMN
# ============================================================

df["file"] = (
    df[file_column]
    .astype(str)
    .str.strip()
)


# ============================================================
# REQUIRED EVENT COLUMNS
# ============================================================

required_columns = [
    "subject",
    "session",
    "run",
    "file",
    "latency",
    "sample",
    "trial",
    "type",
    "task_role",
    "memory_cond",
    "value"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print_section("MISSING COLUMNS")

    print(
        "WARNING: The following columns are missing:"
    )

    for column in missing_columns:
        print("  ", column)

else:

    print_section("REQUIRED COLUMNS")

    print("All required columns are present.")


# ============================================================
# SUBJECT / RUN NORMALIZATION
# ============================================================

print_section("SUBJECT / RUN NORMALIZATION")


# Subject
if "subject" in df.columns:

    df["subject"] = (
        df["subject"]
        .astype(str)
        .str.strip()
    )

else:

    df["subject"] = df["file"].apply(
        extract_subject
    )


# Fix possible subject formats
df["subject_from_file"] = df["file"].apply(
    extract_subject
)


# Run
if "run" in df.columns:

    df["run"] = pd.to_numeric(
        df["run"],
        errors="coerce"
    )

else:

    df["run"] = df["file"].apply(
        extract_run
    )


print(
    "Unique subjects:",
    df["subject"].nunique()
)

print(
    "Unique files:",
    df["file"].nunique()
)

print(
    "Unique runs:",
    df["run"].nunique()
)


# ============================================================
# FILE COVERAGE
# ============================================================

print_section("FILE COVERAGE")

files_per_subject = (
    df.groupby("subject")["file"]
    .nunique()
    .sort_index()
)

print("Files per subject:")
print(
    files_per_subject.to_string()
)

print()
print(
    "Total unique EEG runs represented:",
    df["file"].nunique()
)

print(
    "Expected EEG runs:",
    83
)

if df["file"].nunique() == 83:

    print(
        "STATUS: ALL 83 EEG RUNS REPRESENTED"
    )

else:

    print(
        "STATUS: CHECK RUN COVERAGE"
    )


# ============================================================
# EVENTS PER FILE
# ============================================================

print_section("EVENTS PER EEG RUN")

events_per_file = (
    df.groupby(
        [
            "subject",
            "run",
            "file"
        ]
    )
    .size()
    .reset_index(
        name="event_count"
    )
    .sort_values(
        [
            "subject",
            "run"
        ]
    )
)

print(
    events_per_file.to_string(
        index=False
    )
)

events_per_file.to_csv(
    OUTPUT_DIR / "events_per_file.csv",
    index=False
)


# ============================================================
# EVENT COUNT STATISTICS
# ============================================================

print_section("EVENT COUNT STATISTICS")

print(
    events_per_file["event_count"]
    .describe()
    .to_string()
)


# ============================================================
# LATENCY VALIDATION
# ============================================================

print_section("LATENCY VALIDATION")

if "latency" in df.columns:

    df["latency_num"] = pd.to_numeric(
        df["latency"],
        errors="coerce"
    )

    invalid_latency = df[
        df["latency_num"].isna()
        |
        (df["latency_num"] < 0)
    ].copy()

    print(
        "Invalid latency rows:",
        len(invalid_latency)
    )

else:

    invalid_latency = pd.DataFrame()

    print(
        "Latency column not available."
    )


# ============================================================
# SAMPLE VALIDATION
# ============================================================

print_section("SAMPLE VALIDATION")

if "sample" in df.columns:

    df["sample_num"] = pd.to_numeric(
        df["sample"],
        errors="coerce"
    )

    invalid_sample = df[
        df["sample_num"].isna()
        |
        (df["sample_num"] < 0)
    ].copy()

    print(
        "Invalid sample rows:",
        len(invalid_sample)
    )

else:

    invalid_sample = pd.DataFrame()

    print(
        "Sample column not available."
    )


# ============================================================
# TRIAL VALIDATION
# ============================================================

print_section("TRIAL VALIDATION")

if "trial" in df.columns:

    df["trial_num"] = pd.to_numeric(
        df["trial"],
        errors="coerce"
    )

    invalid_trial = df[
        df["trial_num"].isna()
        |
        (df["trial_num"] < 0)
    ].copy()

    print(
        "Invalid trial rows:",
        len(invalid_trial)
    )

    print()
    print(
        "Unique trials:",
        df["trial_num"].nunique()
    )

    print(
        "Minimum trial:",
        df["trial_num"].min()
    )

    print(
        "Maximum trial:",
        df["trial_num"].max()
    )

else:

    invalid_trial = pd.DataFrame()

    print(
        "Trial column not available."
    )


# ============================================================
# DUPLICATE EVENT CHECK
# ============================================================

print_section("DUPLICATE EVENT CHECK")

duplicate_columns = [
    column
    for column in [
        "file",
        "latency",
        "sample",
        "trial",
        "type",
        "task_role",
        "value"
    ]
    if column in df.columns
]

duplicates = df[
    df.duplicated(
        subset=duplicate_columns,
        keep=False
    )
].copy()

print(
    "Duplicate event rows:",
    len(duplicates)
)

if len(duplicates) > 0:

    print()
    print(
        "WARNING: Duplicate events detected."
    )

else:

    print(
        "STATUS: No duplicate events detected."
    )


# ============================================================
# EVENT TYPE ANALYSIS
# ============================================================

print_section("EVENT TYPE ANALYSIS")

if "type" in df.columns:

    event_types = (
        df["type"]
        .astype(str)
        .value_counts()
        .rename_axis("event_type")
        .reset_index(
            name="count"
        )
    )

    print(
        event_types.to_string(
            index=False
        )
    )

    event_types.to_csv(
        OUTPUT_DIR / "event_type_summary.csv",
        index=False
    )


# ============================================================
# TASK ROLE ANALYSIS
# ============================================================

print_section("TASK ROLE ANALYSIS")

if "task_role" in df.columns:

    task_roles = (
        df["task_role"]
        .astype(str)
        .value_counts()
        .rename_axis("task_role")
        .reset_index(
            name="count"
        )
    )

    print(
        task_roles.to_string(
            index=False
        )
    )

    task_roles.to_csv(
        OUTPUT_DIR / "task_role_summary.csv",
        index=False
    )


# ============================================================
# MEMORY CONDITION ANALYSIS
# ============================================================

print_section("MEMORY CONDITION ANALYSIS")

if "memory_cond" in df.columns:

    memory_conditions = (
        df["memory_cond"]
        .astype(str)
        .value_counts()
        .rename_axis("memory_cond")
        .reset_index(
            name="count"
        )
    )

    print(
        memory_conditions.to_string(
            index=False
        )
    )

    memory_conditions.to_csv(
        OUTPUT_DIR / "memory_condition_summary.csv",
        index=False
    )


# ============================================================
# EVENT VALUE ANALYSIS
# ============================================================

print_section("EVENT VALUE ANALYSIS")

if "value" in df.columns:

    event_values = (
        df["value"]
        .astype(str)
        .value_counts()
        .rename_axis("value")
        .reset_index(
            name="count"
        )
    )

    print(
        event_values.head(100).to_string(
            index=False
        )
    )

    event_values.to_csv(
        OUTPUT_DIR / "event_value_summary.csv",
        index=False
    )


# ============================================================
# TASK ROLE × MEMORY CONDITION
# ============================================================

print_section(
    "TASK ROLE × MEMORY CONDITION"
)

if (
    "task_role" in df.columns
    and
    "memory_cond" in df.columns
):

    role_memory = pd.crosstab(
        df["task_role"],
        df["memory_cond"]
    )

    print(
        role_memory.to_string()
    )

    role_memory.to_csv(
        OUTPUT_DIR
        / "task_role_by_memory_condition.csv"
    )


# ============================================================
# EVENT TYPE × TASK ROLE
# ============================================================

print_section(
    "EVENT TYPE × TASK ROLE"
)

if (
    "type" in df.columns
    and
    "task_role" in df.columns
):

    type_role = pd.crosstab(
        df["type"],
        df["task_role"]
    )

    print(
        type_role.to_string()
    )

    type_role.to_csv(
        OUTPUT_DIR
        / "event_type_by_task_role.csv"
    )


# ============================================================
# TRIAL COMPLETENESS
# ============================================================

print_section("TRIAL STRUCTURE")

if (
    "trial" in df.columns
    and
    "task_role" in df.columns
):

    trial_structure = (
        df.groupby(
            [
                "file",
                "trial"
            ]
        )
        .agg(
            event_count=(
                "type",
                "count"
            ),
            task_roles=(
                "task_role",
                lambda x: "|".join(
                    sorted(
                        set(
                            x.astype(str)
                        )
                    )
                )
            ),
            event_types=(
                "type",
                lambda x: "|".join(
                    sorted(
                        set(
                            x.astype(str)
                        )
                    )
                )
            )
        )
        .reset_index()
    )

    print(
        "Unique file × trial combinations:",
        len(trial_structure)
    )

    print()
    print(
        "Trial event-count statistics:"
    )

    print(
        trial_structure[
            "event_count"
        ].describe().to_string()
    )

    trial_structure.to_csv(
        OUTPUT_DIR
        / "trial_structure.csv",
        index=False
    )


# ============================================================
# SUBJECT × RUN SUMMARY
# ============================================================

print_section("SUBJECT × RUN SUMMARY")

subject_run_summary = (
    df.groupby(
        [
            "subject",
            "run"
        ]
    )
    .agg(
        files=(
            "file",
            "nunique"
        ),
        events=(
            "file",
            "size"
        )
    )
    .reset_index()
    .sort_values(
        [
            "subject",
            "run"
        ]
    )
)

print(
    subject_run_summary.to_string(
        index=False
    )
)

subject_run_summary.to_csv(
    OUTPUT_DIR
    / "subject_run_summary.csv",
    index=False
)


# ============================================================
# SAVE INVALID RECORDS
# ============================================================

invalid_latency.to_csv(
    OUTPUT_DIR
    / "invalid_latency.csv",
    index=False
)

invalid_sample.to_csv(
    OUTPUT_DIR
    / "invalid_sample.csv",
    index=False
)

invalid_trial.to_csv(
    OUTPUT_DIR
    / "invalid_trial.csv",
    index=False
)

duplicates.to_csv(
    OUTPUT_DIR
    / "duplicate_events.csv",
    index=False
)


# ============================================================
# SAVE VALIDATED EVENT TABLE
# ============================================================

validated_file = (
    OUTPUT_DIR
    / "ALL_EVENTS_VALIDATED.csv"
)

df.to_csv(
    validated_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_section("FINAL VALIDATION SUMMARY")

total_events = len(df)

unique_files = df["file"].nunique()

unique_subjects = df["subject"].nunique()

unique_runs = (
    df[
        ["subject", "run"]
    ]
    .drop_duplicates()
    .shape[0]
)

duplicate_count = len(duplicates)

invalid_latency_count = len(
    invalid_latency
)

invalid_sample_count = len(
    invalid_sample
)

invalid_trial_count = len(
    invalid_trial
)


summary = pd.DataFrame(
    [
        {
            "total_event_rows": total_events,
            "unique_eeg_files": unique_files,
            "unique_subjects": unique_subjects,
            "unique_subject_run_pairs": unique_runs,
            "expected_eeg_files": 83,
            "duplicate_event_rows": duplicate_count,
            "invalid_latency_rows": invalid_latency_count,
            "invalid_sample_rows": invalid_sample_count,
            "invalid_trial_rows": invalid_trial_count
        }
    ]
)

print(
    "Total event rows:",
    total_events
)

print(
    "Unique EEG files:",
    unique_files
)

print(
    "Unique subjects:",
    unique_subjects
)

print(
    "Unique subject/run pairs:",
    unique_runs
)

print(
    "Expected EEG files:",
    83
)

print(
    "Duplicate rows:",
    duplicate_count
)

print(
    "Invalid latency rows:",
    invalid_latency_count
)

print(
    "Invalid sample rows:",
    invalid_sample_count
)

print(
    "Invalid trial rows:",
    invalid_trial_count
)


# ============================================================
# OVERALL STATUS
# ============================================================

print()

if (
    unique_files == 83
    and
    duplicate_count == 0
    and
    invalid_latency_count == 0
    and
    invalid_sample_count == 0
    and
    invalid_trial_count == 0
):

    overall_status = "PASS"

else:

    overall_status = "REVIEW"


print(
    "OVERALL STATUS:",
    overall_status
)

summary[
    "overall_status"
] = overall_status


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = (
    OUTPUT_DIR
    / "VALIDATION_SUMMARY.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# OUTPUT FILES
# ============================================================

print_section("OUTPUT FILES")

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()

for file in sorted(
    OUTPUT_DIR.glob("*.csv")
):

    print(
        " ",
        file.name
    )


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 80)
print("EVENT VALIDATION COMPLETE")
print("=" * 80)
