from pathlib import Path
import numpy as np
import pandas as pd
import mne


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

INPUT_DIR = (
    BASE_DIR
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v5"
    / "ELIGIBLE"
)

OUTPUT_DIR = (
    INPUT_DIR
    / "event_distribution_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

EPOCH_CSV = (
    OUTPUT_DIR
    / "event_distribution_epoch_level.csv"
)

FILE_CSV = (
    OUTPUT_DIR
    / "event_distribution_file_level.csv"
)

SUBJECT_CSV = (
    OUTPUT_DIR
    / "event_distribution_subject_level.csv"
)

RUN_CSV = (
    OUTPUT_DIR
    / "event_distribution_run_level.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "event_distribution_v5_summary.txt"
)


# ============================================================
# CANONICAL EVENT MAPPING
# ============================================================

EVENT_MAP = {
    1: "left_click",
    2: "right_click",
    3: "show_cross",
    4: "show_dash",
    5: "show_letter",
    6: "sound_beep",
    7: "sound_buzz",
    8: "right_click/show_cross",
}


# ============================================================
# FIND FILES
# ============================================================

files = sorted(
    INPUT_DIR.glob(
        "*_harmonized_epo.fif"
    )
)


print("=" * 80)
print("EVENT DISTRIBUTION ANALYSIS - V5")
print("=" * 80)

print()
print(
    f"Input directory:\n{INPUT_DIR}"
)

print()
print(
    f"Files found: {len(files)}"
)

print()


if len(files) != 82:

    print(
        "WARNING: Expected 82 files."
    )

    print(
        f"Found {len(files)} files."
    )

    print()


# ============================================================
# RECORDS
# ============================================================

epoch_records = []
file_records = []


# ============================================================
# PROCESS FILES
# ============================================================

for i, file_path in enumerate(
    files,
    start=1
):

    print(
        f"[{i}/{len(files)}] "
        f"{file_path.name}"
    )

    try:

        epochs = mne.read_epochs(
            file_path,
            preload=False,
            verbose=False
        )

        event_ids = (
            epochs.events[:, 2]
            .astype(int)
        )

        # ----------------------------------------------------
        # SUBJECT / RUN
        # ----------------------------------------------------

        filename = file_path.name

        subject = "unknown"
        run = "unknown"

        parts = filename.split("_")

        for part in parts:

            if part.startswith("sub-"):

                subject = part

            if part.startswith("run-"):

                run = part


        # ----------------------------------------------------
        # FILE RECORD
        # ----------------------------------------------------

        file_record = {

            "file": filename,

            "subject": subject,

            "run": run,

            "epochs": len(epochs),

            "unique_event_ids":
                len(np.unique(event_ids)),

        }


        # ----------------------------------------------------
        # EVENT COUNTS
        # ----------------------------------------------------

        for event_id, name in EVENT_MAP.items():

            count = int(
                np.sum(
                    event_ids == event_id
                )
            )

            file_record[
                name
            ] = count


        file_records.append(
            file_record
        )


        # ----------------------------------------------------
        # EPOCH LEVEL RECORDS
        # ----------------------------------------------------

        for epoch_index, event_id in enumerate(
            event_ids
        ):

            event_id = int(
                event_id
            )

            condition = EVENT_MAP.get(
                event_id,
                "UNKNOWN"
            )

            epoch_records.append({

                "file": filename,

                "subject": subject,

                "run": run,

                "epoch_index": epoch_index,

                "event_id": event_id,

                "condition": condition

            })


    except Exception as e:

        print(
            "ERROR:",
            e
        )


# ============================================================
# DATAFRAMES
# ============================================================

epoch_df = pd.DataFrame(
    epoch_records
)

file_df = pd.DataFrame(
    file_records
)


# ============================================================
# SUBJECT LEVEL
# ============================================================

if len(epoch_df) > 0:

    subject_df = (
        epoch_df
        .groupby(
            [
                "subject",
                "condition"
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

else:

    subject_df = pd.DataFrame()


# ============================================================
# RUN LEVEL
# ============================================================

if len(epoch_df) > 0:

    run_df = (
        epoch_df
        .groupby(
            [
                "subject",
                "run",
                "condition"
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

else:

    run_df = pd.DataFrame()


# ============================================================
# SAVE
# ============================================================

epoch_df.to_csv(
    EPOCH_CSV,
    index=False,
    encoding="utf-8-sig"
)

file_df.to_csv(
    FILE_CSV,
    index=False,
    encoding="utf-8-sig"
)

subject_df.to_csv(
    SUBJECT_CSV,
    index=False,
    encoding="utf-8-sig"
)

run_df.to_csv(
    RUN_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# GLOBAL COUNTS
# ============================================================

global_counts = (
    epoch_df["condition"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.append(
    "=" * 80
)

summary.append(
    "EVENT DISTRIBUTION ANALYSIS - V5"
)

summary.append(
    "=" * 80
)

summary.append("")

summary.append(
    f"Files analyzed: {len(files)}"
)

summary.append(
    f"Total epochs:   {len(epoch_df)}"
)

summary.append("")

summary.append(
    "=" * 80
)

summary.append(
    "GLOBAL CONDITION COUNTS"
)

summary.append(
    "=" * 80
)

summary.append("")

for condition, count in (
    global_counts.items()
):

    percentage = (
        100.0
        * count
        / len(epoch_df)
    )

    summary.append(
        f"{condition:25s} "
        f"{count:6d} "
        f"({percentage:6.2f}%)"
    )


summary.append("")

summary.append(
    "=" * 80
)

summary.append(
    "EVENT ID COUNTS"
)

summary.append(
    "=" * 80
)

summary.append("")

event_counts = (
    epoch_df["event_id"]
    .value_counts()
    .sort_index()
)

for event_id, count in (
    event_counts.items()
):

    name = EVENT_MAP.get(
        int(event_id),
        "UNKNOWN"
    )

    summary.append(
        f"ID {event_id} | "
        f"{name:25s} | "
        f"{count}"
    )


summary.append("")

summary.append(
    "=" * 80
)

summary.append(
    "SUBJECTS"
)

summary.append(
    "=" * 80
)

subjects = sorted(
    epoch_df["subject"]
    .unique()
)

summary.append(
    f"Subjects found: {len(subjects)}"
)

summary.append("")

for subject in subjects:

    total = int(
        (
            epoch_df["subject"]
            == subject
        ).sum()
    )

    summary.append(
        f"{subject}: {total} epochs"
    )


summary.append("")

summary.append(
    "=" * 80
)

summary.append(
    "AMBIGUOUS EVENT"
)

summary.append(
    "=" * 80
)

ambiguous_count = int(
    (
        epoch_df["event_id"]
        == 8
    ).sum()
)

summary.append(
    f"ID 8 count: {ambiguous_count}"
)

summary.append(
    "ID 8 = right_click/show_cross"
)

summary.append("")

summary.append(
    "=" * 80
)

summary.append(
    "DATA INTEGRITY"
)

summary.append(
    "=" * 80
)

summary.append("")

summary.append(
    f"Expected epochs: 21816"
)

summary.append(
    f"Observed epochs: {len(epoch_df)}"
)

if len(epoch_df) == 21816:

    summary.append(
        "Epoch count: PASS"
    )

else:

    summary.append(
        "Epoch count: REVIEW"
    )


summary.append("")

summary.append(
    "NO DATA WAS MODIFIED."
)

summary.append(
    "READ-ONLY EVENT DISTRIBUTION ANALYSIS."
)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print()

print("=" * 80)
print("GLOBAL CONDITION COUNTS")
print("=" * 80)

print()

print(
    global_counts
)

print()

print("=" * 80)
print("EVENT ID COUNTS")
print("=" * 80)

print()

for event_id, count in (
    event_counts.items()
):

    name = EVENT_MAP.get(
        int(event_id),
        "UNKNOWN"
    )

    print(
        f"ID {event_id} | "
        f"{name} | "
        f"{count}"
    )


print()

print("=" * 80)
print("FINAL")
print("=" * 80)

print()

print(
    f"Files analyzed: {len(files)}"
)

print(
    f"Total epochs:   {len(epoch_df)}"
)

print()

print(
    "Saved:"
)

print(
    EPOCH_CSV
)

print(
    FILE_CSV
)

print(
    SUBJECT_CSV
)

print(
    RUN_CSV
)

print(
    SUMMARY_TXT
)

print()

print(
    "NO DATA WAS MODIFIED."
)

print("=" * 80)