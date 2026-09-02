from pathlib import Path
from collections import Counter

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
    / "trial_structure_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EVENT MAP
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
# FILES
# ============================================================

files = sorted(
    INPUT_DIR.glob(
        "*_harmonized_epo.fif"
    )
)

print("=" * 80)
print("TRIAL STRUCTURE INSPECTION - V5")
print("=" * 80)

print()
print(f"Files found: {len(files)}")
print()


# ============================================================
# RECORDS
# ============================================================

transition_records = []
file_summary = []


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

        names = [
            EVENT_MAP.get(
                int(x),
                f"UNKNOWN_{x}"
            )
            for x in event_ids
        ]

        # ----------------------------------------------------
        # SUBJECT / RUN
        # ----------------------------------------------------

        parts = file_path.name.split("_")

        subject = next(
            (
                p
                for p in parts
                if p.startswith("sub-")
            ),
            "unknown"
        )

        run = next(
            (
                p
                for p in parts
                if p.startswith("run-")
            ),
            "unknown"
        )


        # ----------------------------------------------------
        # BASIC COUNTS
        # ----------------------------------------------------

        counts = Counter(names)

        rec = {

            "file": file_path.name,

            "subject": subject,

            "run": run,

            "epochs": len(names),

        }

        for event_id, name in EVENT_MAP.items():

            rec[name] = counts.get(
                name,
                0
            )

        file_summary.append(rec)


        # ----------------------------------------------------
        # TRANSITIONS
        # ----------------------------------------------------

        for j in range(
            len(names) - 1
        ):

            current = names[j]

            next_event = names[j + 1]

            transition_records.append({

                "file": file_path.name,

                "subject": subject,

                "run": run,

                "epoch_index": j,

                "current_event": current,

                "next_event": next_event,

                "transition":
                    f"{current} -> {next_event}"

            })


    except Exception as e:

        print(
            "ERROR:",
            e
        )


# ============================================================
# DATAFRAMES
# ============================================================

transition_df = pd.DataFrame(
    transition_records
)

file_df = pd.DataFrame(
    file_summary
)


# ============================================================
# TRANSITION COUNTS
# ============================================================

if len(transition_df) > 0:

    transition_counts = (
        transition_df[
            "transition"
        ]
        .value_counts()
        .reset_index()
    )

    transition_counts.columns = [
        "transition",
        "count"
    ]

else:

    transition_counts = pd.DataFrame(
        columns=[
            "transition",
            "count"
        ]
    )


# ============================================================
# SUBJECT / RUN SEQUENCES
# ============================================================

sequence_records = []

for i, file_path in enumerate(
    files,
    start=1
):

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

        names = [
            EVENT_MAP.get(
                int(x),
                f"UNKNOWN_{x}"
            )
            for x in event_ids
        ]

        parts = file_path.name.split("_")

        subject = next(
            (
                p
                for p in parts
                if p.startswith("sub-")
            ),
            "unknown"
        )

        run = next(
            (
                p
                for p in parts
                if p.startswith("run-")
            ),
            "unknown"
        )

        sequence_records.append({

            "file": file_path.name,

            "subject": subject,

            "run": run,

            "sequence_length":
                len(names),

            "first_event":
                names[0]
                if names
                else "NONE",

            "last_event":
                names[-1]
                if names
                else "NONE",

            "sequence":
                " | ".join(names)

        })

    except Exception as e:

        print(
            "SEQUENCE ERROR:",
            e
        )


sequence_df = pd.DataFrame(
    sequence_records
)


# ============================================================
# SAVE
# ============================================================

TRANSITION_CSV = (
    OUTPUT_DIR
    / "event_transitions_v5.csv"
)

TRANSITION_COUNTS_CSV = (
    OUTPUT_DIR
    / "event_transition_counts_v5.csv"
)

FILE_CSV = (
    OUTPUT_DIR
    / "trial_structure_file_summary_v5.csv"
)

SEQUENCE_CSV = (
    OUTPUT_DIR
    / "event_sequences_v5.csv"
)

SUMMARY_TXT = (
    OUTPUT_DIR
    / "trial_structure_v5_summary.txt"
)


transition_df.to_csv(
    TRANSITION_CSV,
    index=False,
    encoding="utf-8-sig"
)

transition_counts.to_csv(
    TRANSITION_COUNTS_CSV,
    index=False,
    encoding="utf-8-sig"
)

file_df.to_csv(
    FILE_CSV,
    index=False,
    encoding="utf-8-sig"
)

sequence_df.to_csv(
    SEQUENCE_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.append("=" * 80)
summary.append(
    "TRIAL STRUCTURE INSPECTION - V5"
)
summary.append("=" * 80)

summary.append("")

summary.append(
    f"Files analyzed: {len(files)}"
)

summary.append(
    f"Total transition records: "
    f"{len(transition_df)}"
)

summary.append("")

summary.append("=" * 80)
summary.append(
    "MOST COMMON EVENT TRANSITIONS"
)
summary.append("=" * 80)

summary.append("")

for _, row in (
    transition_counts
    .head(30)
    .iterrows()
):

    summary.append(
        f"{row['transition']:45s}"
        f" | {row['count']}"
    )


summary.append("")

summary.append("=" * 80)
summary.append(
    "FIRST EVENT DISTRIBUTION"
)
summary.append("=" * 80)

summary.append("")

if len(sequence_df) > 0:

    first_counts = (
        sequence_df[
            "first_event"
        ]
        .value_counts()
    )

    for name, count in (
        first_counts.items()
    ):

        summary.append(
            f"{name:30s} | {count}"
        )


summary.append("")

summary.append("=" * 80)
summary.append(
    "LAST EVENT DISTRIBUTION"
)
summary.append("=" * 80)

summary.append("")

if len(sequence_df) > 0:

    last_counts = (
        sequence_df[
            "last_event"
        ]
        .value_counts()
    )

    for name, count in (
        last_counts.items()
    ):

        summary.append(
            f"{name:30s} | {count}"
        )


summary.append("")

summary.append("=" * 80)
summary.append(
    "IMPORTANT"
)
summary.append("=" * 80)

summary.append("")

summary.append(
    "ID 8 = right_click/show_cross"
)

summary.append(
    "ID 8 is retained for audit purposes."
)

summary.append(
    "No EEG data was modified."
)

summary.append(
    "This script performs read-only "
    "trial structure inspection."
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
# CONSOLE
# ============================================================

print()

print("=" * 80)
print("MOST COMMON EVENT TRANSITIONS")
print("=" * 80)

print()

print(
    transition_counts
    .head(30)
    .to_string(
        index=False
    )
)

print()

print("=" * 80)
print("FIRST EVENT DISTRIBUTION")
print("=" * 80)

print()

if len(sequence_df) > 0:

    print(
        sequence_df[
            "first_event"
        ]
        .value_counts()
    )

print()

print("=" * 80)
print("LAST EVENT DISTRIBUTION")
print("=" * 80)

print()

if len(sequence_df) > 0:

    print(
        sequence_df[
            "last_event"
        ]
        .value_counts()
    )

print()

print("=" * 80)
print("DONE")
print("=" * 80)

print()

print("Saved:")

print(
    TRANSITION_CSV
)

print(
    TRANSITION_COUNTS_CSV
)

print(
    FILE_CSV
)

print(
    SEQUENCE_CSV
)

print(
    SUMMARY_TXT
)

print()

print(
    "NO DATA WAS MODIFIED."
)