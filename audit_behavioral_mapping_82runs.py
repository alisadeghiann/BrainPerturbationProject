# ============================================================
# AUDIT BEHAVIORAL MAPPING - 82 RUNS
# ============================================================
#
# PURPOSE:
#   Audit whether original BIDS behavioral events can be
#   correctly matched to V5 harmonized EEG epochs.
#
# IMPORTANT:
#   READ-ONLY.
#   No EEG/FIF/TSV/CSV files are modified.
#   No files are deleted.
#   No epochs are removed.
#
# INPUT:
#   Original BIDS events.tsv
#   V5 harmonized epochs
#
# OUTPUT:
#   CSV audit reports only.
#
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import mne


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

BIDS_DIR = BASE_DIR / "data"

V5_DIR = (
    BASE_DIR
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v5"
    / "ELIGIBLE"
)

OUTPUT_DIR = (
    BASE_DIR
    / "qc"
    / "behavioral_mapping_audit_v5"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_FILES = 82

EXPECTED_SFREQ = 500.0
EXPECTED_CHANNELS = 71
EXPECTED_TIMES = 501

# Maximum timing difference allowed between an EEG epoch
# event sample and the corresponding BIDS event sample.
#
# We start conservatively.
MAX_SAMPLE_DIFF = 5


# ============================================================
# HELPER: FIND SUBJECT/RUN FROM FILENAME
# ============================================================

def parse_v5_filename(filename):

    # Example:
    # sub-024_run-01_harmonized_epo.fif

    stem = filename.replace(
        "_harmonized_epo.fif",
        ""
    )

    parts = stem.split("_")

    subject = None
    run = None

    for part in parts:

        if part.startswith("sub-"):
            subject = part

        elif part.startswith("run-"):
            run = part.replace(
                "run-",
                ""
            )

    return subject, run


# ============================================================
# FIND V5 FILES
# ============================================================

v5_files = sorted(
    V5_DIR.glob(
        "*_harmonized_epo.fif"
    )
)

print("=" * 100)
print("BEHAVIORAL MAPPING AUDIT - V5")
print("=" * 100)
print()

print("Project:")
print(BASE_DIR)
print()

print("V5 directory:")
print(V5_DIR)
print()

print("V5 files found:")
print(len(v5_files))
print()

print("Expected:")
print(EXPECTED_FILES)
print()


if len(v5_files) != EXPECTED_FILES:

    raise RuntimeError(
        f"Expected {EXPECTED_FILES} V5 files "
        f"but found {len(v5_files)}."
    )


# ============================================================
# GLOBAL RECORDS
# ============================================================

run_records = []
event_records = []

total_epochs = 0
total_bids_events = 0
total_matched = 0
total_unmatched = 0


# ============================================================
# PROCESS EACH RUN
# ============================================================

for idx, v5_file in enumerate(
    v5_files,
    start=1
):

    print()
    print("=" * 100)
    print(
        f"[{idx}/{len(v5_files)}] "
        f"{v5_file.name}"
    )
    print("=" * 100)

    subject, run = parse_v5_filename(
        v5_file.name
    )

    if subject is None or run is None:

        print(
            "WARNING: Could not parse subject/run."
        )

        continue

    run_int = int(run)

    # --------------------------------------------------------
    # ORIGINAL BIDS EVENTS FILE
    # --------------------------------------------------------

    events_file = (
        BIDS_DIR
        / subject
        / "ses-01"
        / "eeg"
        / (
            f"{subject}_ses-01_"
            f"task-WorkingMemory_"
            f"run-{run_int}_events.tsv"
        )
    )

    print()
    print("BIDS events:")
    print(events_file)

    if not events_file.exists():

        print(
            "ERROR: BIDS events file does not exist."
        )

        run_records.append({

            "subject": subject,
            "run": run_int,
            "v5_file": v5_file.name,
            "events_file": str(events_file),
            "status": "MISSING_EVENTS_FILE",

            "epochs": np.nan,
            "bids_events": np.nan,
            "matched": np.nan,
            "unmatched": np.nan,

            "remember": np.nan,
            "ignore": np.nan,
            "probe_target": np.nan,
            "probe_not_shown": np.nan,
            "work_memory": np.nan,

        })

        continue


    # --------------------------------------------------------
    # READ TSV
    # --------------------------------------------------------

    events_df = pd.read_csv(
        events_file,
        sep="\t"
    )

    total_bids_events += len(
        events_df
    )

    print(
        f"BIDS events: {len(events_df)}"
    )


    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "onset",
        "duration",
        "sample",
        "event_type",
        "task_role",
        "trial",
        "memory_cond",
        "value",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in events_df.columns
    ]

    if missing_columns:

        print(
            "ERROR: Missing columns:"
        )

        print(
            missing_columns
        )

        run_records.append({

            "subject": subject,
            "run": run_int,
            "v5_file": v5_file.name,
            "events_file": str(events_file),
            "status": "MISSING_COLUMNS",

            "epochs": np.nan,
            "bids_events": len(events_df),
            "matched": np.nan,
            "unmatched": np.nan,

            "remember": np.nan,
            "ignore": np.nan,
            "probe_target": np.nan,
            "probe_not_shown": np.nan,
            "work_memory": np.nan,

        })

        continue


    # --------------------------------------------------------
    # READ V5 EPOCHS
    # --------------------------------------------------------

    print()
    print("Reading V5 epochs...")

    epochs = mne.read_epochs(
        v5_file,
        preload=False,
        verbose=False
    )

    n_epochs = len(
        epochs
    )

    n_channels = len(
        epochs.ch_names
    )

    n_times = len(
        epochs.times
    )

    sfreq = float(
        epochs.info["sfreq"]
    )

    total_epochs += n_epochs

    print(
        f"Epochs:   {n_epochs}"
    )

    print(
        f"Channels: {n_channels}"
    )

    print(
        f"SFREQ:    {sfreq}"
    )

    print(
        f"Samples:  {n_times}"
    )


    # --------------------------------------------------------
    # STRUCTURE CHECK
    # --------------------------------------------------------

    structure_ok = (

        n_channels == EXPECTED_CHANNELS

        and n_times == EXPECTED_TIMES

        and np.isclose(
            sfreq,
            EXPECTED_SFREQ,
            atol=1e-3
        )
    )


    if not structure_ok:

        print(
            "WARNING: V5 structural mismatch."
        )


    # --------------------------------------------------------
    # EEG EVENT SAMPLES
    # --------------------------------------------------------

    eeg_event_samples = (
        epochs.events[:, 0].astype(int)
    )

    eeg_event_codes = (
        epochs.events[:, 2].astype(int)
    )


    # --------------------------------------------------------
    # EVENT ID
    # --------------------------------------------------------

    event_id_reverse = {

        int(code): name

        for name, code
        in epochs.event_id.items()

    }


    # --------------------------------------------------------
    # BIDS SAMPLE COLUMN
    # --------------------------------------------------------

    bids_samples = pd.to_numeric(
        events_df["sample"],
        errors="coerce"
    )

    events_df = events_df.copy()

    events_df["_sample_numeric"] = (
        bids_samples
    )

    events_df = events_df[
        events_df["_sample_numeric"].notna()
    ].copy()

    events_df[
        "_sample_numeric"
    ] = events_df[
        "_sample_numeric"
    ].astype(int)


    # --------------------------------------------------------
    # MATCH EACH EEG EPOCH TO BIDS EVENT
    # --------------------------------------------------------

    used_bids_indices = set()

    matched_count = 0
    unmatched_count = 0

    remember_count = 0
    ignore_count = 0
    probe_target_count = 0
    probe_not_shown_count = 0
    work_memory_count = 0


    for epoch_index, eeg_sample in enumerate(
        eeg_event_samples
    ):

        eeg_code = int(
            eeg_event_codes[
                epoch_index
            ]
        )

        eeg_condition = (
            event_id_reverse.get(
                eeg_code,
                "UNKNOWN"
            )
        )


        # ----------------------------------------------------
        # FIND CLOSEST UNUSED BIDS EVENT
        # ----------------------------------------------------

        candidate_distances = (
            np.abs(
                events_df[
                    "_sample_numeric"
                ].values
                - eeg_sample
            )
        )

        candidate_order = np.argsort(
            candidate_distances
        )

        matched_idx = None

        for candidate_position in candidate_order:

            original_index = (
                events_df.index[
                    candidate_position
                ]
            )

            if original_index in used_bids_indices:
                continue

            distance = int(
                candidate_distances[
                    candidate_position
                ]
            )

            if distance <= MAX_SAMPLE_DIFF:

                matched_idx = original_index
                break


        # ----------------------------------------------------
        # UNMATCHED
        # ----------------------------------------------------

        if matched_idx is None:

            unmatched_count += 1

            total_unmatched += 1

            event_records.append({

                "subject": subject,
                "run": run_int,

                "epoch_index": epoch_index,

                "eeg_sample": eeg_sample,

                "eeg_event_code": eeg_code,

                "eeg_event_condition":
                    eeg_condition,

                "matched": False,

                "bids_index": np.nan,

                "bids_sample": np.nan,

                "sample_difference": np.nan,

                "bids_event_type": "",

                "task_role": "",

                "letter": "",

                "trial": np.nan,

                "memory_cond": "",

                "value": "",

            })

            continue


        # ----------------------------------------------------
        # MATCHED
        # ----------------------------------------------------

        used_bids_indices.add(
            matched_idx
        )

        matched_count += 1
        total_matched += 1

        row = events_df.loc[
            matched_idx
        ]

        bids_sample = int(
            row["_sample_numeric"]
        )

        sample_difference = (
            eeg_sample
            - bids_sample
        )

        task_role = str(
            row["task_role"]
        )

        event_type = str(
            row["event_type"]
        )

        letter = (
            row["letter"]
            if "letter" in row.index
            else ""
        )

        trial = row["trial"]

        memory_cond = str(
            row["memory_cond"]
        )

        value = str(
            row["value"]
        )


        # ----------------------------------------------------
        # CONDITION COUNTS
        # ----------------------------------------------------

        if task_role == "to_remember":

            remember_count += 1

        if task_role == "to_ignore":

            ignore_count += 1

        if task_role == "probe_target":

            probe_target_count += 1

        if task_role == "probe_not_shown":

            probe_not_shown_count += 1

        if task_role == "work_memory":

            work_memory_count += 1


        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        event_records.append({

            "subject": subject,
            "run": run_int,

            "epoch_index": epoch_index,

            "eeg_sample": eeg_sample,

            "eeg_event_code": eeg_code,

            "eeg_event_condition":
                eeg_condition,

            "matched": True,

            "bids_index": matched_idx,

            "bids_sample": bids_sample,

            "sample_difference":
                sample_difference,

            "bids_event_type":
                event_type,

            "task_role":
                task_role,

            "letter":
                letter,

            "trial":
                trial,

            "memory_cond":
                memory_cond,

            "value":
                value,

        })


    # --------------------------------------------------------
    # RUN SUMMARY
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("RUN SUMMARY")
    print("-" * 100)

    print(
        f"Epochs:              {n_epochs}"
    )

    print(
        f"BIDS events:         {len(events_df)}"
    )

    print(
        f"Matched:              {matched_count}"
    )

    print(
        f"Unmatched:            {unmatched_count}"
    )

    print(
        f"Remember:             {remember_count}"
    )

    print(
        f"Ignore:               {ignore_count}"
    )

    print(
        f"Probe target:         {probe_target_count}"
    )

    print(
        f"Probe not shown:      {probe_not_shown_count}"
    )

    print(
        f"Work memory:          {work_memory_count}"
    )


    # --------------------------------------------------------
    # MAPPING STATUS
    # --------------------------------------------------------

    if (
        matched_count == n_epochs
    ):

        mapping_status = "PASS"

    else:

        mapping_status = "REVIEW"


    run_records.append({

        "subject": subject,
        "run": run_int,

        "v5_file":
            v5_file.name,

        "events_file":
            str(events_file),

        "status":
            mapping_status,

        "epochs":
            n_epochs,

        "bids_events":
            len(events_df),

        "matched":
            matched_count,

        "unmatched":
            unmatched_count,

        "remember":
            remember_count,

        "ignore":
            ignore_count,

        "probe_target":
            probe_target_count,

        "probe_not_shown":
            probe_not_shown_count,

        "work_memory":
            work_memory_count,

        "structure_ok":
            structure_ok,

    })


# ============================================================
# SAVE EVENT-LEVEL AUDIT
# ============================================================

event_audit_df = pd.DataFrame(
    event_records
)

event_audit_path = (
    OUTPUT_DIR
    / "event_level_mapping_audit_v5.csv"
)

event_audit_df.to_csv(
    event_audit_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE RUN-LEVEL AUDIT
# ============================================================

run_audit_df = pd.DataFrame(
    run_records
)

run_audit_path = (
    OUTPUT_DIR
    / "run_level_mapping_audit_v5.csv"
)

run_audit_df.to_csv(
    run_audit_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# CONDITION SUMMARY
# ============================================================

condition_summary = (

    event_audit_df[
        event_audit_df["matched"] == True
    ]

    .groupby(
        [
            "subject",
            "run",
            "task_role"
        ]
    )

    .size()

    .reset_index(
        name="count"
    )

)

condition_summary_path = (
    OUTPUT_DIR
    / "condition_summary_v5.csv"
)

condition_summary.to_csv(
    condition_summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

passed_runs = int(
    (
        run_audit_df["status"]
        == "PASS"
    ).sum()
)

review_runs = int(
    (
        run_audit_df["status"]
        == "REVIEW"
    ).sum()
)


summary_path = (
    OUTPUT_DIR
    / "behavioral_mapping_audit_v5_summary.txt"
)


summary_lines = []

summary_lines.append(
    "=" * 100
)

summary_lines.append(
    "BEHAVIORAL MAPPING AUDIT V5"
)

summary_lines.append(
    "=" * 100
)

summary_lines.append("")

summary_lines.append(
    f"V5 files found:      {len(v5_files)}"
)

summary_lines.append(
    f"Expected files:      {EXPECTED_FILES}"
)

summary_lines.append(
    f"Runs PASS:           {passed_runs}"
)

summary_lines.append(
    f"Runs REVIEW:         {review_runs}"
)

summary_lines.append("")

summary_lines.append(
    f"Total V5 epochs:     {total_epochs}"
)

summary_lines.append(
    f"Total BIDS events:   {total_bids_events}"
)

summary_lines.append(
    f"Total matched:       {total_matched}"
)

summary_lines.append(
    f"Total unmatched:     {total_unmatched}"
)

summary_lines.append("")

summary_lines.append(
    "=" * 100
)

summary_lines.append(
    "IMPORTANT"
)

summary_lines.append(
    "=" * 100
)

summary_lines.append("")

summary_lines.append(
    "This was a READ-ONLY audit."
)

summary_lines.append(
    "No EEG/FIF/TSV/CSV input file was modified."
)

summary_lines.append(
    "No epochs were removed."
)

summary_lines.append(
    "No original V5 file was modified."
)

summary_lines.append("")

summary_lines.append(
    "Behavioral conditions were read from the "
    "original BIDS events.tsv files."
)

summary_lines.append(
    "EEG epochs were matched to BIDS events "
    "using the event sample number."
)

summary_lines.append(
    f"Maximum allowed sample difference: "
    f"{MAX_SAMPLE_DIFF}"
)

summary_lines.append("")

summary_lines.append(
    "=" * 100
)

summary_lines.append(
    "DECISION"
)

summary_lines.append(
    "=" * 100
)

summary_lines.append("")

if (
    passed_runs == EXPECTED_FILES
    and total_unmatched == 0
):

    final_status = (
        "PASS - ALL V5 EPOCHS "
        "CAN BE MATCHED TO BIDS EVENTS"
    )

else:

    final_status = (
        "REVIEW - BEHAVIORAL MAPPING "
        "REQUIRES INVESTIGATION"
    )

summary_lines.append(
    final_status
)

summary_lines.append("")

summary_lines.append(
    "Output files:"
)

summary_lines.append(
    str(run_audit_path)
)

summary_lines.append(
    str(event_audit_path)
)

summary_lines.append(
    str(condition_summary_path)
)

summary_lines.append(
    str(summary_path)
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            summary_lines
        )
    )


# ============================================================
# FINAL CONSOLE
# ============================================================

print()
print("=" * 100)
print("BEHAVIORAL MAPPING AUDIT COMPLETE")
print("=" * 100)

print()

print(
    f"V5 files:             {len(v5_files)}"
)

print(
    f"PASS runs:             {passed_runs}"
)

print(
    f"REVIEW runs:           {review_runs}"
)

print(
    f"Total epochs:          {total_epochs}"
)

print(
    f"Total BIDS events:     {total_bids_events}"
)

print(
    f"Total matched:         {total_matched}"
)

print(
    f"Total unmatched:       {total_unmatched}"
)

print()

print(
    "FINAL STATUS:"
)

print(
    final_status
)

print()

print(
    "Saved:"
)

print(
    run_audit_path
)

print(
    event_audit_path
)

print(
    condition_summary_path
)

print(
    summary_path
)

print()

print(
    "READ-ONLY AUDIT."
)

print(
    "NO ORIGINAL EEG/FIF/TSV/CSV FILES WERE MODIFIED."
)

print("=" * 100)