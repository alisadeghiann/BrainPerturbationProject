# =============================================================================
# EPOCH EXTRACTION - BRAIN PERTURBATION PROJECT
# =============================================================================
#
# Purpose:
#   Create trial-level EEG epochs from the PREPROCESSED EEG files.
#
# IMPORTANT:
#   - Original .SET/.FDT files are NEVER modified.
#   - Preprocessed files are only READ.
#   - No subjects are deleted here.
#   - No trials are deleted here.
#   - All exclusions are recorded in QC reports.
#
# Output:
#   qc/epochs/
#       epoch_metadata.csv
#       epoch_extraction_summary.csv
#       excluded_epochs.csv
#       subject_epoch_summary.csv
#       epochs/
#
# =============================================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd
import mne

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

PREPROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "qc",
    "preprocessed"
)

EVENT_FILE = os.path.join(
    PROJECT_ROOT,
    "qc",
    "events",
    "ALL_EVENTS_83_RUNS.csv"
)

TRIAL_FILE = os.path.join(
    PROJECT_ROOT,
    "qc",
    "trial_table",
    "TRIAL_LEVEL_TABLE.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "qc",
    "epochs"
)

EPOCH_DATA_DIR = os.path.join(
    OUTPUT_DIR,
    "epochs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EPOCH_DATA_DIR, exist_ok=True)

# =============================================================================
# SETTINGS
# =============================================================================

# Epoch around SHOW_LETTER.
#
# -500 ms before stimulus
# 1500 ms after stimulus
#
# This gives us:
# baseline period
# encoding
# early working-memory processing
# maintenance
#
TMIN = -0.5
TMAX = 1.5

BASELINE = (-0.2, 0.0)

# Only these EEG event types are used as primary epoch anchors.
EPOCH_EVENT = "show_letter"

# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("EPOCH EXTRACTION - BRAIN PERTURBATION PROJECT")
print("=" * 80)

print("\nProject:")
print(PROJECT_ROOT)

print("\nPreprocessed directory:")
print(PREPROCESSED_DIR)

print("\nEvent file:")
print(EVENT_FILE)

print("\nTrial table:")
print(TRIAL_FILE)

# =============================================================================
# CHECK FILES
# =============================================================================

if not os.path.exists(PREPROCESSED_DIR):
    raise FileNotFoundError(
        f"Preprocessed directory not found:\n{PREPROCESSED_DIR}"
    )

if not os.path.exists(EVENT_FILE):
    raise FileNotFoundError(
        f"Event file not found:\n{EVENT_FILE}"
    )

if not os.path.exists(TRIAL_FILE):
    raise FileNotFoundError(
        f"Trial table not found:\n{TRIAL_FILE}"
    )

# =============================================================================
# LOAD EVENT DATA
# =============================================================================

print("\n" + "=" * 80)
print("LOADING EVENTS")
print("=" * 80)

events_df = pd.read_csv(EVENT_FILE)

print(f"Event rows: {len(events_df):,}")

print("\nColumns:")
for col in events_df.columns:
    print(f"  {col}")

# =============================================================================
# LOAD TRIAL DATA
# =============================================================================

print("\n" + "=" * 80)
print("LOADING TRIAL TABLE")
print("=" * 80)

trials_df = pd.read_csv(TRIAL_FILE)

print(f"Trial rows: {len(trials_df):,}")

print("\nTrial columns:")
for col in trials_df.columns:
    print(f"  {col}")

# =============================================================================
# NORMALIZE COLUMN NAMES
# =============================================================================

events_df.columns = [
    str(c).strip().lower()
    for c in events_df.columns
]

trials_df.columns = [
    str(c).strip().lower()
    for c in trials_df.columns
]

# =============================================================================
# BASIC VALIDATION
# =============================================================================

required_event_columns = [
    "subject",
    "source_file",
    "trial",
    "type",
    "latency",
    "sample",
    "memory_cond",
    "task_role"
]

missing_events = [
    c for c in required_event_columns
    if c not in events_df.columns
]

if missing_events:
    raise ValueError(
        f"Missing event columns: {missing_events}"
    )

required_trial_columns = [
    "subject",
    "trial"
]

missing_trials = [
    c for c in required_trial_columns
    if c not in trials_df.columns
]

if missing_trials:
    raise ValueError(
        f"Missing trial columns: {missing_trials}"
    )

# =============================================================================
# CLEAN TYPES
# =============================================================================

events_df["subject"] = events_df["subject"].astype(str)
events_df["trial"] = pd.to_numeric(
    events_df["trial"],
    errors="coerce"
)

events_df["latency"] = pd.to_numeric(
    events_df["latency"],
    errors="coerce"
)

events_df["sample"] = pd.to_numeric(
    events_df["sample"],
    errors="coerce"
)

events_df["type"] = (
    events_df["type"]
    .astype(str)
    .str.strip()
)

trials_df["subject"] = trials_df["subject"].astype(str)

trials_df["trial"] = pd.to_numeric(
    trials_df["trial"],
    errors="coerce"
)

# =============================================================================
# SELECT EPOCH EVENTS
# =============================================================================

epoch_events = events_df[
    events_df["type"].str.lower() == EPOCH_EVENT.lower()
].copy()

print("\n" + "=" * 80)
print("EPOCH ANCHOR EVENTS")
print("=" * 80)

print(f"Selected event type: {EPOCH_EVENT}")
print(f"Number of anchor events: {len(epoch_events):,}")

# =============================================================================
# PREPARE TRIAL INFORMATION
# =============================================================================

trial_columns = [
    c for c in [
        "subject",
        "trial",
        "memory_cond",
        "accuracy",
        "bad_trial",
        "event_count",
        "expected_14_events"
    ]
    if c in trials_df.columns
]

trial_info = trials_df[trial_columns].copy()

# Prevent duplicate trial rows from contaminating merge
trial_info = trial_info.drop_duplicates(
    subset=["subject", "trial"]
)

# =============================================================================
# MERGE EVENT + TRIAL INFORMATION
# =============================================================================

epoch_events = epoch_events.merge(
    trial_info,
    on=["subject", "trial"],
    how="left",
    suffixes=("", "_trial")
)

print("\nEpoch events after trial merge:")
print(len(epoch_events))

# =============================================================================
# FIND PREPROCESSED EEG FILES
# =============================================================================

print("\n" + "=" * 80)
print("SEARCHING PREPROCESSED EEG FILES")
print("=" * 80)

set_files = glob.glob(
    os.path.join(
        PREPROCESSED_DIR,
        "**",
        "*.set"
    ),
    recursive=True
)

print(f"Found .SET files: {len(set_files)}")

if len(set_files) == 0:
    raise RuntimeError(
        "No preprocessed .SET files found."
    )

# =============================================================================
# CREATE FILE MAP
# =============================================================================

file_map = {}

for path in set_files:
    filename = os.path.basename(path)
    file_map[filename] = path

# =============================================================================
# EXTRACT SUBJECT FROM FILENAME
# =============================================================================

def get_subject_from_filename(filename):

    base = os.path.basename(filename)

    parts = base.split("_")

    for p in parts:
        if p.startswith("sub-"):
            return p

    return None

# =============================================================================
# EXTRACT RUN
# =============================================================================

def get_run_from_filename(filename):

    base = os.path.basename(filename)

    parts = base.split("_")

    for p in parts:
        if p.startswith("run-"):
            return p.replace(
                "run-",
                ""
            )

    return None

# =============================================================================
# PROCESS FILES
# =============================================================================

metadata_rows = []
excluded_rows = []
summary_rows = []

total_epochs = 0
total_excluded = 0

for file_index, eeg_path in enumerate(
    set_files,
    start=1
):

    filename = os.path.basename(eeg_path)

    subject = get_subject_from_filename(filename)
    run = get_run_from_filename(filename)

    print("\n" + "=" * 80)
    print(
        f"PROCESSING {file_index}/{len(set_files)}"
    )
    print("=" * 80)

    print(filename)

    if subject is None:
        print("WARNING: Could not identify subject.")
        continue

    # -------------------------------------------------------------------------
    # GET EVENTS FOR THIS FILE
    # -------------------------------------------------------------------------

    file_events = epoch_events[
        epoch_events["source_file"].astype(str).apply(
            lambda x: os.path.basename(x) == filename
        )
    ].copy()

    # If source_file matching fails, use subject + run
    if len(file_events) == 0:

        file_events = epoch_events[
            epoch_events["subject"].astype(str) == subject
        ].copy()

        if run is not None and "run" in file_events.columns:

            file_events = file_events[
                file_events["run"].astype(str) == str(run)
            ]

    print(
        f"Anchor events: {len(file_events)}"
    )

    if len(file_events) == 0:

        summary_rows.append({
            "subject": subject,
            "run": run,
            "file": filename,
            "anchor_events": 0,
            "epochs_created": 0,
            "epochs_excluded": 0,
            "status": "NO_ANCHOR_EVENTS"
        })

        continue

    # -------------------------------------------------------------------------
    # LOAD PREPROCESSED EEG
    # -------------------------------------------------------------------------

    try:

        raw = mne.io.read_raw_eeglab(
            eeg_path,
            preload=True,
            verbose=False
        )

    except Exception as e:

        print(
            f"ERROR loading {filename}: {e}"
        )

        summary_rows.append({
            "subject": subject,
            "run": run,
            "file": filename,
            "anchor_events": len(file_events),
            "epochs_created": 0,
            "epochs_excluded": 0,
            "status": "LOAD_ERROR"
        })

        continue

    print(
        f"Channels: {len(raw.ch_names)}"
    )

    print(
        f"Sampling rate: {raw.info['sfreq']}"
    )

    print(
        f"Duration: {raw.times[-1]:.2f} sec"
    )

    # -------------------------------------------------------------------------
    # CONVERT LATENCY TO SAMPLE
    # -------------------------------------------------------------------------

    sfreq = raw.info["sfreq"]

    file_events["latency"] = pd.to_numeric(
        file_events["latency"],
        errors="coerce"
    )

    # EEGLAB latency is generally sample-based and may start at 1.
    # We preserve original latency and generate a zero-based sample index.

    file_events["epoch_sample"] = (
        file_events["latency"]
        .round()
        .astype("Int64")
        - 1
    )

    # -------------------------------------------------------------------------
    # CHECK BOUNDARIES
    # -------------------------------------------------------------------------

    start_offset = int(
        round(TMIN * sfreq)
    )

    stop_offset = int(
        round(TMAX * sfreq)
    )

    valid_event_rows = []
    excluded_file_count = 0

    for idx, row in file_events.iterrows():

        trial = row["trial"]

        sample = row["epoch_sample"]

        # -------------------------------------------------------------
        # Invalid sample
        # -------------------------------------------------------------

        if pd.isna(sample):

            excluded_rows.append({
                "subject": subject,
                "run": run,
                "file": filename,
                "trial": trial,
                "reason": "INVALID_LATENCY",
                "latency": row["latency"]
            })

            excluded_file_count += 1
            continue

        sample = int(sample)

        # -------------------------------------------------------------
        # Boundary check
        # -------------------------------------------------------------

        start = sample + start_offset
        stop = sample + stop_offset

        if start < 0:

            excluded_rows.append({
                "subject": subject,
                "run": run,
                "file": filename,
                "trial": trial,
                "reason": "EPOCH_START_OUT_OF_BOUNDS",
                "latency": row["latency"],
                "sample": sample
            })

            excluded_file_count += 1
            continue

        if stop >= raw.n_times:

            excluded_rows.append({
                "subject": subject,
                "run": run,
                "file": filename,
                "trial": trial,
                "reason": "EPOCH_END_OUT_OF_BOUNDS",
                "latency": row["latency"],
                "sample": sample
            })

            excluded_file_count += 1
            continue

        valid_event_rows.append(
            (idx, sample)
        )

    # -------------------------------------------------------------------------
    # NO VALID EVENTS
    # -------------------------------------------------------------------------

    if len(valid_event_rows) == 0:

        summary_rows.append({
            "subject": subject,
            "run": run,
            "file": filename,
            "anchor_events": len(file_events),
            "epochs_created": 0,
            "epochs_excluded": excluded_file_count,
            "status": "NO_VALID_EPOCHS"
        })

        total_excluded += excluded_file_count

        continue

    # -------------------------------------------------------------------------
    # CREATE MNE EVENTS ARRAY
    # -------------------------------------------------------------------------

    mne_events = []

    event_metadata = []

    for idx, sample in valid_event_rows:

        row = file_events.loc[idx]

        # MNE event sample
        mne_events.append([
            sample,
            0,
            1
        ])

        event_metadata.append(
            row.to_dict()
        )

    mne_events = np.array(
        mne_events,
        dtype=int
    )

    event_id = {
        "show_letter": 1
    }

    # -------------------------------------------------------------------------
    # CREATE EPOCHS
    # -------------------------------------------------------------------------

    print(
        f"Creating {len(mne_events)} epochs..."
    )

    try:

        epochs = mne.Epochs(
            raw,
            mne_events,
            event_id=event_id,
            tmin=TMIN,
            tmax=TMAX,
            baseline=BASELINE,
            preload=True,
            reject_by_annotation=True,
            detrend=None,
            verbose=False
        )

    except Exception as e:

        print(
            f"ERROR creating epochs: {e}"
        )

        summary_rows.append({
            "subject": subject,
            "run": run,
            "file": filename,
            "anchor_events": len(file_events),
            "epochs_created": 0,
            "epochs_excluded": excluded_file_count,
            "status": "EPOCH_ERROR"
        })

        continue

    # -------------------------------------------------------------------------
    # SAVE EPOCHS
    # -------------------------------------------------------------------------

    output_name = (
        filename.replace(
            ".set",
            "-epo.fif"
        )
    )

    output_path = os.path.join(
        EPOCH_DATA_DIR,
        output_name
    )

    epochs.save(
        output_path,
        overwrite=True,
        verbose=False
    )

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------

    actual_n_epochs = len(epochs)

    for epoch_index in range(actual_n_epochs):

        if epoch_index >= len(event_metadata):
            continue

        meta = event_metadata[epoch_index]

        metadata_rows.append({

            "subject": subject,

            "run": run,

            "file": filename,

            "epoch_index": epoch_index,

            "trial": meta.get(
                "trial",
                np.nan
            ),

            "memory_cond": meta.get(
                "memory_cond",
                np.nan
            ),

            "accuracy": meta.get(
                "accuracy",
                np.nan
            ),

            "bad_trial": meta.get(
                "bad_trial",
                np.nan
            ),

            "event_count": meta.get(
                "event_count",
                np.nan
            ),

            "task_role": meta.get(
                "task_role",
                np.nan
            ),

            "letter": meta.get(
                "letter",
                np.nan
            ),

            "event_type": EPOCH_EVENT,

            "epoch_tmin": TMIN,

            "epoch_tmax": TMAX,

            "sampling_rate": sfreq

        })

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    created = actual_n_epochs

    excluded = (
        len(file_events)
        - created
    )

    total_epochs += created
    total_excluded += excluded

    summary_rows.append({

        "subject": subject,

        "run": run,

        "file": filename,

        "anchor_events": len(file_events),

        "epochs_created": created,

        "epochs_excluded": excluded,

        "sampling_rate": sfreq,

        "channels": len(raw.ch_names),

        "duration_sec": raw.times[-1],

        "status": "OK"

    })

    print(
        f"Epochs created: {created}"
    )

    print(
        f"Epochs excluded: {excluded}"
    )

# =============================================================================
# SAVE METADATA
# =============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

metadata_df = pd.DataFrame(
    metadata_rows
)

excluded_df = pd.DataFrame(
    excluded_rows
)

summary_df = pd.DataFrame(
    summary_rows
)

# =============================================================================
# SAVE EPOCH METADATA
# =============================================================================

metadata_path = os.path.join(
    OUTPUT_DIR,
    "epoch_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)

# =============================================================================
# SAVE EXTRACTION SUMMARY
# =============================================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "epoch_extraction_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

# =============================================================================
# SAVE EXCLUDED EPOCHS
# =============================================================================

excluded_path = os.path.join(
    OUTPUT_DIR,
    "excluded_epochs.csv"
)

excluded_df.to_csv(
    excluded_path,
    index=False
)

# =============================================================================
# SUBJECT SUMMARY
# =============================================================================

if len(metadata_df) > 0:

    subject_summary = (
        metadata_df
        .groupby("subject")
        .agg(
            epochs=(
                "epoch_index",
                "count"
            ),
            trials=(
                "trial",
                "nunique"
            ),
            mean_accuracy=(
                "accuracy",
                "mean"
            ),
            memory_conditions=(
                "memory_cond",
                lambda x: ",".join(
                    sorted(
                        x.dropna()
                        .astype(str)
                        .unique()
                    )
                )
            )
        )
        .reset_index()
    )

else:

    subject_summary = pd.DataFrame()

subject_summary_path = os.path.join(
    OUTPUT_DIR,
    "subject_epoch_summary.csv"
)

subject_summary.to_csv(
    subject_summary_path,
    index=False
)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("FINAL EPOCH EXTRACTION SUMMARY")
print("=" * 80)

print(
    f"\nTotal EEG files found: {len(set_files)}"
)

print(
    f"Total epoch anchor events: {len(epoch_events):,}"
)

print(
    f"Total epochs created: {total_epochs:,}"
)

print(
    f"Total epochs excluded: {total_excluded:,}"
)

if len(summary_df) > 0:

    print("\nSTATUS:")

    print(
        summary_df["status"]
        .value_counts()
    )

if len(metadata_df) > 0:

    print("\nEPOCHS BY MEMORY CONDITION:")

    print(
        metadata_df["memory_cond"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nEPOCHS BY ACCURACY:")

    print(
        metadata_df["accuracy"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nEPOCHS BY SUBJECT:")

    print(
        metadata_df["subject"]
        .value_counts()
        .sort_index()
    )

if len(excluded_df) > 0:

    print("\nEXCLUSION REASONS:")

    print(
        excluded_df["reason"]
        .value_counts()
    )

# =============================================================================
# OUTPUTS
# =============================================================================

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    f"\nEpoch metadata:"
    f"\n{metadata_path}"
)

print(
    f"\nExtraction summary:"
    f"\n{summary_path}"
)

print(
    f"\nExcluded epochs:"
    f"\n{excluded_path}"
)

print(
    f"\nSubject summary:"
    f"\n{subject_summary_path}"
)

print(
    f"\nEpoch FIF files:"
    f"\n{EPOCH_DATA_DIR}"
)

print("\n" + "=" * 80)
print("EPOCH EXTRACTION COMPLETE")
print("=" * 80)

print("\nIMPORTANT:")
print("- Original EEG files were NOT modified.")
print("- Preprocessed EEG files were NOT modified.")
print("- No subjects were deleted.")
print("- No trials were deleted.")
print("- Epoch exclusions are recorded in excluded_epochs.csv.")

print("\nNEXT STEP:")
print("ERP + TIME-FREQUENCY + THETA/ALPHA/BETA FEATURE EXTRACTION")

print("=" * 80)