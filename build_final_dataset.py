from pathlib import Path
import pandas as pd
import mne

# ============================================================
# CONFIG
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

V5_DIR = (
    BASE
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v5"
    / "ELIGIBLE"
)

BIDS_EVENTS_DIR = BASE / "data"
QC_EVENTS_DIR = BASE / "qc" / "events"

OUTPUT_DIR = BASE / "final_dataset" / "perturbation"
OUTPUT_EPOCHS = OUTPUT_DIR / "epochs"
OUTPUT_LOGS = OUTPUT_DIR / "logs"

OUTPUT_EPOCHS.mkdir(parents=True, exist_ok=True)
OUTPUT_LOGS.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND V5 FILES
# ============================================================

v5_files = sorted(V5_DIR.glob("*_harmonized_epo.fif"))

print("=" * 90)
print("FINAL PERTURBATION DATASET BUILD - CORRECTED")
print("=" * 90)

print(f"V5 files found: {len(v5_files)}")

if len(v5_files) != 82:
    print("WARNING: Expected 82 V5 files.")


# ============================================================
# EVENT SOURCE
# IMPORTANT:
# BIDS filenames use run-1, run-2, etc.
# ============================================================

def get_event_source(subject, run):

    run_int = int(run)

    # Try both possible naming conventions
    run_formats = [
        str(run_int),
        f"{run_int:02d}"
    ]

    # --------------------------------------------------------
    # STANDARD BIDS EVENTS
    # --------------------------------------------------------

    for run_name in run_formats:

        bids_path = (
            BIDS_EVENTS_DIR
            / subject
            / "ses-01"
            / "eeg"
            / f"{subject}_ses-01_task-WorkingMemory_run-{run_name}_events.tsv"
        )

        if bids_path.exists() and bids_path.stat().st_size > 0:
            return bids_path, "BIDS"

    # --------------------------------------------------------
    # FALLBACK QC EVENTS
    # --------------------------------------------------------

    for run_name in run_formats:

        qc_path = (
            QC_EVENTS_DIR
            / f"{subject}_ses-01_task-WorkingMemory_run-{run_name}_eeg_events.csv"
        )

        if qc_path.exists() and qc_path.stat().st_size > 0:
            return qc_path, "QC"

    return None, "NONE"


# ============================================================
# PARSE SUBJECT / RUN
# ============================================================

def parse_v5_filename(path):

    parts = path.name.split("_")

    subject = None
    run = None

    for part in parts:

        if part.startswith("sub-"):
            subject = part

        elif part.startswith("run-"):
            run = part.replace("run-", "")

    if subject is None or run is None:
        raise ValueError(
            f"Could not parse subject/run from: {path.name}"
        )

    return subject, run


# ============================================================
# PROCESS ALL 82 RUNS
# ============================================================

summary = []
errors = []

total_epochs = 0
total_saved = 0

for i, v5_file in enumerate(v5_files, start=1):

    subject, run = parse_v5_filename(v5_file)

    print()
    print("-" * 90)
    print(f"[{i}/{len(v5_files)}] {subject} | run-{run}")
    print("-" * 90)

    # --------------------------------------------------------
    # FIND EVENTS
    # --------------------------------------------------------

    event_file, event_source = get_event_source(
        subject,
        run
    )

    if event_file is None:

        print("ERROR: No event source found")

        errors.append({
            "subject": subject,
            "run": run,
            "v5_file": str(v5_file),
            "error": "NO_EVENT_SOURCE"
        })

        continue

    print(f"Event source: {event_source}")
    print(f"Event file:   {event_file}")

    # --------------------------------------------------------
    # READ EPOCHS
    # --------------------------------------------------------

    try:

        epochs = mne.read_epochs(
            v5_file,
            preload=True,
            verbose=False
        )

        n_epochs = len(epochs)

        print(f"Epochs: {n_epochs}")

        total_epochs += n_epochs

    except Exception as e:

        print(f"ERROR READING EPOCHS: {e}")

        errors.append({
            "subject": subject,
            "run": run,
            "v5_file": str(v5_file),
            "error": f"EPOCH_READ_ERROR: {e}"
        })

        continue

    # --------------------------------------------------------
    # READ EVENTS
    # --------------------------------------------------------

    try:

        if event_source == "BIDS":

            events_df = pd.read_csv(
                event_file,
                sep="\t"
            )

        else:

            events_df = pd.read_csv(
                event_file
            )

        print(f"Event rows: {len(events_df)}")

    except Exception as e:

        print(f"ERROR READING EVENTS: {e}")

        errors.append({
            "subject": subject,
            "run": run,
            "v5_file": str(v5_file),
            "event_file": str(event_file),
            "error": f"EVENT_READ_ERROR: {e}"
        })

        continue

    # --------------------------------------------------------
    # SAVE FINAL EPOCH FILE
    # --------------------------------------------------------

    output_file = (
        OUTPUT_EPOCHS
        / f"{subject}_run-{int(run):02d}_final_epo.fif"
    )

    try:

        epochs.save(
            output_file,
            overwrite=True,
            verbose=False
        )

        print(f"SAVED: {output_file}")

        total_saved += 1

    except Exception as e:

        print(f"ERROR SAVING: {e}")

        errors.append({
            "subject": subject,
            "run": run,
            "v5_file": str(v5_file),
            "error": f"SAVE_ERROR: {e}"
        })

        continue

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary.append({
        "subject": subject,
        "run": int(run),
        "v5_file": str(v5_file),
        "event_file": str(event_file),
        "event_source": event_source,
        "n_epochs": n_epochs,
        "n_event_rows": len(events_df),
        "output_file": str(output_file)
    })


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

summary_file = (
    OUTPUT_LOGS
    / "final_dataset_build_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# SAVE ERRORS
# ============================================================

errors_df = pd.DataFrame(errors)

error_file = (
    OUTPUT_LOGS
    / "final_dataset_build_errors.csv"
)

errors_df.to_csv(
    error_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 90)
print("FINAL DATASET BUILD COMPLETE")
print("=" * 90)

print(f"V5 input files:          {len(v5_files)}")
print(f"Runs successfully saved: {total_saved}")
print(f"Total input epochs:      {total_epochs}")
print(f"Errors:                  {len(errors)}")

print()
print("EVENT SOURCES:")

if len(summary_df) > 0:

    print(
        summary_df["event_source"]
        .value_counts()
        .to_string()
    )

print()
print("OUTPUT:")
print(OUTPUT_DIR)

print()
print("SUMMARY:")
print(summary_file)

print()
print("ERROR LOG:")
print(error_file)

print("=" * 90)