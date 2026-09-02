import os
import glob
import warnings
import numpy as np
import pandas as pd
import mne

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

INPUT_DIR = os.path.join(BASE_DIR, "preprocessed_v2")
OUTPUT_DIR = os.path.join(BASE_DIR, "epochs_v2")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# EPOCH PARAMETERS
# ============================================================

# Epoch from event onset
TMIN = -0.2
TMAX = 0.8

# IMPORTANT:
# No amplitude rejection at this stage.
# We want to diagnose the data before removing epochs.
REJECT = None
FLAT = None

# ============================================================
# EVENT DECODING
# ============================================================

def get_events_from_annotations(raw):

    annotations = raw.annotations

    if len(annotations) == 0:
        return np.empty((0, 3), dtype=int), {}

    event_map = {}
    next_id = 1

    for desc in annotations.description:

        desc = str(desc)

        if desc not in event_map:
            event_map[desc] = next_id
            next_id += 1

    events, _ = mne.events_from_annotations(
        raw,
        event_id=event_map,
        verbose=False
    )

    return events, event_map


# ============================================================
# FILE DISCOVERY
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "*_preprocessed_raw.fif"
        )
    )
)

print("=" * 78)
print("EPOCHING V2 - ANNOTATION BASED")
print("=" * 78)

print()
print("Input directory:")
print(INPUT_DIR)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Files found:", len(files))

if len(files) == 0:
    raise RuntimeError("No preprocessed FIF files found.")

# ============================================================
# PROCESSING
# ============================================================

results = []

for idx, fif_file in enumerate(files, start=1):

    filename = os.path.basename(fif_file)

    print()
    print("=" * 78)
    print(f"[{idx}/{len(files)}] {filename}")
    print("=" * 78)

    row = {
        "file": filename,
        "status": "FAILED",
        "events": 0,
        "epochs_before": 0,
        "epochs_retained": 0,
        "epochs_dropped": 0,
        "channels": 0,
        "samples_per_epoch": 0,
        "sfreq": np.nan,
        "tmin": TMIN,
        "tmax": TMAX,
        "event_types": "",
        "error": ""
    }

    try:

        # ----------------------------------------------------
        # LOAD RAW
        # ----------------------------------------------------

        print("Loading:")
        print(fif_file)

        raw = mne.io.read_raw_fif(
            fif_file,
            preload=True,
            verbose=False
        )

        row["channels"] = len(raw.ch_names)
        row["sfreq"] = raw.info["sfreq"]

        print()
        print("RAW")
        print("-" * 78)
        print("Channels:", len(raw.ch_names))
        print("Sampling rate:", raw.info["sfreq"])
        print("Samples:", raw.n_times)
        print("Duration:", raw.times[-1])

        # ----------------------------------------------------
        # ANNOTATIONS
        # ----------------------------------------------------

        print()
        print("ANNOTATIONS")
        print("-" * 78)

        print("Number of annotations:", len(raw.annotations))

        if len(raw.annotations) == 0:

            print("WARNING: ZERO ANNOTATIONS")

            row["status"] = "ZERO_EVENTS"
            row["error"] = "No annotations found"

            results.append(row)
            continue

        # Show event types
        unique_desc, counts = np.unique(
            raw.annotations.description,
            return_counts=True
        )

        event_summary = []

        for desc, count in zip(unique_desc, counts):

            print(f"{desc}: {count}")

            event_summary.append(
                f"{desc}:{count}"
            )

        row["event_types"] = ";".join(event_summary)

        # ----------------------------------------------------
        # EVENTS
        # ----------------------------------------------------

        print()
        print("EVENT EXTRACTION")
        print("-" * 78)

        events, event_id = get_events_from_annotations(raw)

        print("Events extracted:", len(events))

        row["events"] = len(events)

        if len(events) == 0:

            print("WARNING: ZERO EVENTS")

            row["status"] = "ZERO_EVENTS"
            row["error"] = "events_from_annotations returned zero events"

            results.append(row)
            continue

        print()
        print("Event ID:")
        for name, value in event_id.items():
            print(f"  {value} = {name}")

        # ----------------------------------------------------
        # EPOCH BOUNDARY CHECK
        # ----------------------------------------------------

        sfreq = raw.info["sfreq"]

        start_offset = int(round(TMIN * sfreq))
        end_offset = int(round(TMAX * sfreq))

        valid_events = []

        for event in events:

            sample = int(event[0])

            if (
                sample + start_offset >= 0
                and
                sample + end_offset < raw.n_times
            ):
                valid_events.append(event)

        valid_events = np.asarray(
            valid_events,
            dtype=int
        )

        print()
        print("EPOCH BOUNDARY CHECK")
        print("-" * 78)

        print("Events before boundary check:", len(events))
        print("Events entering epoching:", len(valid_events))
        print(
            "Events removed at edges:",
            len(events) - len(valid_events)
        )

        if len(valid_events) == 0:

            row["status"] = "NO_VALID_EVENTS"
            row["error"] = "All events outside epoch boundaries"

            results.append(row)
            continue

        # ----------------------------------------------------
        # CREATE EPOCHS
        # ----------------------------------------------------

        print()
        print("CREATING EPOCHS")
        print("-" * 78)

        epochs = mne.Epochs(
            raw,
            valid_events,
            event_id=event_id,
            tmin=TMIN,
            tmax=TMAX,
            baseline=None,
            preload=True,
            reject=REJECT,
            flat=FLAT,
            reject_by_annotation=False,
            detrend=None,
            verbose=False
        )

        row["epochs_before"] = len(valid_events)
        row["epochs_retained"] = len(epochs)
        row["epochs_dropped"] = (
            len(valid_events) - len(epochs)
        )

        row["samples_per_epoch"] = len(epochs.times)

        print()
        print("EPOCH RESULTS")
        print("-" * 78)

        print("Epochs before rejection:", len(valid_events))
        print("Epochs retained:", len(epochs))
        print(
            "Epochs rejected:",
            len(valid_events) - len(epochs)
        )

        print(
            "Samples per epoch:",
            len(epochs.times)
        )

        print(
            "Epoch duration:",
            epochs.times[-1] - epochs.times[0]
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        out_name = filename.replace(
            "_preprocessed_raw.fif",
            "_epo.fif"
        )

        out_file = os.path.join(
            OUTPUT_DIR,
            out_name
        )

        print()
        print("Saving:")
        print(out_file)

        epochs.save(
            out_file,
            overwrite=True,
            verbose=False
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print()
        print("VALIDATION")
        print("-" * 78)

        check = mne.read_epochs(
            out_file,
            preload=False,
            verbose=False
        )

        print("Saved epochs:", len(check))
        print("Saved channels:", len(check.ch_names))
        print("Saved samples/epoch:", len(check.times))
        print("Saved sfreq:", check.info["sfreq"])

        if len(check) == 0:

            row["status"] = "ZERO_EPOCH"
            row["error"] = "Saved Epochs object contains zero epochs"

        else:

            row["status"] = "SUCCESS"

        print()
        print("STATUS:", row["status"])

        del check
        del epochs
        del raw

    except Exception as e:

        row["status"] = "FAILED"
        row["error"] = str(e)

        print()
        print("ERROR:")
        print(str(e))

    results.append(row)

# ============================================================
# SAVE LOG
# ============================================================

df = pd.DataFrame(results)

csv_file = os.path.join(
    LOG_DIR,
    "epoching_v2_83runs_log.csv"
)

summary_file = os.path.join(
    LOG_DIR,
    "epoching_v2_83runs_summary.txt"
)

df.to_csv(
    csv_file,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 78)
print("EPOCHING V2 COMPLETE")
print("=" * 78)

print()
print("TOTAL FILES:", len(files))

print()
print("STATUS COUNTS")
print(df["status"].value_counts())

print()
print("TOTAL EVENTS:", df["events"].sum())

print(
    "TOTAL EPOCHS BEFORE:",
    df["epochs_before"].sum()
)

print(
    "TOTAL EPOCHS RETAINED:",
    df["epochs_retained"].sum()
)

print(
    "TOTAL EPOCHS DROPPED:",
    df["epochs_dropped"].sum()
)

print()
print("Files with ZERO epochs:")

zero_df = df[
    df["status"].isin(
        ["ZERO_EPOCH", "ZERO_EVENTS", "NO_VALID_EVENTS"]
    )
]

if len(zero_df) == 0:

    print("NONE")

else:

    for _, r in zero_df.iterrows():

        print(
            f"{r['file']} | "
            f"status={r['status']} | "
            f"events={r['events']} | "
            f"epochs={r['epochs_retained']}"
        )

print()
print("FAILED FILES:")

failed_df = df[
    df["status"] == "FAILED"
]

if len(failed_df) == 0:

    print("NONE")

else:

    for _, r in failed_df.iterrows():

        print(
            f"{r['file']} | {r['error']}"
        )

# ============================================================
# TEXT SUMMARY
# ============================================================

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EPOCHING V2 SUMMARY\n"
    )

    f.write("=" * 78 + "\n\n")

    f.write(
        f"Total input files: {len(files)}\n"
    )

    f.write(
        f"Total events: {df['events'].sum()}\n"
    )

    f.write(
        f"Total epochs before: "
        f"{df['epochs_before'].sum()}\n"
    )

    f.write(
        f"Total epochs retained: "
        f"{df['epochs_retained'].sum()}\n"
    )

    f.write(
        f"Total epochs dropped: "
        f"{df['epochs_dropped'].sum()}\n"
    )

    f.write("\nSTATUS COUNTS\n")
    f.write(
        df["status"]
        .value_counts()
        .to_string()
    )

    f.write("\n\n")

    f.write(
        "RAW DATA WAS NOT MODIFIED.\n"
    )

    f.write(
        "PREPROCESSED FIF FILES WERE NOT MODIFIED.\n"
    )

print()
print("=" * 78)
print("COMPLETE")
print("=" * 78)

print()
print("Saved:")
print(csv_file)
print(summary_file)

print()
print("RAW DATA WAS NOT MODIFIED.")
print("ORIGINAL SET/FDT FILES WERE NOT MODIFIED.")
print("PREPROCESSED_V2 FIF FILES WERE NOT MODIFIED.")