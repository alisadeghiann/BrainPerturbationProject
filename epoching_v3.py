import os
import glob
import numpy as np
import pandas as pd
import mne

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

INPUT_DIR = os.path.join(BASE_DIR, "preprocessed_v2")
OUTPUT_DIR = os.path.join(BASE_DIR, "epochs_v3")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TMIN = -0.2
TMAX = 0.8

files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "*_preprocessed_raw.fif"
        )
    )
)

print("=" * 78)
print("EPOCHING V3")
print("=" * 78)

print("Input files:", len(files))
print("Epoch window:", TMIN, "to", TMAX)

results = []

for n, fif_file in enumerate(files, 1):

    filename = os.path.basename(fif_file)

    print()
    print("=" * 78)
    print(f"[{n}/{len(files)}] {filename}")
    print("=" * 78)

    result = {
        "file": filename,
        "status": "FAILED",
        "events": 0,
        "valid_events": 0,
        "epochs": 0,
        "channels": 0,
        "sfreq": np.nan,
        "error": ""
    }

    try:

        raw = mne.io.read_raw_fif(
            fif_file,
            preload=True,
            verbose=False
        )

        result["channels"] = len(raw.ch_names)
        result["sfreq"] = raw.info["sfreq"]

        print("Channels:", len(raw.ch_names))
        print("Sampling rate:", raw.info["sfreq"])
        print("Samples:", raw.n_times)
        print("Annotations:", len(raw.annotations))

        if len(raw.annotations) == 0:

            print("ERROR: NO ANNOTATIONS")

            result["status"] = "ZERO_EVENTS"
            result["error"] = "No annotations"

            results.append(result)
            continue

        # ------------------------------------------------------
        # Decode annotation descriptions
        # ------------------------------------------------------

        descriptions = []

        for desc in raw.annotations.description:

            desc = str(desc)

            # Handle numpy-array-like ASCII representation
            if desc.startswith("[") and desc.endswith("]"):

                try:

                    nums = [
                        int(x)
                        for x in desc.strip(
                            "[]"
                        ).split()
                    ]

                    if all(
                        0 <= x <= 255
                        for x in nums
                    ):

                        desc = "".join(
                            chr(x)
                            for x in nums
                        )

                except Exception:
                    pass

            descriptions.append(desc)

        # Replace annotations with decoded descriptions
        raw.set_annotations(
            mne.Annotations(
                onset=raw.annotations.onset,
                duration=raw.annotations.duration,
                description=descriptions,
                orig_time=raw.annotations.orig_time
            )
        )

        print()
        print("EVENT TYPES")

        unique_types, counts = np.unique(
            descriptions,
            return_counts=True
        )

        for event_type, count in zip(
            unique_types,
            counts
        ):

            print(
                f"  {event_type:25s} {count}"
            )

        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------

        events, event_id = mne.events_from_annotations(
            raw,
            verbose=False
        )

        result["events"] = len(events)

        print()
        print("Events extracted:", len(events))

        if len(events) == 0:

            result["status"] = "ZERO_EVENTS"
            result["error"] = "No events extracted"

            results.append(result)
            continue

        # ------------------------------------------------------
        # Boundary check
        # ------------------------------------------------------

        sfreq = raw.info["sfreq"]

        start_offset = int(
            round(TMIN * sfreq)
        )

        end_offset = int(
            round(TMAX * sfreq)
        )

        valid_mask = (
            (events[:, 0] + start_offset >= 0)
            &
            (events[:, 0] + end_offset < raw.n_times)
        )

        valid_events = events[valid_mask]

        result["valid_events"] = len(valid_events)

        print(
            "Events entering epoching:",
            len(valid_events)
        )

        if len(valid_events) == 0:

            result["status"] = "NO_VALID_EVENTS"
            result["error"] = "No events within epoch boundaries"

            results.append(result)
            continue

        # ------------------------------------------------------
        # CREATE EPOCHS
        # ------------------------------------------------------

        print()
        print("Creating epochs...")

        epochs = mne.Epochs(
            raw,
            valid_events,
            event_id=event_id,
            tmin=TMIN,
            tmax=TMAX,
            baseline=None,
            preload=True,
            reject=None,
            flat=None,
            reject_by_annotation=False,
            detrend=None,
            event_repeated="merge",
            verbose=False
        )

        result["epochs"] = len(epochs)

        print(
            "Epochs retained:",
            len(epochs)
        )

        print(
            "Samples per epoch:",
            len(epochs.times)
        )

        # ------------------------------------------------------
        # SAVE
        # ------------------------------------------------------

        output_name = filename.replace(
            "_preprocessed_raw.fif",
            "_epo.fif"
        )

        output_file = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        print()
        print("Saving:")
        print(output_file)

        epochs.save(
            output_file,
            overwrite=True,
            verbose=False
        )

        # ------------------------------------------------------
        # VALIDATE
        # ------------------------------------------------------

        check = mne.read_epochs(
            output_file,
            preload=False,
            verbose=False
        )

        print()
        print("VALIDATION")
        print("-" * 78)
        print("Saved epochs:", len(check))
        print("Saved channels:", len(check.ch_names))
        print("Saved samples:", len(check.times))
        print("Saved sfreq:", check.info["sfreq"])

        if len(check) == 0:

            result["status"] = "ZERO_EPOCH"

        else:

            result["status"] = "SUCCESS"

        print()
        print("STATUS:", result["status"])

        del check
        del epochs
        del raw

    except Exception as e:

        result["status"] = "FAILED"
        result["error"] = str(e)

        print()
        print("ERROR:")
        print(str(e))

    results.append(result)


# ============================================================
# SAVE LOG
# ============================================================

df = pd.DataFrame(results)

csv_file = os.path.join(
    LOG_DIR,
    "epoching_v3_83runs_log.csv"
)

summary_file = os.path.join(
    LOG_DIR,
    "epoching_v3_83runs_summary.txt"
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
print("EPOCHING V3 COMPLETE")
print("=" * 78)

print()
print("TOTAL FILES:", len(files))

print()
print("STATUS COUNTS")
print(
    df["status"].value_counts()
)

print()
print("TOTAL EVENTS:")
print(
    int(df["events"].sum())
)

print()
print("TOTAL VALID EVENTS:")
print(
    int(df["valid_events"].sum())
)

print()
print("TOTAL EPOCHS:")
print(
    int(df["epochs"].sum())
)

print()
print("FAILED FILES")

failed = df[
    df["status"] == "FAILED"
]

if len(failed) == 0:

    print("NONE")

else:

    for _, row in failed.iterrows():

        print(
            row["file"],
            "|",
            row["error"]
        )

print()
print("ZERO-EPOCH FILES")

zero = df[
    df["status"] == "ZERO_EPOCH"
]

if len(zero) == 0:

    print("NONE")

else:

    for _, row in zero.iterrows():

        print(row["file"])

# ============================================================
# SUMMARY FILE
# ============================================================

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EPOCHING V3 SUMMARY\n"
    )

    f.write("=" * 78 + "\n\n")

    f.write(
        f"Total files: {len(files)}\n"
    )

    f.write(
        f"Total events: {int(df['events'].sum())}\n"
    )

    f.write(
        f"Total valid events: "
        f"{int(df['valid_events'].sum())}\n"
    )

    f.write(
        f"Total epochs: "
        f"{int(df['epochs'].sum())}\n"
    )

    f.write("\nSTATUS COUNTS\n")

    f.write(
        df["status"]
        .value_counts()
        .to_string()
    )

    f.write(
        "\n\nRAW DATA WAS NOT MODIFIED.\n"
    )

    f.write(
        "PREPROCESSED_V2 FILES WERE NOT MODIFIED.\n"
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
print("PREPROCESSED_V2 FILES WERE NOT MODIFIED.")