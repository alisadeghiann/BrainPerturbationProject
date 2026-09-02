import os
import glob
import numpy as np
import pandas as pd
import mne

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

INPUT_DIR = os.path.join(BASE_DIR, "preprocessed_v2")
OUTPUT_DIR = os.path.join(BASE_DIR, "epochs_v2", "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = [
    "sub-004_ses-01_task-WorkingMemory_run-4_preprocessed_raw.fif",
    "sub-011_ses-01_task-WorkingMemory_run-2_preprocessed_raw.fif",
    "sub-012_ses-01_task-WorkingMemory_run-1_preprocessed_raw.fif",
]

rows = []

print("=" * 78)
print("DUPLICATE EVENT AUDIT")
print("=" * 78)

for filename in TARGETS:

    path = os.path.join(INPUT_DIR, filename)

    print()
    print("=" * 78)
    print(filename)
    print("=" * 78)

    if not os.path.exists(path):
        print("ERROR: FILE NOT FOUND")
        continue

    raw = mne.io.read_raw_fif(
        path,
        preload=False,
        verbose=False
    )

    print("Annotations:", len(raw.annotations))
    print("Sampling rate:", raw.info["sfreq"])
    print("Duration:", raw.times[-1])

    if len(raw.annotations) == 0:
        print("NO ANNOTATIONS")
        continue

    # ----------------------------------------------------------
    # Extract events
    # ----------------------------------------------------------

    events, event_id = mne.events_from_annotations(
        raw,
        verbose=False
    )

    print()
    print("Events extracted:", len(events))

    inverse_event_id = {
        value: key for key, value in event_id.items()
    }

    # ----------------------------------------------------------
    # Find duplicate samples
    # ----------------------------------------------------------

    samples = events[:, 0]

    unique_samples, counts = np.unique(
        samples,
        return_counts=True
    )

    duplicate_samples = unique_samples[counts > 1]

    print()
    print("Unique event samples:", len(unique_samples))
    print("Duplicate samples:", len(duplicate_samples))

    duplicate_event_count = int(
        np.sum(counts[counts > 1] - 1)
    )

    print(
        "Extra duplicate events:",
        duplicate_event_count
    )

    # ----------------------------------------------------------
    # Detailed duplicate information
    # ----------------------------------------------------------

    if len(duplicate_samples) == 0:

        print("NO DUPLICATES FOUND")

        continue

    print()
    print("-" * 78)
    print("DUPLICATE EVENT DETAILS")
    print("-" * 78)

    for sample in duplicate_samples:

        idx = np.where(samples == sample)[0]

        time_sec = sample / raw.info["sfreq"]

        print()
        print(
            f"Sample: {sample} | "
            f"Time: {time_sec:.6f} sec | "
            f"Count: {len(idx)}"
        )

        for i in idx:

            event_code = int(events[i, 2])
            event_name = inverse_event_id.get(
                event_code,
                f"UNKNOWN_{event_code}"
            )

            print(
                f"    event_index={i} | "
                f"code={event_code} | "
                f"type={event_name}"
            )

            rows.append({
                "file": filename,
                "sample": int(sample),
                "time_sec": float(time_sec),
                "event_index": int(i),
                "event_code": event_code,
                "event_type": event_name,
                "duplicate_count": len(idx)
            })

    # ----------------------------------------------------------
    # Event type counts
    # ----------------------------------------------------------

    print()
    print("-" * 78)
    print("EVENT TYPE COUNTS")
    print("-" * 78)

    for code, name in sorted(
        [(v, k) for k, v in event_id.items()]
    ):

        count = int(
            np.sum(events[:, 2] == code)
        )

        print(
            f"{name:25s} | code={code:3d} | count={count}"
        )

    # ----------------------------------------------------------
    # Duplicate type combinations
    # ----------------------------------------------------------

    print()
    print("-" * 78)
    print("DUPLICATE TYPE COMBINATIONS")
    print("-" * 78)

    combinations = {}

    for sample in duplicate_samples:

        idx = np.where(samples == sample)[0]

        types = tuple(
            sorted(
                inverse_event_id.get(
                    int(events[i, 2]),
                    f"UNKNOWN_{int(events[i, 2])}"
                )
                for i in idx
            )
        )

        combinations[types] = combinations.get(
            types,
            0
        ) + 1

    for combo, count in combinations.items():

        print(
            f"{count:4d} occurrence(s): {combo}"
        )

# --------------------------------------------------------------
# SAVE
# --------------------------------------------------------------

df = pd.DataFrame(rows)

output_csv = os.path.join(
    OUTPUT_DIR,
    "duplicate_event_audit.csv"
)

df.to_csv(
    output_csv,
    index=False
)

print()
print("=" * 78)
print("DUPLICATE EVENT AUDIT COMPLETE")
print("=" * 78)

print()
print("Total duplicate-event records:", len(df))

if len(df) > 0:

    print()
    print("By file:")

    print(
        df.groupby("file")
        .size()
        .to_string()
    )

print()
print("Saved:")
print(output_csv)

print()
print("IMPORTANT:")
print("NO RAW DATA WAS MODIFIED.")
print("NO PREPROCESSED FIF FILE WAS MODIFIED.")
print("NO EPOCH FILE WAS MODIFIED.")