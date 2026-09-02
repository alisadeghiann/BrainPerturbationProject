import os
import glob
import pandas as pd
import numpy as np
import h5py

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

EVENT_FILE = os.path.join(
    BASE,
    "qc",
    "events",
    "ALL_EVENTS_83_RUNS.csv"
)

OUT_DIR = os.path.join(
    BASE,
    "qc",
    "subject_inspection"
)

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 90)
print("SUBJECT + EEG QUALITY INSPECTION")
print("=" * 90)

events = pd.read_csv(EVENT_FILE)

print("Events loaded:", len(events))

# ---------------------------------------------------------
# SUBJECT EVENT SUMMARY
# ---------------------------------------------------------

subject_events = (
    events.groupby("subject")
    .agg(
        events=("type", "size"),
        event_types=("type", "nunique"),
        trials=("trial", "nunique")
    )
    .reset_index()
)

# ---------------------------------------------------------
# FIND EEG FILES
# ---------------------------------------------------------

set_files = glob.glob(
    os.path.join(
        BASE,
        "data",
        "**",
        "*_eeg.set"
    ),
    recursive=True
)

print("EEG files:", len(set_files))

records = []

# ---------------------------------------------------------
# PROCESS EEG FILES
# ---------------------------------------------------------

for i, set_file in enumerate(sorted(set_files), 1):

    name = os.path.basename(set_file)

    print("\n" + "=" * 80)
    print(f"{i}/{len(set_files)}")
    print(name)

    try:

        with h5py.File(set_file, "r") as f:

            # -----------------------------
            # EEG metadata
            # -----------------------------

            nbchan = float(
                np.array(f["nbchan"]).squeeze()
            )

            pnts = float(
                np.array(f["pnts"]).squeeze()
            )

            srate = float(
                np.array(f["srate"]).squeeze()
            )

            trials = float(
                np.array(f["trials"]).squeeze()
            )

            # -----------------------------
            # FDT
            # -----------------------------

            fdt_file = set_file.replace(
                "_eeg.set",
                "_eeg.fdt"
            )

            if not os.path.exists(fdt_file):

                print("FDT MISSING")

                records.append({
                    "file": name,
                    "subject": name.split("_")[0],
                    "channels": nbchan,
                    "samples": pnts,
                    "sampling_rate": srate,
                    "trials_eeglab": trials,
                    "fdt_exists": False,
                    "data_std": np.nan,
                    "data_min": np.nan,
                    "data_max": np.nan
                })

                continue

            # -----------------------------
            # Read FDT
            # -----------------------------

            expected = int(nbchan * pnts)

            raw = np.fromfile(
                fdt_file,
                dtype=np.float32
            )

            if len(raw) != expected:

                print(
                    "SIZE MISMATCH:",
                    len(raw),
                    "expected:",
                    expected
                )

                records.append({
                    "file": name,
                    "subject": name.split("_")[0],
                    "channels": nbchan,
                    "samples": pnts,
                    "sampling_rate": srate,
                    "trials_eeglab": trials,
                    "fdt_exists": True,
                    "data_std": np.nan,
                    "data_min": np.nan,
                    "data_max": np.nan
                })

                continue

            data = raw.reshape(
                int(nbchan),
                int(pnts),
                order="C"
            )

            # -----------------------------
            # Signal statistics
            # -----------------------------

            data_min = float(np.min(data))
            data_max = float(np.max(data))
            data_std = float(np.std(data))

            # channel std
            channel_std = np.std(
                data,
                axis=1
            )

            median_std = float(
                np.median(channel_std)
            )

            max_std = float(
                np.max(channel_std)
            )

            # flat channels
            flat_channels = int(
                np.sum(channel_std < 1e-6)
            )

            # extreme channels
            if median_std > 0:

                channels_5x = int(
                    np.sum(
                        channel_std >
                        median_std * 5
                    )
                )

                channels_10x = int(
                    np.sum(
                        channel_std >
                        median_std * 10
                    )
                )

            else:

                channels_5x = 0
                channels_10x = 0

            print("Channels:", nbchan)
            print("Samples:", pnts)
            print("Sampling:", srate)
            print("STD:", data_std)
            print("Median channel STD:", median_std)
            print("Max channel STD:", max_std)
            print("Flat channels:", flat_channels)
            print(">5x channels:", channels_5x)
            print(">10x channels:", channels_10x)

            records.append({
                "file": name,
                "subject": name.split("_")[0],
                "channels": nbchan,
                "samples": pnts,
                "sampling_rate": srate,
                "trials_eeglab": trials,
                "fdt_exists": True,
                "data_std": data_std,
                "data_min": data_min,
                "data_max": data_max,
                "median_channel_std": median_std,
                "max_channel_std": max_std,
                "flat_channels": flat_channels,
                "channels_over_5x": channels_5x,
                "channels_over_10x": channels_10x
            })

    except Exception as e:

        print("ERROR:", e)

        records.append({
            "file": name,
            "subject": name.split("_")[0],
            "error": str(e)
        })

# ---------------------------------------------------------
# CREATE EEG TABLE
# ---------------------------------------------------------

eeg = pd.DataFrame(records)

# ---------------------------------------------------------
# MERGE WITH EVENT INFORMATION
# ---------------------------------------------------------

subject_summary = subject_events.merge(
    eeg.groupby("subject").agg(
        eeg_files=("file", "size"),
        mean_signal_std=("data_std", "mean"),
        min_signal_std=("data_std", "min"),
        max_signal_std=("data_std", "max"),
        max_flat_channels=("flat_channels", "max"),
        max_channels_over_5x=("channels_over_5x", "max"),
        max_channels_over_10x=("channels_over_10x", "max")
    ).reset_index(),
    on="subject",
    how="outer"
)

# ---------------------------------------------------------
# PRINT SUBJECT SUMMARY
# ---------------------------------------------------------

print("\n")
print("=" * 90)
print("FINAL SUBJECT SUMMARY")
print("=" * 90)

print(
    subject_summary
    .sort_values("subject")
    .to_string(index=False)
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

eeg.to_csv(
    os.path.join(
        OUT_DIR,
        "EEG_FILE_SIGNAL_SUMMARY.csv"
    ),
    index=False
)

subject_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "SUBJECT_EEG_EVENT_SUMMARY.csv"
    ),
    index=False
)

# ---------------------------------------------------------
# FLAG POTENTIAL PROBLEMS
# ---------------------------------------------------------

print("\n")
print("=" * 90)
print("POTENTIAL SIGNAL PROBLEMS")
print("=" * 90)

problem_files = eeg[
    (eeg["flat_channels"].fillna(0) > 0) |
    (eeg["channels_over_10x"].fillna(0) > 0)
]

print(
    problem_files[
        [
            "subject",
            "file",
            "sampling_rate",
            "data_std",
            "flat_channels",
            "channels_over_5x",
            "channels_over_10x"
        ]
    ].to_string(index=False)
)

problem_files.to_csv(
    os.path.join(
        OUT_DIR,
        "POTENTIAL_SIGNAL_PROBLEMS.csv"
    ),
    index=False
)

print("\n")
print("=" * 90)
print("COMPLETE")
print("=" * 90)

print("\nOutput directory:")
print(OUT_DIR)

print("\nFiles:")
for f in sorted(os.listdir(OUT_DIR)):
    print(" ", f)

print("\nIMPORTANT:")
print("No EEG data modified.")
print("No trials deleted.")
print("No subjects deleted.")