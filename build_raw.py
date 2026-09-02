import numpy as np
import pandas as pd
import mne
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

EEG_DIR = PROJECT_DIR / "data" / "sub-002" / "ses-01" / "eeg"


# ============================================================
# SETTINGS
# ============================================================

SUBJECT = "sub-002"
TASK = "WorkingMemory"

SAMPLING_RATE = 250.0

all_raws = []


# ============================================================
# PROCESS RUNS 1–4
# ============================================================

for run_number in range(1, 5):

    print("\n" + "=" * 80)
    print(f"PROCESSING RUN {run_number}")
    print("=" * 80)

    prefix = (
        f"sub-002_ses-01_task-WorkingMemory_run-{run_number}"
    )

    fdt_file = EEG_DIR / f"{prefix}_eeg.fdt"
    channels_file = EEG_DIR / f"{prefix}_channels.tsv"
    electrodes_file = EEG_DIR / f"{prefix}_electrodes.tsv"
    events_file = EEG_DIR / f"{prefix}_events.tsv"

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    print("FDT exists:", fdt_file.exists())
    print("Channels exists:", channels_file.exists())
    print("Electrodes exists:", electrodes_file.exists())
    print("Events exists:", events_file.exists())

    required_files = [
        fdt_file,
        channels_file,
        electrodes_file,
        events_file
    ]

    missing_files = [
        str(file)
        for file in required_files
        if not file.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing files:\n" +
            "\n".join(missing_files)
        )

    # --------------------------------------------------------
    # READ CHANNEL INFORMATION
    # --------------------------------------------------------

    channels = pd.read_csv(
        channels_file,
        sep="\t"
    )

    print("\nChannels:", len(channels))

    print(
        "Channel types:",
        channels["type"].value_counts().to_dict()
    )

    # --------------------------------------------------------
    # READ ELECTRODE INFORMATION
    # --------------------------------------------------------

    electrodes = pd.read_csv(
        electrodes_file,
        sep="\t"
    )

    print(
        "Electrodes:",
        len(electrodes)
    )

    # --------------------------------------------------------
    # READ EEG DATA FROM FDT
    # --------------------------------------------------------

    print("\nReading FDT data...")

    data = np.fromfile(
        fdt_file,
        dtype=np.float32
    )

    n_channels = len(channels)

    if data.size % n_channels != 0:
        raise ValueError(
            f"Run {run_number}: "
            f"FDT data size ({data.size}) is not divisible "
            f"by number of channels ({n_channels})."
        )

    n_samples = data.size // n_channels

    # EEGLAB FDT data are channel × sample.
    data = data.reshape(
        (n_channels, n_samples),
        order="F"
    )

    print(
        "Data shape:",
        data.shape
    )

    # --------------------------------------------------------
    # CHANNEL NAMES
    # --------------------------------------------------------

    ch_names = (
        channels["name"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    # --------------------------------------------------------
    # CHANNEL TYPES
    # --------------------------------------------------------

    ch_types = []

    for channel_type in channels["type"]:

        channel_type = str(
            channel_type
        ).strip().upper()

        if channel_type == "EEG":
            ch_types.append("eeg")

        elif channel_type == "EOG":
            ch_types.append("eog")

        else:
            ch_types.append("misc")

    # --------------------------------------------------------
    # CREATE MNE INFO
    # --------------------------------------------------------

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=SAMPLING_RATE,
        ch_types=ch_types
    )

    # --------------------------------------------------------
    # CREATE RAW OBJECT
    # --------------------------------------------------------

    raw = mne.io.RawArray(
        data,
        info
    )

    # --------------------------------------------------------
    # ELECTRODE POSITIONS
    # --------------------------------------------------------

    montage_positions = {}

    for _, row in electrodes.iterrows():

        name = str(
            row["name"]
        ).strip()

        montage_positions[name] = np.array([
            float(row["x"]),
            float(row["y"]),
            float(row["z"])
        ])

    # --------------------------------------------------------
    # CREATE MONTAGE
    # --------------------------------------------------------

    montage = mne.channels.make_dig_montage(
        ch_pos=montage_positions,
        coord_frame="head"
    )

    # --------------------------------------------------------
    # APPLY MONTAGE
    #
    # EOG channels are intentionally allowed to have no
    # position because their positions are not provided in
    # electrodes.tsv.
    # --------------------------------------------------------

    raw.set_montage(
        montage,
        on_missing="warn"
    )

    # --------------------------------------------------------
    # READ EVENTS
    # --------------------------------------------------------

    events = pd.read_csv(
        events_file,
        sep="\t"
    )

    print(
        "Number of events:",
        len(events)
    )

    print(
        "Number of trials:",
        events["trial"].nunique()
    )

    # --------------------------------------------------------
    # BUILD EVENT DESCRIPTIONS
    # --------------------------------------------------------

    descriptions = []

    for _, event in events.iterrows():

        description = (
            f"{event['event_type']}__"
            f"{event['task_role']}__"
            f"cond-{event['memory_cond']}"
        )

        descriptions.append(
            description
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # EVENT ONSET = SECONDS
    # EVENT DURATION = MILLISECONDS
    #
    # MNE Annotations require both in SECONDS.
    #
    # Therefore duration is divided by 1000.
    # --------------------------------------------------------

    onset = (
        events["onset"]
        .astype(float)
        .to_numpy()
    )

    duration = (
        events["duration"]
        .astype(float)
        .to_numpy()
        / 1000.0
    )

    # --------------------------------------------------------
    # PROTECT AGAINST ANNOTATIONS EXTENDING BEYOND DATA
    # --------------------------------------------------------

    data_duration = raw.n_times / raw.info["sfreq"]

    annotation_end = onset + duration

    outside = annotation_end > data_duration

    if np.any(outside):

        print(
            f"\nWARNING: "
            f"{outside.sum()} annotations extend beyond "
            f"the end of Run {run_number}."
        )

        # Limit duration to the actual recording boundary.
        duration[outside] = np.maximum(
            0,
            data_duration - onset[outside]
        )

    # --------------------------------------------------------
    # CREATE MNE ANNOTATIONS
    # --------------------------------------------------------

    annotations = mne.Annotations(
        onset=onset,
        duration=duration,
        description=descriptions
    )

    raw.set_annotations(
        annotations
    )

    # --------------------------------------------------------
    # ADD RUN DESCRIPTION
    # --------------------------------------------------------

    raw.info["description"] = (
        f"{SUBJECT} {TASK} run-{run_number}"
    )

    # --------------------------------------------------------
    # STORE RUN
    # --------------------------------------------------------

    all_raws.append(
        raw
    )

    print(
        f"\nRun {run_number} successfully converted to MNE Raw."
    )


# ============================================================
# ALL RUNS SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ALL RUNS LOADED")
print("=" * 80)

for i, raw in enumerate(
    all_raws,
    start=1
):

    print(
        f"Run {i}: "
        f"{len(raw.ch_names)} channels | "
        f"{raw.n_times} samples | "
        f"{raw.times[-1]:.2f} sec | "
        f"{raw.info['sfreq']} Hz"
    )


# ============================================================
# COMBINE RUNS
# ============================================================

print("\n" + "=" * 80)
print("COMBINING RUNS")
print("=" * 80)

combined_raw = mne.concatenate_raws(
    all_raws,
    preload=True
)

print("\nCombined EEG:")
print(combined_raw)

print(
    "\nChannels:",
    len(combined_raw.ch_names)
)

print(
    "Total samples:",
    combined_raw.n_times
)

print(
    "Sampling rate:",
    combined_raw.info["sfreq"]
)

print(
    "Total duration:",
    combined_raw.times[-1],
    "seconds"
)


# ============================================================
# ANNOTATION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ANNOTATION SUMMARY")
print("=" * 80)

print(
    "Total annotations:",
    len(combined_raw.annotations)
)

# Show first few annotations
print("\nFirst 10 annotations:")

for onset, duration, description in zip(
    combined_raw.annotations.onset[:10],
    combined_raw.annotations.duration[:10],
    combined_raw.annotations.description[:10]
):

    print(
        f"{onset:10.3f}s | "
        f"{duration:8.3f}s | "
        f"{description}"
    )


# ============================================================
# MEMORY CONDITION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MEMORY CONDITION ANNOTATION COUNTS")
print("=" * 80)

descriptions = combined_raw.annotations.description

for condition in [
    "cond-3",
    "cond-5",
    "cond-7"
]:

    count = sum(
        condition in description
        for description in descriptions
    )

    print(
        f"{condition}: {count}"
    )


# ============================================================
# SAVE FINAL MNE FILE
# ============================================================

output_file = (
    PROJECT_DIR /
    "sub-002_working_memory_combined_raw.fif"
)

print("\n" + "=" * 80)
print("SAVING MNE FILE")
print("=" * 80)

print(output_file)

combined_raw.save(
    output_file,
    overwrite=True
)

print("\n" + "=" * 80)
print("MNE FILE SAVED SUCCESSFULLY")
print("=" * 80)

print(
    output_file
)

print("\n" + "=" * 80)
print("BUILD RAW COMPLETED SUCCESSFULLY")
print("=" * 80)