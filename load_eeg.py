import h5py
import numpy as np
import mne
from pathlib import Path

# ============================================================
# CONFIGURATION - RUN 2
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

SET_FILE = PROJECT_DIR / (
    r"data\sub-009\ses-01\eeg"
    r"\sub-009_ses-01_task-WorkingMemory_run-2_eeg.set"
)

FDT_FILE = PROJECT_DIR / (
    r"data\sub-009\ses-01\eeg"
    r"\sub-009_ses-01_task-WorkingMemory_run-2_eeg.fdt"
)

N_CHANNELS = 71
SFREQ = 250.0

# ============================================================
# READ SAMPLE COUNT FROM FDT
# ============================================================

print("\n========== CHECKING RUN 2 FILES ==========")

if not SET_FILE.exists():
    raise FileNotFoundError(f"SET file not found:\n{SET_FILE}")

if not FDT_FILE.exists():
    raise FileNotFoundError(f"FDT file not found:\n{FDT_FILE}")

with h5py.File(SET_FILE, "r") as f:
    n_samples = int(np.array(f["pnts"]).squeeze())

print("SET file found.")
print("FDT file found.")
print("Number of samples:", n_samples)

# ============================================================
# HDF5 DECODING FUNCTIONS
# ============================================================

def decode_string(f, ref):
    obj = f[ref]
    values = np.array(obj).flatten()

    return "".join(
        chr(int(x))
        for x in values
    )


def decode_number(f, ref):
    obj = f[ref]
    values = np.array(obj).flatten()

    return float(values[0])


# ============================================================
# READ CHANNEL INFORMATION
# ============================================================

print("\n========== READING CHANNEL INFORMATION ==========")

with h5py.File(SET_FILE, "r") as f:

    chanlocs = f["chanlocs"]

    labels_dataset = chanlocs["labels"]
    X_dataset = chanlocs["X"]
    Y_dataset = chanlocs["Y"]
    Z_dataset = chanlocs["Z"]

    channel_names = []
    positions = {}

    for i in range(N_CHANNELS):

        label = decode_string(
            f,
            labels_dataset[i, 0]
        )

        x = decode_number(
            f,
            X_dataset[i, 0]
        )

        y = decode_number(
            f,
            Y_dataset[i, 0]
        )

        z = decode_number(
            f,
            Z_dataset[i, 0]
        )

        channel_names.append(label)

        positions[label] = np.array(
            [x, y, z],
            dtype=float
        )

print("Number of channels:", len(channel_names))

print("\nChannel names:")

for i, name in enumerate(channel_names):
    print(i, name)


# ============================================================
# LOAD EEG DATA
# ============================================================

print("\n========== LOADING EEG DATA ==========")

data = np.fromfile(
    FDT_FILE,
    dtype="<f4"
)

print("Raw values:", len(data))

expected_values = N_CHANNELS * n_samples

print("Expected values:", expected_values)

if len(data) != expected_values:

    raise ValueError(
        f"Unexpected FDT size.\n"
        f"Expected: {expected_values}\n"
        f"Found: {len(data)}"
    )

data = data.reshape(
    N_CHANNELS,
    n_samples,
    order="F"
)

print("EEG shape:", data.shape)


# ============================================================
# CREATE MNE INFO
# ============================================================

print("\n========== CREATING MNE RAW ==========")

ch_types = []

for name in channel_names:

    if name in ["LEYE", "REYE"]:
        ch_types.append("eog")
    else:
        ch_types.append("eeg")


info = mne.create_info(
    ch_names=channel_names,
    sfreq=SFREQ,
    ch_types=ch_types
)


# ============================================================
# CREATE RAW OBJECT
# ============================================================

print("\n========== CREATING MNE RAW ==========")

raw = mne.io.RawArray(
    data,
    info
)

print(raw)


# ============================================================
# SET ELECTRODE POSITIONS
# ============================================================

print("\n========== SETTING CHANNEL POSITIONS ==========")

montage = mne.channels.make_dig_montage(
    ch_pos=positions,
    coord_frame="head"
)

raw.set_montage(
    montage,
    on_missing="warn"
)

print("Montage successfully assigned.")


# ============================================================
# FINAL EEG INFORMATION
# ============================================================

print("\n========== FINAL EEG INFO ==========")

print("Channels:", raw.info["nchan"])

print(
    "EEG channels:",
    len(
        mne.pick_types(
            raw.info,
            eeg=True,
            exclude=[]
        )
    )
)

print(
    "EOG channels:",
    len(
        mne.pick_types(
            raw.info,
            eog=True,
            exclude=[]
        )
    )
)

print(
    "Sampling rate:",
    raw.info["sfreq"]
)

print(
    "Duration:",
    raw.times[-1],
    "seconds"
)

print(
    "Data shape:",
    raw.get_data().shape
)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n========== DATA QUALITY CHECK ==========")

eeg_data = raw.get_data()

print("NaN count:", np.isnan(eeg_data).sum())
print("Inf count:", np.isinf(eeg_data).sum())
print("Minimum:", np.nanmin(eeg_data))
print("Maximum:", np.nanmax(eeg_data))
print("Mean:", np.nanmean(eeg_data))
print("STD:", np.nanstd(eeg_data))


# ============================================================
# SAVE RAW FIF
# ============================================================

OUTPUT_DIR = PROJECT_DIR / "processed"

OUTPUT_DIR.mkdir(
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / (
    "sub-009_ses-01_"
    "task-WorkingMemory_"
    "run-2_raw.fif"
)

print("\n========== SAVING RAW EEG ==========")

raw.save(
    OUTPUT_FILE,
    overwrite=True
)

print("[DONE]")

print("\nSaved:")
print(OUTPUT_FILE)


# ============================================================
# PLOT EEG
# ============================================================

print("\n========== OPENING EEG VIEWER ==========")

raw.plot(
    n_channels=20,
    duration=20,
    scalings="auto",
    title="Subject 009 - Working Memory - Run 2"
)