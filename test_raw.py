import numpy as np
import mne

# =========================
# FILE PATH
# =========================

fdt_path = r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-009\ses-01\eeg\sub-009_ses-01_task-WorkingMemory_run-1_eeg.fdt"

# =========================
# EEG PARAMETERS
# =========================

n_channels = 71
n_samples = 153971
sfreq = 250.0

# =========================
# LOAD FDT
# =========================

data = np.fromfile(fdt_path, dtype="<f4")

print("Total values:", len(data))

# Verify expected size
expected = n_channels * n_samples

if len(data) != expected:
    raise ValueError(
        f"Unexpected data size: {len(data)} "
        f"(expected {expected})"
    )

# EEGLAB/MATLAB ordering
data = data.reshape(
    n_channels,
    n_samples,
    order="F"
)

print("EEG shape:", data.shape)

# =========================
# CREATE CHANNEL NAMES
# =========================

ch_names = [f"EEG{i:03d}" for i in range(1, n_channels + 1)]

# EEG channel types
ch_types = ["eeg"] * n_channels

# =========================
# CREATE MNE INFO
# =========================

info = mne.create_info(
    ch_names=ch_names,
    sfreq=sfreq,
    ch_types=ch_types
)

# =========================
# CREATE RAW OBJECT
# =========================

raw = mne.io.RawArray(
    data,
    info
)

print(raw)

print("\n========== EEG INFO ==========")
print("Channels:", raw.info["nchan"])
print("Sampling rate:", raw.info["sfreq"])
print("Duration:", raw.times[-1], "seconds")
print("Data shape:", raw.get_data().shape)

# =========================
# BASIC STATISTICS
# =========================

print("\n========== DATA STATISTICS ==========")

print("Min:", data.min())
print("Max:", data.max())
print("Mean:", data.mean())
print("STD:", data.std())
# =========================
# CHECK EXTREME VALUES
# =========================

n_minus_1000 = np.sum(data == -1000.0)
n_nan = np.sum(np.isnan(data))
n_inf = np.sum(np.isinf(data))

print("\n========== QUALITY CHECK ==========")
print("Number of -1000 values:", n_minus_1000)
print("Number of NaN values:", n_nan)
print("Number of Inf values:", n_inf)

print(
    "\n-1000 percentage:",
    100 * n_minus_1000 / data.size,
    "%"
)
# =========================
# PLOT EEG
# =========================

raw.plot(
    n_channels=20,
    duration=20,
    scalings="auto",
    title="Subject 009 - Working Memory - Run 1"
)
