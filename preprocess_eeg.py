import os
import mne
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = r"C:\Users\Ali\Desktop\BrainPerturbationProject\processed\sub-009_ses-01_task-WorkingMemory_run-1_raw.fif"

OUTPUT_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject\processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD EEG
# ============================================================

print("=" * 70)
print("LOADING EEG")
print("=" * 70)

raw = mne.io.read_raw_fif(
    INPUT_FILE,
    preload=True
)

print(raw)


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print("Channels:", raw.info["nchan"])
print("Sampling rate:", raw.info["sfreq"])
print("Duration:", raw.times[-1])
print("Samples:", raw.n_times)


# ============================================================
# 3. CHANNEL TYPES
# ============================================================

print("\n" + "=" * 70)
print("CHANNEL TYPES")
print("=" * 70)

eeg_channels = mne.pick_types(
    raw.info,
    eeg=True,
    eog=False
)

eog_channels = mne.pick_types(
    raw.info,
    eeg=False,
    eog=True
)

print("EEG channels:", len(eeg_channels))
print("EOG channels:", len(eog_channels))


# ============================================================
# 4. AMPLITUDE CHECK
# ============================================================

print("\n" + "=" * 70)
print("AMPLITUDE CHECK")
print("=" * 70)

eeg_data = raw.get_data(
    picks=eeg_channels
)

print("EEG shape:", eeg_data.shape)
print("Minimum:", np.min(eeg_data))
print("Maximum:", np.max(eeg_data))
print("Mean:", np.mean(eeg_data))
print("STD:", np.std(eeg_data))

print(
    "NaN:",
    np.isnan(eeg_data).sum()
)

print(
    "Inf:",
    np.isinf(eeg_data).sum()
)


# ============================================================
# 5. CHECK EXTREME VALUES
# ============================================================

print("\n" + "=" * 70)
print("EXTREME VALUE CHECK")
print("=" * 70)

bad_value_count = np.sum(eeg_data == -1000)

print("Number of -1000 values:", bad_value_count)

percentage = (
    bad_value_count / eeg_data.size
) * 100

print(
    "-1000 percentage:",
    percentage,
    "%"
)


# ============================================================
# 6. FILTERING
# ============================================================

print("\n" + "=" * 70)
print("FILTERING EEG")
print("=" * 70)

print("Applying high-pass filter: 1 Hz")
print("Applying low-pass filter: 40 Hz")

raw_filtered = raw.copy()

raw_filtered.filter(
    l_freq=1.0,
    h_freq=40.0,
    picks="eeg",
    method="fir",
    phase="zero"
)

print("Filtering completed.")


# ============================================================
# 7. NOTCH FILTER
# ============================================================

print("\n" + "=" * 70)
print("NOTCH FILTER")
print("=" * 70)

print("Applying 50 Hz notch filter...")

raw_filtered.notch_filter(
    freqs=50,
    picks="eeg"
)

print("Notch filtering completed.")


# ============================================================
# 8. SAVE FILTERED EEG
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "sub-009_ses-01_task-WorkingMemory_run-1_filtered_raw.fif"
)

print("\n" + "=" * 70)
print("SAVING FILTERED EEG")
print("=" * 70)

print("Output:")
print(output_file)

raw_filtered.save(
    output_file,
    overwrite=True
)

print("\n[DONE]")
print("Filtered EEG saved successfully.")


# ============================================================
# 9. OPEN VIEWER
# ============================================================

print("\n" + "=" * 70)
print("OPENING FILTERED EEG VIEWER")
print("=" * 70)

raw_filtered.plot(
    n_channels=20,
    duration=10,
    scalings="auto",
    block=True
)