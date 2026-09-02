import os
import numpy as np
import mne

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

EPOCH_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "sub-009_ses-01_task-WorkingMemory_run-1_clean-epo.fif"
)

# ============================================================
# LOAD CLEAN EPOCHS
# ============================================================

print("=" * 70)
print("LOADING CLEAN EPOCHS")
print("=" * 70)

epochs = mne.read_epochs(
    EPOCH_FILE,
    preload=True,
    verbose=True
)

print(epochs)

# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print("Number of epochs:", len(epochs))
print("Number of channels:", len(epochs.ch_names))
print("Sampling rate:", epochs.info["sfreq"])
print("Time range:", epochs.times[0], "to", epochs.times[-1])
print("Data shape:", epochs.get_data().shape)

# ============================================================
# CHANNEL TYPES
# ============================================================

print("\n" + "=" * 70)
print("CHANNEL TYPES")
print("=" * 70)

eeg_picks = mne.pick_types(
    epochs.info,
    eeg=True,
    eog=False
)

eog_picks = mne.pick_types(
    epochs.info,
    eeg=False,
    eog=True
)

print("EEG channels:", len(eeg_picks))
print("EOG channels:", len(eog_picks))

print("EOG channels:")
for idx in eog_picks:
    print(" ", epochs.ch_names[idx])

# ============================================================
# DATA QUALITY
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)

data = epochs.get_data()

print("NaN count:", np.isnan(data).sum())
print("Inf count:", np.isinf(data).sum())

print("Minimum:", np.min(data))
print("Maximum:", np.max(data))
print("Mean:", np.mean(data))
print("STD:", np.std(data))

# ============================================================
# EPOCH EVENT COUNTS
# ============================================================

print("\n" + "=" * 70)
print("EPOCH COUNTS")
print("=" * 70)

for event_name, event_code in epochs.event_id.items():

    count = np.sum(
        epochs.events[:, 2] == event_code
    )

    print(f"{event_name:25s}: {count}")

# ============================================================
# DROP BAD EPOCHS CHECK
# ============================================================

print("\n" + "=" * 70)
print("BAD EPOCH INFORMATION")
print("=" * 70)

print("Number of dropped epochs:", len(epochs.drop_log))

dropped = sum(
    len(reason) > 0
    for reason in epochs.drop_log
)

print("Actually dropped epochs:", dropped)

# ============================================================
# PLOT A SAMPLE OF CLEAN EEG
# ============================================================

print("\n" + "=" * 70)
print("OPENING CLEAN EEG VIEWER")
print("=" * 70)

epochs.plot(
    n_epochs=10,
    n_channels=20,
    scalings="auto",
    block=True
)

print("\n" + "=" * 70)
print("QC COMPLETED")
print("=" * 70)