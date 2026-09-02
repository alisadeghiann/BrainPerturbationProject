import mne
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. FILE PATH
# ============================================================

fif_file = r"C:\Users\Ali\Desktop\BrainPerturbationProject\processed\sub-009_ses-01_task-WorkingMemory_run-1_raw.fif"

print("=" * 60)
print("LOADING PROCESSED EEG")
print("=" * 60)

# Load EEG
raw = mne.io.read_raw_fif(
    fif_file,
    preload=True
)

print(raw)

# ============================================================
# 2. BASIC EEG INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("BASIC EEG INFORMATION")
print("=" * 60)

print("Number of channels:", len(raw.ch_names))
print("Sampling rate:", raw.info["sfreq"])
print("Duration:", raw.times[-1])
print("Number of samples:", raw.n_times)

print("\nChannel names:")
print(raw.ch_names)

# ============================================================
# 3. CHANNEL TYPES
# ============================================================

print("\n" + "=" * 60)
print("CHANNEL TYPES")
print("=" * 60)

eeg_picks = mne.pick_types(
    raw.info,
    eeg=True,
    eog=False
)

eog_picks = mne.pick_types(
    raw.info,
    eeg=False,
    eog=True
)

print("EEG channels:", len(eeg_picks))
print("EOG channels:", len(eog_picks))

# ============================================================
# 4. DATA STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DATA STATISTICS")
print("=" * 60)

data = raw.get_data()

print("Data shape:", data.shape)
print("Minimum:", data.min())
print("Maximum:", data.max())
print("Mean:", data.mean())
print("STD:", data.std())

nan_count = np.isnan(data).sum()
inf_count = np.isinf(data).sum()

print("NaN count:", int(nan_count))
print("Inf count:", int(inf_count))

# ============================================================
# 5. EEG CHANNEL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("EEG CHANNELS")
print("=" * 60)

print("Number of EEG channels:", len(eeg_picks))

print("\nEEG channel names:")

for i in eeg_picks:
    print(i, raw.ch_names[i])

# ============================================================
# 6. EOG CHANNEL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("EOG CHANNELS")
print("=" * 60)

print("Number of EOG channels:", len(eog_picks))

for i in eog_picks:
    print(i, raw.ch_names[i])

# ============================================================
# 7. CHECK CHANNEL POSITIONS
# ============================================================

print("\n" + "=" * 60)
print("MONTAGE / CHANNEL POSITIONS")
print("=" * 60)

if raw.get_montage() is not None:
    print("Montage:", raw.get_montage())
    print("Montage successfully detected.")
else:
    print("WARNING: No montage detected.")

# ============================================================
# 8. CHECK BAD CHANNELS
# ============================================================

print("\n" + "=" * 60)
print("BAD CHANNELS")
print("=" * 60)

print("Bad channels:", raw.info["bads"])

if len(raw.info["bads"]) == 0:
    print("No bad channels currently marked.")

# ============================================================
# 9. AMPLITUDE CHECK
# ============================================================

print("\n" + "=" * 60)
print("AMPLITUDE CHECK")
print("=" * 60)

eeg_data = data[eeg_picks]

print("EEG-only minimum:", eeg_data.min())
print("EEG-only maximum:", eeg_data.max())
print("EEG-only mean:", eeg_data.mean())
print("EEG-only STD:", eeg_data.std())

# ============================================================
# 10. RAW EEG VIEWER
# ============================================================

print("\n" + "=" * 60)
print("OPENING EEG VIEWER")
print("=" * 60)

print("A graphical EEG window should open now.")
print("Close the EEG window after inspecting the signal.")

raw.plot(
    duration=20,
    n_channels=20,
    scalings="auto",
    title="Sub-009 Working Memory EEG - Raw Signal"
)

plt.show()

# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("QC SCRIPT FINISHED")
print("=" * 60)