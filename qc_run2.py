import mne
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")
FILE = PROJECT_DIR / "processed" / "sub-009_ses-01_task-WorkingMemory_run-2_clean-epo.fif"

print("=" * 60)
print("RUN 2 - FINAL EEG QUALITY CONTROL")
print("=" * 60)

epochs = mne.read_epochs(FILE, preload=True)

print("\nEpochs:", len(epochs))
print("Channels:", len(epochs.ch_names))
print("Sampling rate:", epochs.info["sfreq"])
print("Time range:", epochs.times[0], "to", epochs.times[-1])

print("\n" + "=" * 60)
print("CHANNEL INFORMATION")
print("=" * 60)

print("EEG:", len(mne.pick_types(epochs.info, eeg=True)))
print("EOG:", len(mne.pick_types(epochs.info, eog=True)))

print("\n" + "=" * 60)
print("DATA QUALITY")
print("=" * 60)

data = epochs.get_data()

print("NaN:", mne.utils._check_pandas_installed.__name__ if False else __import__("numpy").isnan(data).sum())
print("Inf:", __import__("numpy").isinf(data).sum())
print("Minimum:", data.min())
print("Maximum:", data.max())
print("Mean:", data.mean())
print("STD:", data.std())

print("\n" + "=" * 60)
print("EPOCH COUNTS")
print("=" * 60)

for name, code in epochs.event_id.items():
    print(f"{name:25s}: {(epochs.events[:, 2] == code).sum()}")

print("\n" + "=" * 60)
print("OPENING CLEAN EEG VIEWER")
print("=" * 60)

epochs["to_remember"].plot(
    n_epochs=5,
    n_channels=20,
    scalings="auto",
    title="Run 2 - Clean EEG - Remember"
)

print("\n[DONE]")
print("QC completed successfully.")