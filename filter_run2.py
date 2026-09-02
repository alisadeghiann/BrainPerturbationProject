import mne
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

INPUT_FILE = PROJECT_DIR / (
    r"processed\sub-009_ses-01_task-WorkingMemory_run-2_raw.fif"
)

OUTPUT_FILE = PROJECT_DIR / (
    r"processed\sub-009_ses-01_task-WorkingMemory_run-2_filtered_raw.fif"
)

print("=" * 70)
print("LOADING RUN 2 EEG")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

raw = mne.io.read_raw_fif(
    INPUT_FILE,
    preload=True
)

print(raw)

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print("Channels:", raw.info["nchan"])
print("Sampling rate:", raw.info["sfreq"])
print("Duration:", raw.times[-1])
print("Samples:", raw.n_times)

print("\n" + "=" * 70)
print("FILTERING EEG")
print("=" * 70)

print("High-pass: 1 Hz")
print("Low-pass: 40 Hz")

raw.filter(
    l_freq=1.0,
    h_freq=40.0,
    picks="eeg",
    method="fir",
    phase="zero"
)

print("Filtering completed.")

print("\n" + "=" * 70)
print("NOTCH FILTER")
print("=" * 70)

print("Applying 50 Hz notch filter...")

raw.notch_filter(
    freqs=50.0,
    picks="eeg",
    method="fir",
    phase="zero"
)

print("Notch filtering completed.")

print("\n" + "=" * 70)
print("QUALITY CHECK")
print("=" * 70)

data = raw.get_data()

print("NaN:", np.isnan(data).sum())
print("Inf:", np.isinf(data).sum())
print("Minimum:", np.nanmin(data))
print("Maximum:", np.nanmax(data))
print("Mean:", np.nanmean(data))
print("STD:", np.nanstd(data))

print("\n" + "=" * 70)
print("SAVING FILTERED EEG")
print("=" * 70)

raw.save(
    OUTPUT_FILE,
    overwrite=True
)

print("\n[DONE]")
print("Filtered EEG saved successfully:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("OPENING FILTERED EEG VIEWER")
print("=" * 70)

raw.plot(
    n_channels=20,
    duration=20,
    scalings="auto",
    title="Subject 009 - Working Memory - Run 2 - Filtered"
)