import mne
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")
PROCESSED = PROJECT_DIR / "processed"

INPUT = PROCESSED / "sub-009_ses-01_task-WorkingMemory_run-2_clean-epo.fif"
OUTPUT = PROCESSED / "sub-009_ses-01_task-WorkingMemory_run-2_psd.npz"

print("=" * 60)
print("LOADING RUN 2 CLEAN EEG")
print("=" * 60)

epochs = mne.read_epochs(INPUT, preload=True)

remember = epochs["to_remember"]
ignore = epochs["to_ignore"]

print("Remember epochs:", len(remember))
print("Ignore epochs:", len(ignore))

eeg_picks = mne.pick_types(
    epochs.info,
    eeg=True,
    exclude="bads"
)

print("EEG channels:", len(eeg_picks))

print("\n" + "=" * 60)
print("COMPUTING PSD")
print("=" * 60)

psd_remember = remember.compute_psd(
    method="welch",
    fmin=4,
    fmax=30,
    picks=eeg_picks,
    n_fft=500,
    n_per_seg=250,
    verbose=True
)

psd_ignore = ignore.compute_psd(
    method="welch",
    fmin=4,
    fmax=30,
    picks=eeg_picks,
    n_fft=500,
    n_per_seg=250,
    verbose=True
)

power_remember = psd_remember.get_data()
power_ignore = psd_ignore.get_data()

freqs = psd_remember.freqs

print("PSD Remember:", power_remember.shape)
print("PSD Ignore:", power_ignore.shape)

bands = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}

results = {}

print("\n" + "=" * 60)
print("BAND POWER")
print("=" * 60)

for name, (low, high) in bands.items():

    mask = (freqs >= low) & (freqs < high)

    remember_power = power_remember[:, :, mask].mean(axis=2)
    ignore_power = power_ignore[:, :, mask].mean(axis=2)

    remember_mean = remember_power.mean()
    ignore_mean = ignore_power.mean()

    difference = remember_mean - ignore_mean

    results[f"{name}_remember"] = remember_power
    results[f"{name}_ignore"] = ignore_power

    print("\n" + name.upper())
    print("Remember mean:", remember_mean)
    print("Ignore mean:", ignore_mean)
    print("Difference:", difference)

np.savez(
    OUTPUT,
    freqs=freqs,
    ch_names=np.array(
        [epochs.ch_names[i] for i in eeg_picks]
    ),
    **results
)

print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

print("Saved:")
print(OUTPUT)

print("\n[DONE]")
print("Run 2 time-frequency analysis completed.")