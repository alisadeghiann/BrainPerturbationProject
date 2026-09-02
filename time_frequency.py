import mne
import numpy as np
import os

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

epochs_file = os.path.join(
    BASE,
    "processed",
    "sub-009_ses-01_task-WorkingMemory_run-1_clean-epo.fif"
)

print("=" * 60)
print("LOADING CLEAN EPOCHS")
print("=" * 60)

epochs = mne.read_epochs(epochs_file, preload=True)

print(epochs)
print("Epochs:", len(epochs))
print("Channels:", len(epochs.ch_names))
print("Sampling rate:", epochs.info["sfreq"])

# فقط EEG
eeg = epochs.copy().pick("eeg")

# شرایط
remember = eeg["to_remember"]
ignore = eeg["to_ignore"]

print("\nConditions:")
print("to_remember:", len(remember))
print("to_ignore:", len(ignore))

print("\n" + "=" * 60)
print("COMPUTING PSD")
print("=" * 60)

# 4 تا 30 هرتز
# n_fft=256 چون epoch فقط 501 نمونه دارد
psd_r = remember.compute_psd(
    method="welch",
    fmin=4,
    fmax=30,
    n_fft=256,
    n_per_seg=256,
    n_overlap=128
)

psd_i = ignore.compute_psd(
    method="welch",
    fmin=4,
    fmax=30,
    n_fft=256,
    n_per_seg=256,
    n_overlap=128
)

power_r = psd_r.get_data()
power_i = psd_i.get_data()

freqs = psd_r.freqs

print("PSD shape remember:", power_r.shape)
print("PSD shape ignore:", power_i.shape)
print("Frequencies:", freqs[0], "to", freqs[-1], "Hz")

# باندهای فرکانسی
bands = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}

print("\n" + "=" * 60)
print("BAND POWER")
print("=" * 60)

results = {}

for name, (low, high) in bands.items():

    idx = (freqs >= low) & (freqs < high)

    remember_power = power_r[:, :, idx].mean(axis=2)
    ignore_power = power_i[:, :, idx].mean(axis=2)

    results[name] = {
        "remember": remember_power,
        "ignore": ignore_power
    }

    print("\n" + name.upper())
    print("Remember mean:", np.mean(remember_power))
    print("Ignore mean:", np.mean(ignore_power))
    print(
        "Difference:",
        np.mean(remember_power) - np.mean(ignore_power)
    )

print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

output = os.path.join(
    BASE,
    "processed",
    "sub-009_ses-01_task-WorkingMemory_run-1_psd.npz"
)

np.savez(
    output,
    freqs=freqs,
    theta_remember=results["theta"]["remember"],
    theta_ignore=results["theta"]["ignore"],
    alpha_remember=results["alpha"]["remember"],
    alpha_ignore=results["alpha"]["ignore"],
    beta_remember=results["beta"]["remember"],
    beta_ignore=results["beta"]["ignore"]
)

print("Saved:")
print(output)

print("\n[DONE]")
print("Time-frequency analysis completed successfully.")