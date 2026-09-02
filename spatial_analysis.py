import mne
import numpy as np
import os
import matplotlib.pyplot as plt

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

epochs_file = os.path.join(
    BASE,
    "processed",
    "sub-009_ses-01_task-WorkingMemory_run-1_clean-epo.fif"
)

print("=" * 70)
print("LOADING CLEAN EEG")
print("=" * 70)

epochs = mne.read_epochs(epochs_file, preload=True)

eeg = epochs.copy().pick("eeg")

remember = eeg["to_remember"]
ignore = eeg["to_ignore"]

print("Remember epochs:", len(remember))
print("Ignore epochs:", len(ignore))
print("EEG channels:", len(eeg.ch_names))

# ---------------------------------------------------------
# PSD
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("COMPUTING PSD")
print("=" * 70)

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

print("PSD Remember:", power_r.shape)
print("PSD Ignore:", power_i.shape)

# ---------------------------------------------------------
# Frequency bands
# ---------------------------------------------------------

bands = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}

results = {}

print("\n" + "=" * 70)
print("COMPUTING CHANNEL-WISE BAND POWER")
print("=" * 70)

for band, (low, high) in bands.items():

    idx = (freqs >= low) & (freqs < high)

    r = power_r[:, :, idx].mean(axis=2)
    i = power_i[:, :, idx].mean(axis=2)

    r_mean = r.mean(axis=0)
    i_mean = i.mean(axis=0)

    difference = r_mean - i_mean

    results[band] = {
        "remember": r_mean,
        "ignore": i_mean,
        "difference": difference
    }

    print("\n" + band.upper())

    # Top 10 channels with largest absolute differences
    order = np.argsort(np.abs(difference))[::-1]

    print("Top 10 channels:")

    for rank in range(10):
        ch = eeg.ch_names[order[rank]]

        print(
            f"{rank + 1:2d}. "
            f"{ch:6s} "
            f"Remember={r_mean[order[rank]]:.4f} "
            f"Ignore={i_mean[order[rank]]:.4f} "
            f"Diff={difference[order[rank]]:.4f}"
        )

# ---------------------------------------------------------
# Save numerical results
# ---------------------------------------------------------

output = os.path.join(
    BASE,
    "processed",
    "sub-009_ses-01_task-WorkingMemory_run-1_spatial.npz"
)

np.savez(
    output,
    channels=np.array(eeg.ch_names),
    theta_remember=results["theta"]["remember"],
    theta_ignore=results["theta"]["ignore"],
    theta_difference=results["theta"]["difference"],
    alpha_remember=results["alpha"]["remember"],
    alpha_ignore=results["alpha"]["ignore"],
    alpha_difference=results["alpha"]["difference"],
    beta_remember=results["beta"]["remember"],
    beta_ignore=results["beta"]["ignore"],
    beta_difference=results["beta"]["difference"]
)

print("\n" + "=" * 70)
print("NUMERICAL RESULTS SAVED")
print("=" * 70)

print(output)

# ---------------------------------------------------------
# Topographic maps
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CREATING TOPOGRAPHIC MAPS")
print("=" * 70)

fig_dir = os.path.join(
    BASE,
    "results"
)

os.makedirs(fig_dir, exist_ok=True)

for band in bands:

    difference = results[band]["difference"]

    fig, ax = plt.subplots(figsize=(7, 6))

    mne.viz.plot_topomap(
        difference,
        eeg.info,
        axes=ax,
        show=False
    )

    ax.set_title(
        f"{band.capitalize()} Power Difference\n"
        "Remember - Ignore"
    )

    filename = os.path.join(
        fig_dir,
        f"sub-009_run-1_{band}_remember_vs_ignore.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", filename)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("Spatial analysis completed successfully.")