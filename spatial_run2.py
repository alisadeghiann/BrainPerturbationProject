import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")
PROCESSED = PROJECT_DIR / "processed"
RESULTS = PROJECT_DIR / "results"
RESULTS.mkdir(exist_ok=True)

EEG_FILE = PROCESSED / "sub-009_ses-01_task-WorkingMemory_run-2_clean-epo.fif"
PSD_FILE = PROCESSED / "sub-009_ses-01_task-WorkingMemory_run-2_psd.npz"

print("=" * 60)
print("LOADING RUN 2 CLEAN EEG")
print("=" * 60)

epochs = mne.read_epochs(EEG_FILE, preload=True)

remember = epochs["to_remember"]
ignore = epochs["to_ignore"]

eeg_picks = mne.pick_types(
    epochs.info,
    eeg=True,
    exclude="bads"
)

print("Remember:", len(remember))
print("Ignore:", len(ignore))
print("EEG channels:", len(eeg_picks))

print("\n" + "=" * 60)
print("LOADING PSD RESULTS")
print("=" * 60)

psd = np.load(PSD_FILE)

ch_names = list(psd["ch_names"])

print("Channels:", len(ch_names))

bands = ["theta", "alpha", "beta"]

for band in bands:

    print("\n" + "=" * 60)
    print(band.upper())
    print("=" * 60)

    remember_power = psd[f"{band}_remember"]
    ignore_power = psd[f"{band}_ignore"]

    remember_mean = remember_power.mean(axis=0)
    ignore_mean = ignore_power.mean(axis=0)

    difference = remember_mean - ignore_mean

    order = np.argsort(difference)

    print("Top 10 channels:")
    for rank, idx in enumerate(order[:10], 1):
        print(
            f"{rank:2d}. {ch_names[idx]:5s} "
            f"Remember={remember_mean[idx]:.4f} "
            f"Ignore={ignore_mean[idx]:.4f} "
            f"Diff={difference[idx]:.4f}"
        )

    print("\nTop 10 positive differences:")
    positive_order = np.argsort(difference)[::-1]

    for rank, idx in enumerate(positive_order[:10], 1):
        print(
            f"{rank:2d}. {ch_names[idx]:5s} "
            f"Diff={difference[idx]:.4f}"
        )

    print("\nCreating topographic map...")

    info = epochs.copy().pick(eeg_picks).info

    evoked = mne.EvokedArray(
        difference[:, np.newaxis],
        info,
        tmin=0
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    mne.viz.plot_topomap(
        difference,
        info,
        axes=ax,
        show=False
    )

    ax.set_title(
        f"Run 2 - {band.upper()}\n"
        "Remember - Ignore"
    )

    output = RESULTS / (
        f"sub-009_run-2_{band}_"
        "remember_vs_ignore.png"
    )

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print("Run 2 spatial analysis completed.")