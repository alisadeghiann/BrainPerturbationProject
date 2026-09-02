import os
import numpy as np
import mne
import matplotlib.pyplot as plt

# ============================================================
# EFFECT SIZE TOPOGRAPHIC MAPS
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

STAT_FILE = os.path.join(
    PROCESSED_DIR,
    "sub-009_combined_run1_run2_statistics.npz"
)

EPOCHS_FILE = os.path.join(
    PROCESSED_DIR,
    "sub-009_ses-01_task-WorkingMemory_run-1_clean-epo.fif"
)

# ============================================================
# LOAD STATISTICAL RESULTS
# ============================================================

print("=" * 70)
print("LOADING STATISTICAL RESULTS")
print("=" * 70)

data = np.load(STAT_FILE, allow_pickle=True)

print("Available keys:")
for key in data.files:
    print(" ", key)

# ============================================================
# LOAD EEG CHANNEL INFORMATION
# ============================================================

print()
print("=" * 70)
print("LOADING CHANNEL INFORMATION")
print("=" * 70)

epochs = mne.read_epochs(
    EPOCHS_FILE,
    preload=False,
    verbose=False
)

eeg_channels = [
    ch
    for ch, ch_type in zip(
        epochs.ch_names,
        epochs.get_channel_types()
    )
    if ch_type == "eeg"
]

saved_channels = list(data["ch_names"])

print("Total channels:", len(epochs.ch_names))
print("EEG channels:", len(eeg_channels))
print("Channels in statistics:", len(saved_channels))

# ============================================================
# MATCH CHANNELS
# ============================================================

if set(saved_channels) != set(eeg_channels):

    print()
    print("WARNING: Channel sets differ.")
    print("Matching common channels...")

common_channels = [
    ch for ch in saved_channels
    if ch in eeg_channels
]

print("Common channels:", len(common_channels))

# ============================================================
# GET EFFECT SIZE
# ============================================================

def get_effect_size(band):

    key = f"{band}_d"

    if key not in data.files:
        raise KeyError(
            f"{key} not found in statistical results."
        )

    raw_d = np.asarray(data[key]).flatten()

    channel_to_d = dict(
        zip(saved_channels, raw_d)
    )

    d = np.array([
        channel_to_d[ch]
        for ch in common_channels
    ])

    return d


# ============================================================
# PREPARE MONTAGE
# ============================================================

print()
print("=" * 70)
print("PREPARING MONTAGE")
print("=" * 70)

montage = mne.channels.make_standard_montage(
    "standard_1020"
)

# ------------------------------------------------------------
# IMPORTANT:
# Remove duplicated / overlapping positions automatically
# ------------------------------------------------------------

montage_positions = montage.get_positions()["ch_pos"]

valid_channels = []
valid_positions = []

seen_positions = []

for ch in common_channels:

    if ch not in montage_positions:
        print(
            f"Skipping {ch}: no position in montage"
        )
        continue

    pos = montage_positions[ch][:2]

    # Check if position already exists
    duplicate = False

    for old_pos in seen_positions:

        if np.linalg.norm(
            np.array(pos) - np.array(old_pos)
        ) < 1e-6:

            duplicate = True
            break

    if duplicate:

        print(
            f"Skipping {ch}: overlapping position"
        )

    else:

        valid_channels.append(ch)
        valid_positions.append(pos)
        seen_positions.append(pos)

print()
print("Channels before removing overlaps:", len(common_channels))
print("Channels used for topomap:", len(valid_channels))

# ============================================================
# CREATE CUSTOM INFO
# ============================================================

info = mne.create_info(
    ch_names=valid_channels,
    sfreq=250.0,
    ch_types="eeg"
)

# Set only valid channel locations
ch_pos = {
    ch: montage_positions[ch]
    for ch in valid_channels
}

custom_montage = mne.channels.make_dig_montage(
    ch_pos=ch_pos,
    coord_frame="head"
)

info.set_montage(
    custom_montage,
    on_missing="ignore"
)

# ============================================================
# CREATE TOPOGRAMS
# ============================================================

bands = [
    ("theta", "Theta"),
    ("alpha", "Alpha"),
    ("beta", "Beta")
]

for band_key, band_name in bands:

    print()
    print("=" * 70)
    print(f"{band_name.upper()} EFFECT SIZE")
    print("=" * 70)

    d_all = get_effect_size(band_key)

    # Map d values to channels
    d_dict = dict(
        zip(common_channels, d_all)
    )

    d = np.array([
        d_dict[ch]
        for ch in valid_channels
    ])

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("Channels:", len(d))

    print(
        "Minimum Cohen's d:",
        np.min(d)
    )

    print(
        "Maximum Cohen's d:",
        np.max(d)
    )

    print(
        "Mean Cohen's d:",
        np.mean(d)
    )

    # --------------------------------------------------------
    # TOP EFFECTS
    # --------------------------------------------------------

    order = np.argsort(np.abs(d))[::-1]

    print()
    print("Top 10 absolute effects:")

    for idx in order[:10]:

        print(
            f"{valid_channels[idx]:>5} "
            f"d={d[idx]: .4f}"
        )

    # --------------------------------------------------------
    # SYMMETRIC COLOR SCALE
    # --------------------------------------------------------

    max_abs = np.max(
        np.abs(d)
    )

    if max_abs == 0:
        max_abs = 1.0

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    im, contour = mne.viz.plot_topomap(
        d,
        info,
        axes=ax,
        cmap="RdBu_r",
        contours=6,
        sensors=True,
        show=False,
        vlim=(-max_abs, max_abs)
    )

    ax.set_title(
        f"{band_name} Band\n"
        f"Remember vs Ignore — Cohen's d",
        fontsize=15
    )

    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------

    cbar = plt.colorbar(
        im,
        ax=ax,
        shrink=0.8
    )

    cbar.set_label(
        "Cohen's d",
        rotation=270,
        labelpad=18
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = os.path.join(
        RESULTS_DIR,
        f"sub-009_{band_key}_cohens_d_topomap.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print()
    print("Saved:")
    print(output_file)

# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("EFFECT SIZE TOPOGRAPHIC ANALYSIS COMPLETED")
print("=" * 70)

print()
print("Generated files:")

for band_key, _ in bands:

    print(
        os.path.join(
            RESULTS_DIR,
            f"sub-009_{band_key}_cohens_d_topomap.png"
        )
    )

print()
print("[DONE]")