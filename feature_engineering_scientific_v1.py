# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import mne

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

EPOCH_DIR = BASE / "final_dataset" / "perturbation" / "epochs"
BEHAVIOR_FILE = (
    BASE
    / "features"
    / "behavior_aligned"
    / "final"
    / "final_behavioral_trials.csv"
)

OUT_DIR = BASE / "features" / "scientific_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "scientific_features_v1.csv"

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

EPS = 1e-12

# ---------------------------------------------------------
# LOAD BEHAVIOR
# ---------------------------------------------------------

behavior = pd.read_csv(BEHAVIOR_FILE)

# Key used to connect trial-level behavior to EEG epochs
behavior_key = (
    behavior["subject"].astype(str)
    + "_"
    + behavior["run"].astype(str)
    + "_"
    + behavior["trial"].astype(str)
)

behavior = behavior.copy()
behavior["behavior_key"] = behavior_key

# ---------------------------------------------------------
# CHANNEL REGIONS
# ---------------------------------------------------------

FRONTAL_PREFIX = (
    "AF", "FP", "F", "FC"
)

PARIETAL_PREFIX = (
    "P", "CP", "PO"
)

# ---------------------------------------------------------
# STORAGE
# ---------------------------------------------------------

all_rows = []

files = sorted(EPOCH_DIR.glob("*_final_epo.fif"))

print("=" * 90)
print("SCIENTIFIC EEG FEATURE ENGINEERING V1")
print("=" * 90)
print(f"Files: {len(files)}")

# ---------------------------------------------------------
# PROCESS
# ---------------------------------------------------------

for i, f in enumerate(files, 1):

    print(f"[{i}/{len(files)}] {f.name}")

    epochs = mne.read_epochs(
        f,
        preload=True,
        verbose=False
    )

    data = epochs.get_data()

    sfreq = epochs.info["sfreq"]

    subject = f.name.split("_run-")[0]
    run = f.name.split("_run-")[1].split("_")[0]

    # -----------------------------------------------------
    # Only EEG channels
    # -----------------------------------------------------

    eeg_indices = [
        j for j, ch in enumerate(epochs.ch_names)
        if epochs.get_channel_types()[j] == "eeg"
    ]

    eeg_names = [
        epochs.ch_names[j]
        for j in eeg_indices
    ]

    data = data[:, eeg_indices, :]

    # -----------------------------------------------------
    # Welch PSD
    # -----------------------------------------------------

    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=1.0,
        fmax=40.0,
        n_fft=min(256, data.shape[-1]),
        verbose=False
    )

    # psd:
    # epochs x channels x frequencies

    # -----------------------------------------------------
    # Absolute + Relative band power
    # -----------------------------------------------------

    band_power = {}

    for band, (lo, hi) in BANDS.items():

        mask = (freqs >= lo) & (freqs < hi)

        power = psd[:, :, mask].mean(axis=-1)

        band_power[band] = power

    total_power = sum(band_power.values()) + EPS

    # -----------------------------------------------------
    # REGION INDEX
    # -----------------------------------------------------

    frontal_idx = [
        i for i, ch in enumerate(eeg_names)
        if ch.upper().startswith(FRONTAL_PREFIX)
    ]

    parietal_idx = [
        i for i, ch in enumerate(eeg_names)
        if ch.upper().startswith(PARIETAL_PREFIX)
    ]

    # -----------------------------------------------------
    # CREATE FEATURES
    # -----------------------------------------------------

    for epoch_idx in range(data.shape[0]):

        row = {
            "file": f.name,
            "subject": subject,
            "run": run,
            "epoch": epoch_idx,
        }

        # -----------------------------
        # Global band features
        # -----------------------------

        for band in BANDS:

            abs_power = band_power[band][epoch_idx].mean()

            rel_power = (
                band_power[band][epoch_idx].mean()
                / total_power[epoch_idx].mean()
            )

            row[f"{band}_abs_global"] = abs_power
            row[f"{band}_rel_global"] = rel_power

        # -----------------------------
        # Band ratios
        # -----------------------------

        theta = band_power["theta"][epoch_idx].mean()
        alpha = band_power["alpha"][epoch_idx].mean()
        beta = band_power["beta"][epoch_idx].mean()
        delta = band_power["delta"][epoch_idx].mean()

        row["theta_alpha_ratio"] = theta / (alpha + EPS)
        row["theta_beta_ratio"] = theta / (beta + EPS)
        row["theta_alpha_beta_ratio"] = theta / (
            alpha + beta + EPS
        )
        row["delta_beta_ratio"] = delta / (beta + EPS)

        # -----------------------------
        # Frontal features
        # -----------------------------

        if frontal_idx:

            for band in BANDS:

                row[f"{band}_frontal"] = (
                    band_power[band][epoch_idx, frontal_idx].mean()
                )

        # -----------------------------
        # Parietal features
        # -----------------------------

        if parietal_idx:

            for band in BANDS:

                row[f"{band}_parietal"] = (
                    band_power[band][epoch_idx, parietal_idx].mean()
                )

        # -----------------------------
        # Frontal / Parietal ratios
        # -----------------------------

        if frontal_idx and parietal_idx:

            row["theta_frontal_parietal_ratio"] = (
                band_power["theta"][epoch_idx, frontal_idx].mean()
                /
                (
                    band_power["theta"][epoch_idx, parietal_idx].mean()
                    + EPS
                )
            )

            row["alpha_frontal_parietal_ratio"] = (
                band_power["alpha"][epoch_idx, frontal_idx].mean()
                /
                (
                    band_power["alpha"][epoch_idx, parietal_idx].mean()
                    + EPS
                )
            )

        all_rows.append(row)

# ---------------------------------------------------------
# DATAFRAME
# ---------------------------------------------------------

features = pd.DataFrame(all_rows)

# ---------------------------------------------------------
# BASIC NUMERICAL QC
# ---------------------------------------------------------

numeric_cols = features.select_dtypes(
    include=[np.number]
).columns

nan_count = int(
    features[numeric_cols].isna().sum().sum()
)

inf_count = int(
    np.isinf(
        features[numeric_cols].to_numpy()
    ).sum()
)

print()
print("=" * 90)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 90)

print(f"Rows:          {len(features):,}")
print(f"Columns:       {len(features.columns)}")
print(f"Subjects:      {features['subject'].nunique()}")
print(f"Runs:          {features[['subject','run']].drop_duplicates().shape[0]}")
print(f"NaN values:    {nan_count}")
print(f"Inf values:    {inf_count}")

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

features.to_csv(
    OUTPUT,
    index=False
)

print()
print("Saved:")
print(OUTPUT)
print()
print("READ-ONLY INPUT")
print("No previous files modified.")