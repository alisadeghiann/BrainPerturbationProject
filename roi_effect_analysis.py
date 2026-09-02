# ============================================================
# ROI-LEVEL EEG POWER ANALYSIS
# Brain Perturbation Project
# Remember vs Ignore
# Runs 1 + 2 combined
# ============================================================

import os
import numpy as np
import pandas as pd
import mne
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

RUN_FILES = [
    os.path.join(
        PROCESSED_DIR,
        "sub-009_ses-01_task-WorkingMemory_run-1_clean-epo.fif"
    ),
    os.path.join(
        PROCESSED_DIR,
        "sub-009_ses-01_task-WorkingMemory_run-2_clean-epo.fif"
    )
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0)
}

# Welch settings
N_FFT = 512

# ============================================================
# ROI DEFINITIONS
# ============================================================

ROI_CHANNELS = {

    "Frontal": [
        "AF7", "AF3", "AFZ", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "FZ",
        "F2", "F4", "F6", "F8", "F9", "F10"
    ],

    "FrontoCentral": [
        "FC5", "FC3", "FC1",
        "FCZ",
        "FC2", "FC4"
    ],

    "Central": [
        "C5", "C3", "C1",
        "CZ",
        "C2", "C4"
    ],

    "CentroParietal": [
        "CP5", "CP3", "CP1",
        "CPZ",
        "CP2", "CP4"
    ],

    "Parietal": [
        "P7", "P5", "P3", "P1",
        "PZ",
        "P2", "P4", "P6", "P8"
    ],

    "ParietoOccipital": [
        "PO7", "PO3", "POZ",
        "PO4", "PO8", "PO9", "PO10"
    ],

    "Occipital": [
        "O1", "OZ", "O2"
    ],

    "Temporal": [
        "FT7", "FT9",
        "T7",
        "TP7", "TP9",
        "TP8", "TP10",
        "T8",
        "FT8", "FT10"
    ],

    "Midline": [
        "AFZ", "FZ", "FCZ",
        "CZ", "CPZ",
        "PZ", "POZ", "OZ"
    ]
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def cohens_d(group1, group2):
    """
    Cohen's d for two independent groups.
    """

    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)

    n1 = len(group1)
    n2 = len(group2)

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    pooled_sd = np.sqrt(
        ((n1 - 1) * var1 + (n2 - 1) * var2)
        / (n1 + n2 - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (mean1 - mean2) / pooled_sd


def compute_band_power(data, sfreq, fmin, fmax):
    """
    Compute mean PSD power within a frequency band.

    Input:
        data = epochs x channels x time

    Output:
        epochs x channels
    """

    n_times = data.shape[-1]

    # Important:
    # n_fft cannot be larger than n_times unless n_per_seg
    # is explicitly specified.
    n_per_seg = min(256, n_times)

    n_fft = min(N_FFT, n_per_seg)

    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=n_per_seg // 2,
        verbose=False
    )

    # Average power across frequencies
    band_mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(band_mask):
        raise ValueError(
            f"No frequencies found for band {fmin}-{fmax} Hz"
        )

    band_power = np.mean(
        psd[:, :, band_mask],
        axis=-1
    )

    return band_power


# ============================================================
# LOAD RUNS
# ============================================================

all_remember = {
    "theta": [],
    "alpha": [],
    "beta": []
}

all_ignore = {
    "theta": [],
    "alpha": [],
    "beta": []
}

channel_names = None
sfreq = None

print("=" * 70)
print("ROI-LEVEL EEG POWER ANALYSIS")
print("=" * 70)

# ============================================================
# PROCESS EACH RUN
# ============================================================

for run_idx, run_file in enumerate(RUN_FILES, start=1):

    print("\n")
    print("=" * 70)
    print(f"RUN {run_idx}")
    print("=" * 70)

    print(f"Loading:")
    print(run_file)

    if not os.path.exists(run_file):
        raise FileNotFoundError(
            f"File not found:\n{run_file}"
        )

    epochs = mne.read_epochs(
        run_file,
        preload=True,
        verbose=True
    )

    print("\nEpoch information:")
    print(f"Epochs: {len(epochs)}")
    print(f"Channels: {len(epochs.ch_names)}")
    print(f"Sampling rate: {epochs.info['sfreq']}")
    print(
        f"Time range: "
        f"{epochs.times[0]:.3f} to {epochs.times[-1]:.3f}"
    )

    # --------------------------------------------------------
    # Channel information
    # --------------------------------------------------------

    if channel_names is None:
        channel_names = list(epochs.ch_names)
        sfreq = epochs.info["sfreq"]

    else:

        if list(epochs.ch_names) != channel_names:
            raise ValueError(
                "Channel order differs between runs."
            )

    # --------------------------------------------------------
    # Identify Remember / Ignore
    # --------------------------------------------------------

    event_names = list(epochs.event_id.keys())

    print("\nAvailable events:")
    print(event_names)

    remember_key = None
    ignore_key = None

    for key in event_names:

        key_lower = key.lower()

        if "remember" in key_lower:
            remember_key = key

        if "ignore" in key_lower:
            ignore_key = key

    if remember_key is None or ignore_key is None:

        raise ValueError(
            "\nCould not identify Remember / Ignore events.\n"
            f"Available events: {event_names}"
        )

    remember_epochs = epochs[remember_key]
    ignore_epochs = epochs[ignore_key]

    print("\nEpoch counts:")
    print(f"Remember: {len(remember_epochs)}")
    print(f"Ignore:   {len(ignore_epochs)}")

    # --------------------------------------------------------
    # Extract raw data
    # --------------------------------------------------------

    remember_data = remember_epochs.get_data()
    ignore_data = ignore_epochs.get_data()

    # --------------------------------------------------------
    # Frequency bands
    # --------------------------------------------------------

    for band_name, (fmin, fmax) in BANDS.items():

        print("\n" + "-" * 60)
        print(f"Extracting {band_name.upper()}")
        print("-" * 60)

        print("Computing PSD for Remember...")

        remember_power = compute_band_power(
            remember_data,
            sfreq,
            fmin,
            fmax
        )

        print(
            f"Remember power shape: "
            f"{remember_power.shape}"
        )

        print("Computing PSD for Ignore...")

        ignore_power = compute_band_power(
            ignore_data,
            sfreq,
            fmin,
            fmax
        )

        print(
            f"Ignore power shape: "
            f"{ignore_power.shape}"
        )

        all_remember[band_name].append(
            remember_power
        )

        all_ignore[band_name].append(
            ignore_power
        )


# ============================================================
# COMBINE RUNS
# ============================================================

print("\n")
print("=" * 70)
print("COMBINING RUN 1 + RUN 2")
print("=" * 70)

combined = {}

for band_name in BANDS:

    combined_remember = np.concatenate(
        all_remember[band_name],
        axis=0
    )

    combined_ignore = np.concatenate(
        all_ignore[band_name],
        axis=0
    )

    combined[band_name] = {
        "remember": combined_remember,
        "ignore": combined_ignore
    }

    print(f"\n{band_name.upper()}")
    print(
        f"Combined Remember: "
        f"{combined_remember.shape}"
    )
    print(
        f"Combined Ignore:   "
        f"{combined_ignore.shape}"
    )


# ============================================================
# CHANNEL INDEX
# ============================================================

channel_to_index = {
    ch.upper(): idx
    for idx, ch in enumerate(channel_names)
}

print("\n")
print("=" * 70)
print("CHANNEL / ROI CHECK")
print("=" * 70)

print(f"Total EEG channels: {len(channel_names)}")

# ============================================================
# ROI VALIDATION
# ============================================================

valid_rois = {}

for roi_name, roi_channels in ROI_CHANNELS.items():

    valid_channels = []

    for ch in roi_channels:

        if ch.upper() in channel_to_index:

            valid_channels.append(ch)

        else:

            print(
                f"WARNING: {ch} not found "
                f"in EEG channels"
            )

    valid_rois[roi_name] = valid_channels

    print(
        f"{roi_name:20s}: "
        f"{len(valid_channels)} channels"
    )


# ============================================================
# ROI ANALYSIS
# ============================================================

results = []

print("\n")
print("=" * 70)
print("ROI STATISTICAL ANALYSIS")
print("=" * 70)

for band_name in BANDS:

    print("\n")
    print("-" * 70)
    print(f"{band_name.upper()} ROI ANALYSIS")
    print("-" * 70)

    remember = combined[band_name]["remember"]
    ignore = combined[band_name]["ignore"]

    band_results = []

    for roi_name, roi_channels in valid_rois.items():

        if len(roi_channels) == 0:
            continue

        indices = [
            channel_to_index[ch.upper()]
            for ch in roi_channels
        ]

        # ----------------------------------------------------
        # Average power across ROI channels
        # ----------------------------------------------------

        remember_roi = np.mean(
            remember[:, indices],
            axis=1
        )

        ignore_roi = np.mean(
            ignore[:, indices],
            axis=1
        )

        # ----------------------------------------------------
        # Log-transform power
        # ----------------------------------------------------
        # EEG power is typically positively skewed.
        # Log10 makes distributions more suitable for
        # parametric statistical testing.

        remember_log = np.log10(
            np.maximum(remember_roi, 1e-20)
        )

        ignore_log = np.log10(
            np.maximum(ignore_roi, 1e-20)
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        t_stat, p_value = ttest_ind(
            remember_log,
            ignore_log,
            equal_var=False
        )

        d = cohens_d(
            remember_log,
            ignore_log
        )

        mean_remember = np.mean(
            remember_log
        )

        mean_ignore = np.mean(
            ignore_log
        )

        difference = (
            mean_remember -
            mean_ignore
        )

        band_results.append({
            "band": band_name,
            "ROI": roi_name,
            "n_channels": len(indices),
            "channels": ", ".join(roi_channels),
            "n_remember": len(remember_log),
            "n_ignore": len(ignore_log),
            "mean_log10_power_remember": mean_remember,
            "mean_log10_power_ignore": mean_ignore,
            "difference_remember_minus_ignore": difference,
            "t": t_stat,
            "p": p_value,
            "cohens_d": d
        })

    # --------------------------------------------------------
    # FDR across ROIs within each frequency band
    # --------------------------------------------------------

    p_values = np.array([
        row["p"]
        for row in band_results
    ])

    reject, p_fdr, _, _ = multipletests(
        p_values,
        alpha=0.05,
        method="fdr_bh"
    )

    for i, row in enumerate(band_results):

        row["p_fdr"] = p_fdr[i]
        row["significant_fdr"] = bool(
            reject[i]
        )

        results.append(row)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    for row in band_results:

        print(
            f"{row['ROI']:20s} "
            f"d={row['cohens_d']: .3f} "
            f"p={row['p']:.6f} "
            f"FDR={row['p_fdr']:.6f} "
            f"{'SIGNIFICANT' if row['significant_fdr'] else ''}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

csv_file = os.path.join(
    RESULTS_DIR,
    "sub-009_ROI_power_statistics.csv"
)

results_df.to_csv(
    csv_file,
    index=False
)

print("\n")
print("=" * 70)
print("ROI RESULTS SAVED")
print("=" * 70)

print(csv_file)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL ROI SUMMARY")
print("=" * 70)

for band_name in BANDS:

    band_df = results_df[
        results_df["band"] == band_name
    ]

    sig_df = band_df[
        band_df["significant_fdr"]
    ]

    print("\n" + band_name.upper())

    print(
        f"Significant ROIs: "
        f"{len(sig_df)}/{len(band_df)}"
    )

    if len(sig_df) > 0:

        sig_sorted = sig_df.sort_values(
            "cohens_d"
        )

        print("\nSignificant ROIs:")

        for _, row in sig_sorted.iterrows():

            print(
                f"  {row['ROI']:20s} "
                f"d={row['cohens_d']: .3f} "
                f"p={row['p']:.6f} "
                f"FDR={row['p_fdr']:.6f}"
            )

        strongest = sig_df.loc[
            sig_df["cohens_d"].abs().idxmax()
        ]

        print("\nStrongest ROI:")

        print(
            f"  {strongest['ROI']}"
        )

        print(
            f"  Cohen's d = "
            f"{strongest['cohens_d']:.3f}"
        )

        print(
            f"  FDR = "
            f"{strongest['p_fdr']:.6f}"
        )

    else:

        print(
            "  No ROI survived FDR correction."
        )


# ============================================================
# SAVE COMPACT SUMMARY
# ============================================================

summary_rows = []

for band_name in BANDS:

    band_df = results_df[
        results_df["band"] == band_name
    ]

    sig_df = band_df[
        band_df["significant_fdr"]
    ]

    if len(sig_df) > 0:

        strongest = sig_df.loc[
            sig_df["cohens_d"].abs().idxmax()
        ]

        summary_rows.append({
            "band": band_name,
            "significant_ROIs": len(sig_df),
            "total_ROIs": len(band_df),
            "strongest_ROI": strongest["ROI"],
            "strongest_d": strongest["cohens_d"],
            "strongest_p": strongest["p"],
            "strongest_p_fdr": strongest["p_fdr"]
        })

    else:

        summary_rows.append({
            "band": band_name,
            "significant_ROIs": 0,
            "total_ROIs": len(band_df),
            "strongest_ROI": "",
            "strongest_d": np.nan,
            "strongest_p": np.nan,
            "strongest_p_fdr": np.nan
        })

summary_df = pd.DataFrame(summary_rows)

summary_file = os.path.join(
    RESULTS_DIR,
    "sub-009_ROI_power_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

print("\n")
print("=" * 70)
print("SUMMARY SAVED")
print("=" * 70)

print(summary_file)

print("\n")
print("=" * 70)
print("ROI POWER ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)