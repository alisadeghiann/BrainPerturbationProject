import os
import numpy as np
import mne
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

SUBJECT = "sub-009"
SESSION = "ses-01"
TASK = "WorkingMemory"

RUNS = [1, 2]

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

FDR_ALPHA = 0.05


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_epoch_file(run):
    return os.path.join(
        PROCESSED_DIR,
        f"{SUBJECT}_{SESSION}_task-{TASK}_run-{run}_clean-epo.fif"
    )


def load_epochs(run):
    path = get_epoch_file(run)

    print("\n" + "=" * 70)
    print(f"RUN {run}")
    print("=" * 70)
    print("Loading:")
    print(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Epoch file not found:\n{path}"
        )

    epochs = mne.read_epochs(path, preload=True, verbose=True)

    # Only EEG channels
    epochs = epochs.copy().pick("eeg")

    print("\nEpoch information:")
    print(f"Epochs: {len(epochs)}")
    print(f"Channels: {len(epochs.ch_names)}")
    print(f"Sampling rate: {epochs.info['sfreq']}")
    print(f"Time range: {epochs.times[0]:.3f} to {epochs.times[-1]:.3f}")

    return epochs


def get_condition_epochs(epochs):
    """
    Extract Remember and Ignore epochs.
    """

    available = set(epochs.event_id.keys())

    if "to_remember" not in available:
        raise RuntimeError(
            "Condition 'to_remember' not found in epochs."
        )

    if "to_ignore" not in available:
        raise RuntimeError(
            "Condition 'to_ignore' not found in epochs."
        )

    remember = epochs["to_remember"]
    ignore = epochs["to_ignore"]

    print("\nEpoch counts:")
    print(f"Remember: {len(remember)}")
    print(f"Ignore:   {len(ignore)}")

    return remember, ignore


def compute_psd_band_power(epochs, band_name, fmin, fmax):
    """
    Compute Welch PSD and average power inside a frequency band.

    Output:
        shape = (n_epochs, n_channels)
    """

    data = epochs.get_data()

    sfreq = epochs.info["sfreq"]

    n_epochs, n_channels, n_times = data.shape

    print(f"\nComputing PSD for {band_name.upper()}...")
    print(f"Input shape: {data.shape}")

    # --------------------------------------------------------
    # IMPORTANT:
    # n_times = 501
    # Therefore n_fft MUST NOT be > 501 unless n_per_seg
    # is explicitly provided.
    #
    # We use n_fft = 500 for safe operation.
    # --------------------------------------------------------

    n_fft = min(500, n_times)

    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_per_seg=n_fft,
        n_overlap=n_fft // 2,
        average="mean",
        verbose=False
    )

    # PSD shape:
    # epochs x channels x frequencies

    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(freq_mask):
        raise RuntimeError(
            f"No frequencies found for band {band_name}"
        )

    band_power = psd[:, :, freq_mask].mean(axis=2)

    print(f"PSD shape: {psd.shape}")
    print(f"Band power shape: {band_power.shape}")
    print(
        f"Frequency range used: "
        f"{freqs[freq_mask][0]:.2f} - "
        f"{freqs[freq_mask][-1]:.2f} Hz"
    )

    return band_power


def cohens_d(x, y):
    """
    Cohen's d for two independent groups.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    nx = len(x)
    ny = len(y)

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled_sd = np.sqrt(
        ((nx - 1) * vx + (ny - 1) * vy)
        / (nx + ny - 2)
    )

    if pooled_sd == 0:
        return 0.0

    return (np.mean(x) - np.mean(y)) / pooled_sd


def statistical_analysis(remember, ignore, ch_names):
    """
    Channel-wise independent t-test + FDR correction.
    """

    n_channels = len(ch_names)

    t_values = np.zeros(n_channels)
    p_values = np.zeros(n_channels)
    d_values = np.zeros(n_channels)

    for ch in range(n_channels):

        x = remember[:, ch]
        y = ignore[:, ch]

        t, p = ttest_ind(
            x,
            y,
            equal_var=False,
            nan_policy="omit"
        )

        t_values[ch] = t
        p_values[ch] = p
        d_values[ch] = cohens_d(x, y)

    # --------------------------------------------------------
    # FDR correction
    # --------------------------------------------------------

    reject, p_fdr, _, _ = multipletests(
        p_values,
        alpha=FDR_ALPHA,
        method="fdr_bh"
    )

    significant_indices = np.where(reject)[0]

    return {
        "t": t_values,
        "p": p_values,
        "p_fdr": p_fdr,
        "d": d_values,
        "reject": reject,
        "significant_indices": significant_indices,
    }


def save_statistics_csv(
    band_name,
    result,
    ch_names,
    output_path
):
    """
    Save channel-wise statistical results as CSV.
    """

    import csv

    rows = []

    for i, ch in enumerate(ch_names):

        rows.append([
            ch,
            result["t"][i],
            result["p"][i],
            result["p_fdr"][i],
            result["d"][i],
            bool(result["reject"][i])
        ])

    rows.sort(
        key=lambda row: abs(row[4]),
        reverse=True
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "channel",
            "t",
            "p",
            "p_fdr",
            "cohens_d",
            "significant_fdr"
        ])

        writer.writerows(rows)


def create_topomap(
    band_name,
    remember,
    ignore,
    result,
    info,
    output_path
):
    """
    Create Remember vs Ignore difference topomap.
    """

    ch_names = info["ch_names"]

    remember_mean = np.mean(remember, axis=0)
    ignore_mean = np.mean(ignore, axis=0)

    difference = remember_mean - ignore_mean

    print("\nCreating topographic map...")

    # --------------------------------------------------------
    # Use only channels available in the info structure.
    # This prevents index 69 errors.
    # --------------------------------------------------------

    n_channels = len(ch_names)

    if len(difference) != n_channels:
        raise RuntimeError(
            f"Channel mismatch:\n"
            f"difference = {len(difference)}\n"
            f"info channels = {n_channels}"
        )

    # Mask non-significant channels
    mask = result["reject"]

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    mne.viz.plot_topomap(
        difference,
        info,
        axes=ax,
        show=False,
        contours=6,
        sensors=True,
        mask=mask,
        mask_params=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            linewidth=0,
            markersize=5
        )
    )

    ax.set_title(
        f"{band_name.upper()} Power\n"
        f"Remember - Ignore\n"
        f"FDR q < {FDR_ALPHA}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# MAIN ANALYSIS
# ============================================================

print("=" * 70)
print("COMBINED RUN 1 + RUN 2 EEG ANALYSIS")
print("=" * 70)


all_results = {}

combined_info = None


# ============================================================
# PROCESS EACH RUN
# ============================================================

run_data = {}

for run in RUNS:

    epochs = load_epochs(run)

    remember_epochs, ignore_epochs = get_condition_epochs(
        epochs
    )

    run_results = {}

    for band_name, (fmin, fmax) in BANDS.items():

        print("\n" + "-" * 60)
        print(f"Extracting {band_name.upper()}")
        print("-" * 60)

        remember_power = compute_psd_band_power(
            remember_epochs,
            band_name,
            fmin,
            fmax
        )

        ignore_power = compute_psd_band_power(
            ignore_epochs,
            band_name,
            fmin,
            fmax
        )

        run_results[band_name] = {
            "remember": remember_power,
            "ignore": ignore_power
        }

    run_data[run] = run_results

    if combined_info is None:
        combined_info = epochs.info.copy()


# ============================================================
# COMBINE RUNS
# ============================================================

print("\n" + "=" * 70)
print("COMBINING RUN 1 + RUN 2")
print("=" * 70)

combined_data = {}

for band_name in BANDS.keys():

    remember_all = np.concatenate(
        [
            run_data[1][band_name]["remember"],
            run_data[2][band_name]["remember"]
        ],
        axis=0
    )

    ignore_all = np.concatenate(
        [
            run_data[1][band_name]["ignore"],
            run_data[2][band_name]["ignore"]
        ],
        axis=0
    )

    combined_data[band_name] = {
        "remember": remember_all,
        "ignore": ignore_all
    }

    print(f"\n{band_name.upper()}")
    print(
        "Combined Remember:",
        remember_all.shape
    )
    print(
        "Combined Ignore:",
        ignore_all.shape
    )


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)


for band_name in BANDS.keys():

    print("\n" + "-" * 60)
    print(band_name.upper())
    print("-" * 60)

    remember = combined_data[band_name]["remember"]
    ignore = combined_data[band_name]["ignore"]

    ch_names = combined_info["ch_names"]

    # Safety check
    if remember.shape[1] != len(ch_names):
        raise RuntimeError(
            f"Remember channel mismatch: "
            f"{remember.shape[1]} vs {len(ch_names)}"
        )

    if ignore.shape[1] != len(ch_names):
        raise RuntimeError(
            f"Ignore channel mismatch: "
            f"{ignore.shape[1]} vs {len(ch_names)}"
        )

    result = statistical_analysis(
        remember,
        ignore,
        ch_names
    )

    all_results[band_name] = result

    significant = result["significant_indices"]

    print(
        f"\nChannels surviving FDR: "
        f"{len(significant)}"
    )

    if len(significant) > 0:

        print("\nSignificant channels:")

        # Sort significant channels by absolute Cohen's d
        significant_sorted = sorted(
            significant,
            key=lambda i: abs(result["d"][i]),
            reverse=True
        )

        for i in significant_sorted:

            print(
                f"{ch_names[i]:>5s} "
                f"t={result['t'][i]:8.3f} "
                f"p={result['p'][i]:.6f} "
                f"FDR={result['p_fdr'][i]:.6f} "
                f"d={result['d'][i]:7.3f}"
            )

    else:

        print(
            "No channels survived FDR correction."
        )

    # --------------------------------------------------------
    # Top 10 effects
    # --------------------------------------------------------

    sorted_indices = np.argsort(
        np.abs(result["d"])
    )[::-1]

    print("\nTop 10 absolute effects:")

    for i in sorted_indices[:10]:

        print(
            f"{ch_names[i]:>5s} "
            f"d={result['d'][i]:7.3f} "
            f"p={result['p'][i]:.6f} "
            f"FDR={result['p_fdr'][i]:.6f}"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        RESULTS_DIR,
        f"{SUBJECT}_combined_{band_name}_statistics.csv"
    )

    save_statistics_csv(
        band_name,
        result,
        ch_names,
        csv_path
    )

    print(
        f"\nCSV saved: {csv_path}"
    )

    # --------------------------------------------------------
    # Topographic map
    # --------------------------------------------------------

    topo_path = os.path.join(
        RESULTS_DIR,
        f"{SUBJECT}_combined_{band_name}_remember_vs_ignore.png"
    )

    create_topomap(
        band_name,
        remember,
        ignore,
        result,
        combined_info,
        topo_path
    )


# ============================================================
# SAVE NPZ
# ============================================================

npz_path = os.path.join(
    PROCESSED_DIR,
    f"{SUBJECT}_combined_run1_run2_statistics.npz"
)

save_dict = {}

for band_name, result in all_results.items():

    save_dict[f"{band_name}_t"] = result["t"]
    save_dict[f"{band_name}_p"] = result["p"]
    save_dict[f"{band_name}_p_fdr"] = result["p_fdr"]
    save_dict[f"{band_name}_d"] = result["d"]
    save_dict[f"{band_name}_reject"] = result["reject"]


save_dict["ch_names"] = np.array(
    combined_info["ch_names"],
    dtype=object
)

np.savez(
    npz_path,
    **save_dict
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for band_name, result in all_results.items():

    significant_count = np.sum(
        result["reject"]
    )

    max_effect_idx = np.argmax(
        np.abs(result["d"])
    )

    print(
        f"\n{band_name.upper()}:"
    )

    print(
        f"  Significant channels: "
        f"{significant_count}"
    )

    print(
        f"  Strongest channel: "
        f"{combined_info['ch_names'][max_effect_idx]}"
    )

    print(
        f"  Cohen's d: "
        f"{result['d'][max_effect_idx]:.3f}"
    )

    print(
        f"  p-value: "
        f"{result['p'][max_effect_idx]:.6f}"
    )

    print(
        f"  FDR: "
        f"{result['p_fdr'][max_effect_idx]:.6f}"
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nNPZ saved:")
print(npz_path)

print("\nCSV and topographic maps saved in:")
print(RESULTS_DIR)

print("\n[DONE]")