import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# ============================================================
# FINAL EEG FIGURE
# Remember vs Ignore
# Theta / Alpha / Beta
# Channel-level + ROI-level results
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

STAT_FILE = os.path.join(
    PROCESSED_DIR,
    "sub-009_combined_run1_run2_statistics.npz"
)

ROI_FILE = os.path.join(
    RESULTS_DIR,
    "sub-009_ROI_power_statistics.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "sub-009_FINAL_EEG_FIGURE.png"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

BANDS = ["theta", "alpha", "beta"]

BAND_NAMES = {
    "theta": "Theta (4–8 Hz)",
    "alpha": "Alpha (8–13 Hz)",
    "beta": "Beta (13–30 Hz)"
}

# ============================================================
# LOAD STATISTICAL RESULTS
# ============================================================

print("=" * 70)
print("LOADING CHANNEL-LEVEL STATISTICAL RESULTS")
print("=" * 70)

if not os.path.exists(STAT_FILE):
    raise FileNotFoundError(
        f"Statistical file not found:\n{STAT_FILE}"
    )

stats = np.load(STAT_FILE, allow_pickle=True)

ch_names = list(stats["ch_names"])

print(f"Channels in statistics: {len(ch_names)}")

# ============================================================
# LOAD ROI RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LOADING ROI RESULTS")
print("=" * 70)

if not os.path.exists(ROI_FILE):
    raise FileNotFoundError(
        f"ROI file not found:\n{ROI_FILE}"
    )

roi_df = pd.read_csv(ROI_FILE)

print("ROI columns:")
print(list(roi_df.columns))

print("\nROI results:")
print(roi_df.to_string(index=False))


# ============================================================
# FIND ROI COLUMN NAMES SAFELY
# ============================================================

def find_column(df, candidates):

    lower_map = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().strip()

        if key in lower_map:
            return lower_map[key]

    # Partial matching
    for col in df.columns:
        col_lower = str(col).lower()

        for candidate in candidates:
            if candidate.lower() in col_lower:
                return col

    return None


ROI_NAME_COL = find_column(
    roi_df,
    [
        "roi",
        "region",
        "region_name",
        "ROI"
    ]
)

ROI_D_COL = find_column(
    roi_df,
    [
        "d",
        "cohens_d",
        "effect_size",
        "effect"
    ]
)

ROI_FDR_COL = find_column(
    roi_df,
    [
        "fdr",
        "p_fdr",
        "fdr_p"
    ]
)

ROI_SIG_COL = find_column(
    roi_df,
    [
        "significant",
        "significant_fdr",
        "reject"
    ]
)

print("\nDetected ROI columns:")
print("ROI:", ROI_NAME_COL)
print("Cohen's d:", ROI_D_COL)
print("FDR:", ROI_FDR_COL)
print("Significance:", ROI_SIG_COL)


# ============================================================
# LOAD MONTAGE
# ============================================================

print("\n" + "=" * 70)
print("PREPARING EEG MONTAGE")
print("=" * 70)

montage = mne.channels.make_standard_montage("standard_1020")

montage_names = set(montage.ch_names)

valid_channels = [
    ch for ch in ch_names
    if ch in montage_names
]

print(f"Statistics channels: {len(ch_names)}")
print(f"Channels with standard 10-20 positions: {len(valid_channels)}")


# ============================================================
# CREATE INFO
# ============================================================

info = mne.create_info(
    ch_names=valid_channels,
    sfreq=250,
    ch_types="eeg"
)

info.set_montage(
    montage,
    match_case=False,
    on_missing="ignore"
)


# ============================================================
# GET TOPOGRAPHIC POSITIONS
# ============================================================

positions = montage.get_positions()["ch_pos"]

topo_pos = np.array([
    positions[ch]
    for ch in valid_channels
])

# Use x/y coordinates
topo_pos = topo_pos[:, :2]


# ============================================================
# CHANNEL STATISTICS
# ============================================================

band_data = {}

for band in BANDS:

    key = f"{band}_d"

    if key not in stats:
        raise KeyError(
            f"Missing {key} in statistical results."
        )

    d_values = np.asarray(stats[key], dtype=float)

    if len(d_values) != len(ch_names):
        raise ValueError(
            f"{band}: Cohen's d length does not match "
            f"channel count."
        )

    # Select only channels with montage positions
    valid_indices = [
        i for i, ch in enumerate(ch_names)
        if ch in montage_names
    ]

    d_valid = d_values[valid_indices]

    band_data[band] = {
        "d": d_valid,
        "channels": valid_channels
    }

    print(
        f"{band.upper()}: "
        f"min={np.min(d_valid):.3f}, "
        f"max={np.max(d_valid):.3f}"
    )


# ============================================================
# COMMON COLOR SCALE
# ============================================================

all_d = np.concatenate([
    band_data[band]["d"]
    for band in BANDS
])

max_abs_d = np.max(np.abs(all_d))

# Keep symmetric scale
vmax = max_abs_d
vmin = -max_abs_d

print("\nTopographic color scale:")
print(f"vmin = {vmin:.3f}")
print(f"vmax = {vmax:.3f}")


# ============================================================
# CREATE FIGURE
# ============================================================

print("\n" + "=" * 70)
print("CREATING FINAL EEG FIGURE")
print("=" * 70)

fig = plt.figure(
    figsize=(18, 11)
)

# ------------------------------------------------------------
# TOP ROW: TOPOGRAPHS
# ------------------------------------------------------------

topo_axes = []

for i, band in enumerate(BANDS):

    ax = fig.add_subplot(
        2,
        3,
        i + 1
    )

    topo_axes.append(ax)

    d_values = band_data[band]["d"]

    print(
        f"\nCreating {band.upper()} topomap..."
    )

    im, _ = mne.viz.plot_topomap(
        d_values,
        info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        vlim=(vmin, vmax),
        contours=6,
        sensors=True,
        extrapolate="head"
    )

    ax.set_title(
        BAND_NAMES[band],
        fontsize=16,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Mark significant channels
    # --------------------------------------------------------

    reject_key = f"{band}_reject"

    if reject_key in stats:

        reject = np.asarray(
            stats[reject_key],
            dtype=bool
        )

        reject_valid = reject[
            [
                i
                for i, ch in enumerate(ch_names)
                if ch in montage_names
            ]
        ]

        sig_positions = topo_pos[
            reject_valid
        ]

        if len(sig_positions) > 0:

            ax.scatter(
                sig_positions[:, 0],
                sig_positions[:, 1],
                s=35,
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                zorder=10
            )

            print(
                f"  Significant channels: "
                f"{len(sig_positions)}"
            )

    else:

        print(
            f"  No reject array found for {band}"
        )


# ============================================================
# COLORBAR
# ============================================================

cbar_ax = fig.add_axes(
    [0.92, 0.57, 0.018, 0.30]
)

cbar = fig.colorbar(
    im,
    cax=cbar_ax
)

cbar.set_label(
    "Cohen's d\n(Remember − Ignore)",
    fontsize=12
)


# ============================================================
# ROI BAR PLOTS
# ============================================================

for i, band in enumerate(BANDS):

    ax = fig.add_subplot(
        2,
        3,
        i + 4
    )

    # --------------------------------------------------------
    # Find band-specific ROI rows
    # --------------------------------------------------------

    df = roi_df.copy()

    # Try to detect band column
    BAND_COL = find_column(
        df,
        [
            "band",
            "frequency",
            "freq_band"
        ]
    )

    if BAND_COL is not None:

        band_mask = (
            df[BAND_COL]
            .astype(str)
            .str.lower()
            .str.contains(band)
        )

        band_df = df[
            band_mask
        ].copy()

    else:

        # If no band column exists,
        # try detecting columns such as theta_d
        band_d_col = find_column(
            df,
            [
                f"{band}_d",
                f"{band}_cohens_d"
            ]
        )

        if band_d_col is not None:

            band_df = df.copy()

            band_df["_d"] = pd.to_numeric(
                band_df[band_d_col],
                errors="coerce"
            )

        else:

            band_df = pd.DataFrame()

    # --------------------------------------------------------
    # Extract values
    # --------------------------------------------------------

    if len(band_df) > 0:

        if "_d" not in band_df.columns:

            d_col = find_column(
                band_df,
                [
                    "d",
                    "cohens_d",
                    "effect_size"
                ]
            )

            if d_col is not None:

                band_df["_d"] = pd.to_numeric(
                    band_df[d_col],
                    errors="coerce"
                )

        roi_col = find_column(
            band_df,
            [
                "roi",
                "region",
                "region_name"
            ]
        )

        if roi_col is not None:

            roi_names = (
                band_df[roi_col]
                .astype(str)
                .tolist()
            )

        else:

            roi_names = [
                f"ROI {j+1}"
                for j in range(len(band_df))
            ]

        roi_values = (
            band_df["_d"]
            .astype(float)
            .values
        )

    else:

        # ----------------------------------------------------
        # Fallback: use known ROI results if CSV is in
        # wide format
        # ----------------------------------------------------

        possible_d_cols = [
            f"{band}_d",
            f"{band}_cohens_d",
            f"{band}_effect_size"
        ]

        d_col = None

        for col in possible_d_cols:

            if col in roi_df.columns:
                d_col = col
                break

        if d_col is not None:

            roi_col = find_column(
                roi_df,
                [
                    "roi",
                    "region",
                    "region_name"
                ]
            )

            if roi_col is not None:

                roi_names = (
                    roi_df[roi_col]
                    .astype(str)
                    .tolist()
                )

            else:

                roi_names = [
                    f"ROI {j+1}"
                    for j in range(
                        len(roi_df)
                    )
                ]

            roi_values = pd.to_numeric(
                roi_df[d_col],
                errors="coerce"
            ).values

        else:

            roi_names = []
            roi_values = []

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    valid = np.isfinite(
        roi_values
    )

    roi_values = roi_values[valid]
    roi_names = [
        name
        for name, keep in zip(
            roi_names,
            valid
        )
        if keep
    ]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    if len(roi_values) > 0:

        x = np.arange(
            len(roi_values)
        )

        ax.bar(
            x,
            roi_values
        )

        ax.axhline(
            0,
            linewidth=1
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            roi_names,
            rotation=45,
            ha="right",
            fontsize=8
        )

        ax.set_ylabel(
            "Cohen's d"
        )

        ax.set_title(
            f"{BAND_NAMES[band]} — ROI Effects",
            fontsize=13,
            fontweight="bold"
        )

        # Highlight significant ROI if FDR available
        if ROI_FDR_COL is not None:

            try:

                fdr_values = pd.to_numeric(
                    band_df[
                        ROI_FDR_COL
                    ],
                    errors="coerce"
                ).values

                fdr_values = fdr_values[
                    valid
                ]

                for j, fdr in enumerate(
                    fdr_values
                ):

                    if (
                        np.isfinite(fdr)
                        and fdr < 0.05
                    ):

                        ax.text(
                            j,
                            roi_values[j],
                            "*",
                            ha="center",
                            va=(
                                "bottom"
                                if roi_values[j] >= 0
                                else "top"
                            ),
                            fontsize=14,
                            fontweight="bold"
                        )

            except Exception:
                pass

    else:

        ax.text(
            0.5,
            0.5,
            "ROI data unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center"
        )

        ax.set_title(
            f"{BAND_NAMES[band]} — ROI Effects",
            fontsize=13,
            fontweight="bold"
        )


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "Working Memory EEG: Remember vs Ignore",
    fontsize=22,
    fontweight="bold",
    y=0.98
)

fig.text(
    0.5,
    0.945,
    "Combined Run 1 + Run 2 | Effect sizes shown as Cohen's d",
    ha="center",
    fontsize=12
)

fig.text(
    0.5,
    0.015,
    "Black circles indicate channels surviving FDR correction. "
    "Negative Cohen's d indicates lower power in Remember relative to Ignore.",
    ha="center",
    fontsize=10
)


# ============================================================
# LAYOUT
# ============================================================

plt.subplots_adjust(
    left=0.04,
    right=0.90,
    top=0.90,
    bottom=0.10,
    wspace=0.25,
    hspace=0.35
)


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL FIGURE")
print("=" * 70)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("FINAL EEG FIGURE COMPLETED")
print("=" * 70)