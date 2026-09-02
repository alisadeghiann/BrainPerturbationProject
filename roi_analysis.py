# ============================================================
# ROI ANALYSIS
# Brain Perturbation Project
# Subject: sub-009
# Run 1 + Run 2 combined
# ============================================================

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

NPZ_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "sub-009_combined_run1_run2_statistics.npz"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "sub-009_ROI_analysis.csv"
)

# ------------------------------------------------------------
# LOAD RESULTS
# ------------------------------------------------------------

print("=" * 70)
print("ROI ANALYSIS")
print("=" * 70)

print("\nLoading statistical results:")
print(NPZ_FILE)

data = np.load(NPZ_FILE, allow_pickle=True)

print("\nAvailable keys:")
for key in data.files:
    print(" ", key)

# ------------------------------------------------------------
# CHANNEL NAMES
# ------------------------------------------------------------

ch_names = data["ch_names"]

# Convert numpy strings to normal Python strings
ch_names = [str(ch).upper() for ch in ch_names]

print("\nTotal channels:", len(ch_names))

# ------------------------------------------------------------
# ROI DEFINITIONS
# ------------------------------------------------------------

# These are intentionally broad anatomical/electrode regions.
# Midline electrodes are assigned separately to avoid ambiguity.

ROI_CHANNELS = {

    "Frontal": [
        "FP1", "FP2",
        "AF7", "AF8",
        "AF3", "AF4",
        "F7", "F8",
        "F5", "F6",
        "F3", "F4",
        "F1", "F2",
        "F9", "F10"
    ],

    "FrontoCentral": [
        "FC5", "FC6",
        "FC3", "FC4",
        "FC1", "FC2"
    ],

    "Central": [
        "C5", "C6",
        "C3", "C4",
        "C1", "C2"
    ],

    "CentroParietal": [
        "CP5", "CP6",
        "CP3", "CP4",
        "CP1", "CP2"
    ],

    "Parietal": [
        "P7", "P8",
        "P5", "P6",
        "P3", "P4",
        "P1", "P2"
    ],

    "ParietoOccipital": [
        "PO7", "PO8",
        "PO3", "PO4",
        "PO9", "PO10"
    ],

    "Occipital": [
        "O1", "O2"
    ],

    "Temporal": [
        "T7", "T8",
        "TP7", "TP8",
        "TP9", "TP10",
        "FT7", "FT8",
        "FT9", "FT10"
    ],

    "Midline": [
        "FPZ",
        "AFZ",
        "FZ",
        "FCZ",
        "CZ",
        "CPZ",
        "PZ",
        "POZ",
        "OZ"
    ]
}

# ------------------------------------------------------------
# CHECK CHANNEL COVERAGE
# ------------------------------------------------------------

channel_to_roi = {}

for roi, channels in ROI_CHANNELS.items():

    for ch in channels:

        if ch in channel_to_roi:

            print(
                f"WARNING: {ch} already assigned to "
                f"{channel_to_roi[ch]}"
            )

        channel_to_roi[ch] = roi

# ------------------------------------------------------------
# DATA EXTRACTION FUNCTION
# ------------------------------------------------------------

def analyze_band(band):

    print("\n" + "-" * 70)
    print(f"{band.upper()} ROI ANALYSIS")
    print("-" * 70)

    band_key = band.lower()

    t_values = data[f"{band_key}_t"]
    p_values = data[f"{band_key}_p"]
    fdr_values = data[f"{band_key}_p_fdr"]
    d_values = data[f"{band_key}_d"]
    reject_values = data[f"{band_key}_reject"]

    rows = []

    for i, ch in enumerate(ch_names):

        roi = channel_to_roi.get(ch, "Unassigned")

        rows.append({
            "channel": ch,
            "roi": roi,
            "t": float(t_values[i]),
            "p": float(p_values[i]),
            "p_fdr": float(fdr_values[i]),
            "cohens_d": float(d_values[i]),
            "significant_fdr": bool(reject_values[i])
        })

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # ROI SUMMARY
    # --------------------------------------------------------

    roi_rows = []

    for roi in ROI_CHANNELS.keys():

        roi_df = df[df["roi"] == roi]

        if len(roi_df) == 0:
            continue

        significant_df = roi_df[
            roi_df["significant_fdr"] == True
        ]

        n_channels = len(roi_df)
        n_significant = len(significant_df)

        mean_d_all = roi_df["cohens_d"].mean()

        # Mean effect among significant channels only
        if len(significant_df) > 0:
            mean_d_sig = significant_df["cohens_d"].mean()
            median_d_sig = significant_df["cohens_d"].median()
            strongest_channel = significant_df.loc[
                significant_df["cohens_d"].abs().idxmax(),
                "channel"
            ]
            strongest_d = significant_df.loc[
                significant_df["cohens_d"].abs().idxmax(),
                "cohens_d"
            ]
        else:
            mean_d_sig = np.nan
            median_d_sig = np.nan
            strongest_channel = "None"
            strongest_d = np.nan

        roi_rows.append({
            "band": band.upper(),
            "ROI": roi,
            "n_channels": n_channels,
            "n_significant_FDR": n_significant,
            "percent_significant": (
                100 * n_significant / n_channels
            ),
            "mean_cohens_d_all": mean_d_all,
            "mean_cohens_d_significant": mean_d_sig,
            "median_cohens_d_significant": median_d_sig,
            "strongest_channel": strongest_channel,
            "strongest_cohens_d": strongest_d
        })

    roi_summary = pd.DataFrame(roi_rows)

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nROI SUMMARY:\n")

    for _, row in roi_summary.iterrows():

        print(
            f"{row['ROI']:20s} "
            f"Significant: "
            f"{int(row['n_significant_FDR']):2d}/"
            f"{int(row['n_channels']):2d} "
            f"({row['percent_significant']:5.1f}%)"
        )

        if row["n_significant_FDR"] > 0:

            print(
                f"   Mean d = "
                f"{row['mean_cohens_d_significant']:.3f}"
            )

            print(
                f"   Strongest = "
                f"{row['strongest_channel']} "
                f"(d={row['strongest_cohens_d']:.3f})"
            )

    return df, roi_summary


# ------------------------------------------------------------
# ANALYZE ALL BANDS
# ------------------------------------------------------------

all_roi_results = []

band_channel_results = {}

for band in ["theta", "alpha", "beta"]:

    channel_df, roi_df = analyze_band(band)

    band_channel_results[band] = channel_df

    all_roi_results.append(roi_df)


# ------------------------------------------------------------
# COMBINE RESULTS
# ------------------------------------------------------------

final_roi_df = pd.concat(
    all_roi_results,
    ignore_index=True
)

# ------------------------------------------------------------
# SAVE ROI CSV
# ------------------------------------------------------------

final_roi_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n" + "=" * 70)
print("ROI RESULTS SAVED")
print("=" * 70)

print(OUTPUT_CSV)


# ------------------------------------------------------------
# SIGNIFICANT CHANNEL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SIGNIFICANT CHANNEL DISTRIBUTION")
print("=" * 70)

for band in ["theta", "alpha", "beta"]:

    df = band_channel_results[band]

    significant = df[
        df["significant_fdr"] == True
    ]

    print("\n" + band.upper())

    if len(significant) == 0:

        print("No significant channels.")

        continue

    counts = (
        significant["roi"]
        .value_counts()
        .sort_values(ascending=False)
    )

    for roi, count in counts.items():

        print(
            f"  {roi:20s}: {count} significant channels"
        )


# ------------------------------------------------------------
# STRONGEST ROI BY BAND
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STRONGEST ROIs")
print("=" * 70)

for band in ["theta", "alpha", "beta"]:

    df = final_roi_df[
        (final_roi_df["band"] == band.upper()) &
        (final_roi_df["n_significant_FDR"] > 0)
    ].copy()

    print("\n" + band.upper())

    if len(df) == 0:

        print("No ROI survived FDR.")

        continue

    # Rank primarily by number of significant channels,
    # secondarily by absolute mean effect size.

    df["abs_mean_d"] = (
        df["mean_cohens_d_significant"].abs()
    )

    df = df.sort_values(
        by=["n_significant_FDR", "abs_mean_d"],
        ascending=[False, False]
    )

    top = df.iloc[0]

    print(
        f"Strongest ROI: {top['ROI']}"
    )

    print(
        f"Significant channels: "
        f"{int(top['n_significant_FDR'])}"
    )

    print(
        f"Mean Cohen's d: "
        f"{top['mean_cohens_d_significant']:.3f}"
    )

    print(
        f"Strongest channel: "
        f"{top['strongest_channel']}"
    )

    print(
        f"Strongest channel d: "
        f"{top['strongest_cohens_d']:.3f}"
    )


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL ROI ANALYSIS SUMMARY")
print("=" * 70)

for band in ["THETA", "ALPHA", "BETA"]:

    df = final_roi_df[
        final_roi_df["band"] == band
    ]

    total_significant = int(
        df["n_significant_FDR"].sum()
    )

    print(
        f"\n{band}:"
    )

    print(
        f"  Total significant channels: "
        f"{total_significant}"
    )

    if total_significant > 0:

        significant_rois = df[
            df["n_significant_FDR"] > 0
        ].sort_values(
            "n_significant_FDR",
            ascending=False
        )

        print("  Significant ROIs:")

        for _, row in significant_rois.iterrows():

            print(
                f"    {row['ROI']:20s} "
                f"{int(row['n_significant_FDR'])} channels"
            )

print("\n" + "=" * 70)
print("ROI ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput:")
print(OUTPUT_CSV)

print("\n[DONE]")