import os
import glob
import warnings
import numpy as np
import pandas as pd
import mne

from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ============================================================
# BRAIN PERTURBATION PROJECT
# GROUP-LEVEL EEG ANALYSIS
#
# Subjects:
#   sub-002 ... sub-024
#
# Excluded:
#   sub-001 (incomplete)
#
# Conditions:
#   Remember vs Ignore
#
# Frequency bands:
#   Theta  4-8 Hz
#   Alpha  8-13 Hz
#   Beta   13-30 Hz
#
# Analysis:
#   PSD
#   Channel-level group statistics
#   FDR correction
#   Cohen's d
#   ROI-level group statistics
#   Subject-level aggregation
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "processed"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# SUBJECTS
# ============================================================

SUBJECTS = [
    f"sub-{i:03d}"
    for i in range(2, 25)
]

print("=" * 80)
print("BRAIN PERTURBATION PROJECT")
print("GROUP-LEVEL EEG ANALYSIS")
print("=" * 80)

print("\nTarget subjects:")
print(
    ", ".join(SUBJECTS)
)

print(
    f"\nTotal target subjects: {len(SUBJECTS)}"
)

print("Excluded subject: sub-001")
print("Reason: incomplete download")


# ============================================================
# FREQUENCY BANDS
# ============================================================

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


# ============================================================
# ROI DEFINITIONS
# ============================================================

ROIS = {

    "Frontal": [
        "AF7", "AF3", "AFZ", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "FZ",
        "F2", "F4", "F6", "F8", "F9", "F10"
    ],

    "FrontoCentral": [
        "FC5", "FC3", "FC1",
        "FCZ", "FC2", "FC4"
    ],

    "Central": [
        "C5", "C3", "C1",
        "CZ", "C2", "C4"
    ],

    "CentroParietal": [
        "CP5", "CP3", "CP1",
        "CPZ", "CP2", "CP4"
    ],

    "Parietal": [
        "P7", "P5", "P3", "P1",
        "PZ", "P2", "P4", "P6", "P8"
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
        "AFZ", "FZ", "FCZ", "CZ",
        "CPZ", "PZ", "POZ", "OZ"
    ]
}


# ============================================================
# FIND FILES
# ============================================================

def find_epoch_file(subject):

    pattern = os.path.join(
        PROCESSED_DIR,
        f"{subject}_ses-01_task-WorkingMemory_run-*_clean-epo.fif"
    )

    files = sorted(
        glob.glob(pattern)
    )

    return files


# ============================================================
# PSD FUNCTION
# ============================================================

def compute_band_power(
    data,
    sfreq,
    fmin,
    fmax
):

    n_times = data.shape[-1]

    # Safe parameters for 501 samples at 250 Hz
    n_per_seg = min(
        256,
        n_times
    )

    n_fft = n_per_seg

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

    # Integrate PSD across frequency
    band_power = np.trapezoid(
        psd,
        freqs,
        axis=-1
    )

    return band_power


# ============================================================
# FIND EVENT CONDITIONS
# ============================================================

def get_condition_epochs(epochs):

    event_id = epochs.event_id

    remember_name = None
    ignore_name = None

    for name in event_id:

        lower = name.lower()

        if "remember" in lower:
            remember_name = name

        if "ignore" in lower:
            ignore_name = name

    if remember_name is None:
        raise RuntimeError(
            "Could not identify Remember event."
        )

    if ignore_name is None:
        raise RuntimeError(
            "Could not identify Ignore event."
        )

    remember = epochs[
        remember_name
    ]

    ignore = epochs[
        ignore_name
    ]

    return remember, ignore


# ============================================================
# SUBJECT STORAGE
# ============================================================

subject_results = {
    band: {
        "remember": [],
        "ignore": [],
        "subjects": []
    }
    for band in BANDS
}

successful_subjects = []
failed_subjects = []


# ============================================================
# PROCESS SUBJECTS
# ============================================================

for subject in SUBJECTS:

    print("\n")
    print("=" * 80)
    print(f"PROCESSING {subject}")
    print("=" * 80)

    files = find_epoch_file(
        subject
    )

    if len(files) == 0:

        print(
            f"[SKIP] No epoch files found for {subject}"
        )

        failed_subjects.append(
            (
                subject,
                "No epoch files"
            )
        )

        continue

    print(
        f"Found {len(files)} run(s):"
    )

    for f in files:
        print(
            "  ",
            os.path.basename(f)
        )

    # --------------------------------------------------------
    # Load all runs
    # --------------------------------------------------------

    run_epochs = []

    try:

        for file in files:

            print(
                f"\nLoading:\n{file}"
            )

            ep = mne.read_epochs(
                file,
                preload=True,
                verbose=True
            )

            run_epochs.append(
                ep
            )

        # ----------------------------------------------------
        # Combine runs
        # ----------------------------------------------------

        if len(run_epochs) == 1:

            epochs = run_epochs[0]

        else:

            epochs = mne.concatenate_epochs(
                run_epochs
            )

        print(
            "\nCombined epochs:"
        )

        print(
            f"  Epochs: {len(epochs)}"
        )

        print(
            f"  Channels: {len(epochs.ch_names)}"
        )

        print(
            f"  Sampling rate: {epochs.info['sfreq']}"
        )

        # ----------------------------------------------------
        # Get conditions
        # ----------------------------------------------------

        remember, ignore = get_condition_epochs(
            epochs
        )

        print(
            "\nCondition counts:"
        )

        print(
            f"  Remember: {len(remember)}"
        )

        print(
            f"  Ignore:   {len(ignore)}"
        )

        if len(remember) < 20 or len(ignore) < 20:

            print(
                "[WARNING] Very low trial count."
            )

        # ----------------------------------------------------
        # EEG channel selection
        # ----------------------------------------------------

        eeg_picks = mne.pick_types(
            epochs.info,
            eeg=True,
            exclude="bads"
        )

        ch_names = [
            epochs.ch_names[i]
            for i in eeg_picks
        ]

        sfreq = epochs.info["sfreq"]

        # ----------------------------------------------------
        # Process frequency bands
        # ----------------------------------------------------

        for band, (
            fmin,
            fmax
        ) in BANDS.items():

            print(
                f"\nExtracting {band.upper()}"
            )

            remember_data = (
                remember
                .get_data(
                    picks=eeg_picks
                )
            )

            ignore_data = (
                ignore
                .get_data(
                    picks=eeg_picks
                )
            )

            print(
                "Remember data:",
                remember_data.shape
            )

            print(
                "Ignore data:",
                ignore_data.shape
            )

            # ------------------------------------------------
            # PSD
            # ------------------------------------------------

            remember_power = compute_band_power(
                remember_data,
                sfreq,
                fmin,
                fmax
            )

            ignore_power = compute_band_power(
                ignore_data,
                sfreq,
                fmin,
                fmax
            )

            print(
                "Remember power:",
                remember_power.shape
            )

            print(
                "Ignore power:",
                ignore_power.shape
            )

            # ------------------------------------------------
            # Convert to log10 power
            # ------------------------------------------------

            remember_log = np.log10(
                np.maximum(
                    remember_power,
                    np.finfo(float).tiny
                )
            )

            ignore_log = np.log10(
                np.maximum(
                    ignore_power,
                    np.finfo(float).tiny
                )
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Average trials WITHIN SUBJECT
            #
            # This prevents treating hundreds of trials from
            # the same subject as independent participants.
            # ------------------------------------------------

            remember_subject = np.mean(
                remember_log,
                axis=0
            )

            ignore_subject = np.mean(
                ignore_log,
                axis=0
            )

            subject_results[
                band
            ]["remember"].append(
                remember_subject
            )

            subject_results[
                band
            ]["ignore"].append(
                ignore_subject
            )

        successful_subjects.append(
            subject
        )

        print(
            f"\n[SUCCESS] {subject}"
        )

    except Exception as e:

        print(
            f"\n[FAILED] {subject}"
        )

        print(
            "Reason:",
            repr(e)
        )

        failed_subjects.append(
            (
                subject,
                repr(e)
            )
        )


# ============================================================
# SUBJECT SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("SUBJECT PROCESSING SUMMARY")
print("=" * 80)

print(
    f"\nSuccessful subjects: "
    f"{len(successful_subjects)}"
)

print(
    f"Failed subjects: "
    f"{len(failed_subjects)}"
)

print("\nSuccessful:")

for s in successful_subjects:
    print(
        " ",
        s
    )

if failed_subjects:

    print("\nFailed:")

    for subject, reason in failed_subjects:

        print(
            f"  {subject}: {reason}"
        )


# ============================================================
# REQUIRE MINIMUM SAMPLE
# ============================================================

N_SUBJECTS = len(
    successful_subjects
)

if N_SUBJECTS < 5:

    raise RuntimeError(
        "Fewer than 5 valid subjects. "
        "Group analysis is not reliable."
    )


# ============================================================
# CHANNEL NAMES
# ============================================================

# Use the channel names from the last successfully
# processed dataset. All subjects should ideally match.

# Re-read first successful subject to establish channel list

reference_file = find_epoch_file(
    successful_subjects[0]
)[0]

reference_epochs = mne.read_epochs(
    reference_file,
    preload=False,
    verbose=False
)

eeg_picks_reference = mne.pick_types(
    reference_epochs.info,
    eeg=True,
    exclude="bads"
)

group_ch_names = [
    reference_epochs.ch_names[i]
    for i in eeg_picks_reference
]

print(
    "\nGroup EEG channels:",
    len(group_ch_names)
)


# ============================================================
# GROUP STATISTICS
# ============================================================

group_results = {}


for band in BANDS:

    print("\n")
    print("=" * 80)
    print(
        f"GROUP-LEVEL {band.upper()} ANALYSIS"
    )
    print("=" * 80)

    remember = np.asarray(
        subject_results[
            band
        ]["remember"]
    )

    ignore = np.asarray(
        subject_results[
            band
        ]["ignore"]
    )

    print(
        "Remember:",
        remember.shape
    )

    print(
        "Ignore:",
        ignore.shape
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    difference = (
        remember -
        ignore
    )

    # --------------------------------------------------------
    # Paired t-test across SUBJECTS
    # --------------------------------------------------------

    t_values = np.zeros(
        difference.shape[1]
    )

    p_values = np.ones(
        difference.shape[1]
    )

    for ch in range(
        difference.shape[1]
    ):

        t, p = stats.ttest_rel(
            remember[:, ch],
            ignore[:, ch],
            nan_policy="omit"
        )

        t_values[ch] = t
        p_values[ch] = p

    # --------------------------------------------------------
    # FDR correction
    # --------------------------------------------------------

    reject, p_fdr, _, _ = (
        multipletests(
            p_values,
            alpha=0.05,
            method="fdr_bh"
        )
    )

    # --------------------------------------------------------
    # Cohen's d for paired samples
    #
    # dz = mean(diff) / SD(diff)
    # --------------------------------------------------------

    mean_difference = np.mean(
        difference,
        axis=0
    )

    sd_difference = np.std(
        difference,
        axis=0,
        ddof=1
    )

    cohens_d = np.divide(
        mean_difference,
        sd_difference,
        out=np.zeros_like(
            mean_difference
        ),
        where=sd_difference != 0
    )

    # --------------------------------------------------------
    # Build dataframe
    # --------------------------------------------------------

    df = pd.DataFrame({

        "channel": group_ch_names,

        "mean_remember":
            np.mean(
                remember,
                axis=0
            ),

        "mean_ignore":
            np.mean(
                ignore,
                axis=0
            ),

        "difference_remember_minus_ignore":
            mean_difference,

        "t":
            t_values,

        "p":
            p_values,

        "cohens_d":
            cohens_d,

        "p_fdr":
            p_fdr,

        "significant_fdr":
            reject

    })

    # --------------------------------------------------------
    # Sort by effect size
    # --------------------------------------------------------

    df_sorted = df.copy()

    df_sorted[
        "abs_d"
    ] = np.abs(
        df_sorted["cohens_d"]
    )

    df_sorted = df_sorted.sort_values(
        "abs_d",
        ascending=False
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nSignificant channels: "
        f"{reject.sum()}/{len(reject)}"
    )

    print(
        "\nTop 15 effects:"
    )

    print(
        df_sorted[
            [
                "channel",
                "cohens_d",
                "p",
                "p_fdr",
                "significant_fdr"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = os.path.join(
        RESULTS_DIR,
        f"group_{band}_channel_statistics.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        output_file
    )

    group_results[
        band
    ] = {
        "remember": remember,
        "ignore": ignore,
        "difference": difference,
        "t": t_values,
        "p": p_values,
        "p_fdr": p_fdr,
        "d": cohens_d,
        "reject": reject
    }


# ============================================================
# ROI-LEVEL GROUP ANALYSIS
# ============================================================

print("\n")
print("=" * 80)
print("ROI-LEVEL GROUP ANALYSIS")
print("=" * 80)


roi_rows = []


for band in BANDS:

    print("\n")
    print("-" * 70)
    print(
        f"{band.upper()} ROI ANALYSIS"
    )
    print("-" * 70)

    remember = group_results[
        band
    ]["remember"]

    ignore = group_results[
        band
    ]["ignore"]

    for roi_name, roi_channels in ROIS.items():

        indices = [
            i
            for i, ch in enumerate(
                group_ch_names
            )
            if ch in roi_channels
        ]

        if len(indices) == 0:

            print(
                f"{roi_name}: NO MATCHING CHANNELS"
            )

            continue

        # ----------------------------------------------------
        # Average channels within ROI
        # ----------------------------------------------------

        remember_roi = np.mean(
            remember[:, indices],
            axis=1
        )

        ignore_roi = np.mean(
            ignore[:, indices],
            axis=1
        )

        difference_roi = (
            remember_roi -
            ignore_roi
        )

        # ----------------------------------------------------
        # Paired t-test
        # ----------------------------------------------------

        t_value, p_value = (
            stats.ttest_rel(
                remember_roi,
                ignore_roi
            )
        )

        # ----------------------------------------------------
        # Cohen's dz
        # ----------------------------------------------------

        sd_diff = np.std(
            difference_roi,
            ddof=1
        )

        if sd_diff > 0:

            d_value = (
                np.mean(
                    difference_roi
                )
                /
                sd_diff
            )

        else:

            d_value = 0.0

        roi_rows.append({

            "band":
                band,

            "ROI":
                roi_name,

            "n_channels":
                len(indices),

            "channels":
                ", ".join(
                    [
                        group_ch_names[i]
                        for i in indices
                    ]
                ),

            "n_subjects":
                N_SUBJECTS,

            "mean_remember":
                np.mean(
                    remember_roi
                ),

            "mean_ignore":
                np.mean(
                    ignore_roi
                ),

            "difference_remember_minus_ignore":
                np.mean(
                    difference_roi
                ),

            "t":
                t_value,

            "p":
                p_value,

            "cohens_d":
                d_value
        })


# ============================================================
# ROI FDR
# ============================================================

roi_df = pd.DataFrame(
    roi_rows
)

roi_df[
    "p_fdr"
] = np.nan

roi_df[
    "significant_fdr"
] = False


for band in BANDS:

    mask = (
        roi_df["band"]
        == band
    )

    pvals = roi_df.loc[
        mask,
        "p"
    ].values

    if len(pvals) > 0:

        reject_roi, p_fdr_roi, _, _ = (
            multipletests(
                pvals,
                alpha=0.05,
                method="fdr_bh"
            )
        )

        roi_df.loc[
            mask,
            "p_fdr"
        ] = p_fdr_roi

        roi_df.loc[
            mask,
            "significant_fdr"
        ] = reject_roi


# ============================================================
# PRINT ROI RESULTS
# ============================================================

for band in BANDS:

    print("\n")
    print(
        band.upper()
    )

    temp = roi_df[
        roi_df["band"]
        == band
    ].copy()

    temp = temp.sort_values(
        "cohens_d",
        key=np.abs,
        ascending=False
    )

    print(
        temp[
            [
                "ROI",
                "n_channels",
                "cohens_d",
                "p",
                "p_fdr",
                "significant_fdr"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAVE ROI RESULTS
# ============================================================

roi_output = os.path.join(
    RESULTS_DIR,
    "group_ROI_statistics.csv"
)

roi_df.to_csv(
    roi_output,
    index=False
)

print("\n")
print(
    "ROI results saved:"
)

print(
    roi_output
)


# ============================================================
# SAVE GROUP NPZ
# ============================================================

npz_output = os.path.join(
    PROCESSED_DIR,
    "group_level_eeg_statistics.npz"
)

np.savez(
    npz_output,

    subjects=np.array(
        successful_subjects
    ),

    ch_names=np.array(
        group_ch_names
    ),

    theta_t=
        group_results[
            "theta"
        ]["t"],

    theta_p=
        group_results[
            "theta"
        ]["p"],

    theta_p_fdr=
        group_results[
            "theta"
        ]["p_fdr"],

    theta_d=
        group_results[
            "theta"
        ]["d"],

    theta_reject=
        group_results[
            "theta"
        ]["reject"],

    alpha_t=
        group_results[
            "alpha"
        ]["t"],

    alpha_p=
        group_results[
            "alpha"
        ]["p"],

    alpha_p_fdr=
        group_results[
            "alpha"
        ]["p_fdr"],

    alpha_d=
        group_results[
            "alpha"
        ]["d"],

    alpha_reject=
        group_results[
            "alpha"
        ]["reject"],

    beta_t=
        group_results[
            "beta"
        ]["t"],

    beta_p=
        group_results[
            "beta"
        ]["p"],

    beta_p_fdr=
        group_results[
            "beta"
        ]["p_fdr"],

    beta_d=
        group_results[
            "beta"
        ]["d"],

    beta_reject=
        group_results[
            "beta"
        ]["reject"]
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL GROUP-LEVEL SUMMARY")
print("=" * 80)

for band in BANDS:

    result = group_results[
        band
    ]

    n_sig = int(
        np.sum(
            result["reject"]
        )
    )

    d = result["d"]

    strongest_idx = np.argmax(
        np.abs(d)
    )

    print(
        f"\n{band.upper()}"
    )

    print(
        f"  Subjects: {N_SUBJECTS}"
    )

    print(
        f"  Significant channels: "
        f"{n_sig}/{len(d)}"
    )

    print(
        f"  Strongest channel: "
        f"{group_ch_names[strongest_idx]}"
    )

    print(
        f"  Cohen's d: "
        f"{d[strongest_idx]:.4f}"
    )

    print(
        f"  p-value: "
        f"{result['p'][strongest_idx]:.6f}"
    )

    print(
        f"  FDR: "
        f"{result['p_fdr'][strongest_idx]:.6f}"
    )


print("\n")
print("=" * 80)
print("GROUP-LEVEL ANALYSIS COMPLETED")
print("=" * 80)

print("\nGenerated files:")

print(
    "Channel statistics:"
)

for band in BANDS:

    print(
        os.path.join(
            RESULTS_DIR,
            f"group_{band}_channel_statistics.csv"
        )
    )

print(
    "\nROI statistics:"
)

print(
    roi_output
)

print(
    "\nGroup NPZ:"
)

print(
    npz_output
)

print("\n[DONE]")