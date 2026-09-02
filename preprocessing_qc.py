# ============================================================
# Brain Perturbation Project
# preprocessing_qc.py
#
# EEG preprocessing + artifact QC
#
# INPUT:
#   data/sub-*/ses-01/eeg/*.set
#   data/sub-*/ses-01/eeg/*.fdt
#
# OUTPUT:
#   qc/preprocessed/
#   qc/preprocessing_qc/
#
# IMPORTANT:
#   Original EEG files are NEVER modified.
# ============================================================

import os
import glob
import warnings
import traceback

import h5py
import numpy as np
import pandas as pd
import mne

from scipy import signal

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

EVENT_FILE = os.path.join(
    PROJECT_ROOT,
    "qc",
    "events",
    "ALL_EVENTS_83_RUNS.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "qc",
    "preprocessed"
)

QC_DIR = os.path.join(
    PROJECT_ROOT,
    "qc",
    "preprocessing_qc"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(QC_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# EEG preprocessing
LOW_CUT = 1.0
HIGH_CUT = 40.0
NOTCH_FREQ = 50.0

# ICA
ICA_RANDOM_STATE = 97
ICA_MAX_COMPONENTS = 30

# Decimation used ONLY for ICA fitting
# The actual EEG remains at original sampling rate.
ICA_DECIM = 4

# Channel QC thresholds
FLAT_STD_THRESHOLD = 1e-8

# Robust channel outlier threshold
CHANNEL_STD_Z = 6.0

# We do NOT automatically delete suspicious channels.
AUTO_DROP_BAD_CHANNELS = False

# Save FIF files
SAVE_PREPROCESSED = True


# ============================================================
# HELPERS
# ============================================================

def decode_hdf_value(f, obj):
    """
    Robustly decode MATLAB v7.3 / HDF5 values.
    """

    try:

        # HDF reference
        if isinstance(obj, h5py.Reference):

            if not obj:
                return ""

            target = f[obj]

            if isinstance(target, h5py.Dataset):

                arr = target[()]

                # uint16 -> string
                if np.issubdtype(arr.dtype, np.integer):

                    arr = np.asarray(arr).flatten()

                    try:
                        return "".join(
                            chr(int(x))
                            for x in arr
                            if int(x) != 0
                        )
                    except Exception:
                        return str(arr)

                # scalar
                if arr.size == 1:
                    return str(arr.flatten()[0])

                return str(arr)

        # numpy scalar
        if isinstance(obj, np.ndarray):

            if obj.size == 1:
                return decode_hdf_value(f, obj.flatten()[0])

            if np.issubdtype(obj.dtype, np.integer):

                arr = obj.flatten()

                try:
                    return "".join(
                        chr(int(x))
                        for x in arr
                        if int(x) != 0
                    )
                except Exception:
                    return str(arr)

        return str(obj)

    except Exception:
        return ""


def read_string_dataset(f, path):
    """
    Read MATLAB/HDF5 string dataset.
    """

    try:

        ds = f[path]
        arr = ds[()]

        # Reference array
        if arr.dtype == object:

            values = []

            for x in arr.flatten():

                values.append(
                    decode_hdf_value(f, x)
                )

            return values

        # uint16 character array
        if np.issubdtype(arr.dtype, np.integer):

            arr = np.asarray(arr).flatten()

            text = "".join(
                chr(int(x))
                for x in arr
                if int(x) != 0
            )

            return [text]

        return [str(x) for x in np.asarray(arr).flatten()]

    except Exception:

        return []


def extract_channel_labels(set_file, n_channels):
    """
    Extract EEG channel labels from chanlocs/labels.
    """

    labels = []

    try:

        with h5py.File(set_file, "r") as f:

            if "chanlocs/labels" not in f:

                return [
                    f"EEG{idx + 1:03d}"
                    for idx in range(n_channels)
                ]

            ds = f["chanlocs/labels"]
            refs = ds[()]

            for ref in refs.flatten():

                label = decode_hdf_value(
                    f,
                    ref
                )

                label = str(label).strip()

                if not label:
                    label = f"EEG{len(labels)+1:03d}"

                labels.append(label)

    except Exception as e:

        print(
            "WARNING: Could not read channel labels:",
            e
        )

    # Safety
    if len(labels) != n_channels:

        labels = [
            f"EEG{idx + 1:03d}"
            for idx in range(n_channels)
        ]

    # Ensure unique names
    seen = {}

    clean_labels = []

    for label in labels:

        if label not in seen:

            seen[label] = 0
            clean_labels.append(label)

        else:

            seen[label] += 1

            clean_labels.append(
                f"{label}_{seen[label]}"
            )

    return clean_labels


def get_metadata(set_file):
    """
    Read core EEG metadata from MATLAB v7.3 SET.
    """

    with h5py.File(set_file, "r") as f:

        metadata = {}

        for field in [
            "nbchan",
            "pnts",
            "trials",
            "srate",
            "xmin",
            "xmax"
        ]:

            if field in f:

                try:

                    value = np.array(
                        f[field][()]
                    ).squeeze()

                    metadata[field] = float(value)

                except Exception:

                    metadata[field] = np.nan

            else:

                metadata[field] = np.nan

    return metadata


def read_eeg_from_fdt(set_file, metadata):
    """
    Read actual EEG samples from the corresponding FDT file.

    EEGLAB stores the actual continuous data in .fdt.
    """

    fdt_file = os.path.splitext(set_file)[0] + ".fdt"

    if not os.path.exists(fdt_file):

        raise FileNotFoundError(
            f"FDT file missing:\n{fdt_file}"
        )

    n_channels = int(metadata["nbchan"])
    n_samples = int(metadata["pnts"])

    expected_values = (
        n_channels *
        n_samples
    )

    expected_bytes = (
        expected_values *
        np.dtype("<f4").itemsize
    )

    actual_bytes = os.path.getsize(
        fdt_file
    )

    if actual_bytes != expected_bytes:

        raise ValueError(
            f"FDT size mismatch.\n"
            f"Expected bytes: {expected_bytes}\n"
            f"Actual bytes:   {actual_bytes}"
        )

    data = np.fromfile(
        fdt_file,
        dtype="<f4"
    )

    data = data.reshape(
        (n_channels, n_samples),
        order="C"
    )

    return data, fdt_file


def robust_channel_qc(data, labels):
    """
    Detect flat and extreme channels.

    IMPORTANT:
    No channels are deleted here.
    """

    channel_std = np.std(
        data,
        axis=1
    )

    channel_ptp = (
        np.percentile(data, 99, axis=1)
        -
        np.percentile(data, 1, axis=1)
    )

    median_std = np.median(
        channel_std
    )

    mad_std = np.median(
        np.abs(
            channel_std -
            median_std
        )
    )

    if mad_std == 0:

        robust_z = np.zeros_like(
            channel_std
        )

    else:

        robust_z = (
            0.6745 *
            (channel_std - median_std)
            /
            mad_std
        )

    flat_idx = np.where(
        channel_std <= FLAT_STD_THRESHOLD
    )[0]

    extreme_idx = np.where(
        np.abs(robust_z) >= CHANNEL_STD_Z
    )[0]

    rows = []

    for idx, label in enumerate(labels):

        rows.append({

            "channel_index": idx,

            "channel": label,

            "std": float(
                channel_std[idx]
            ),

            "peak_to_peak": float(
                channel_ptp[idx]
            ),

            "robust_z_std": float(
                robust_z[idx]
            ),

            "is_flat":
                idx in flat_idx,

            "is_extreme":
                idx in extreme_idx

        })

    return (
        pd.DataFrame(rows),
        [labels[i] for i in flat_idx],
        [labels[i] for i in extreme_idx]
    )


def detect_eog_channels(labels):
    """
    Detect EOG channels using common EEGLAB labels.
    """

    eog_keywords = [
        "EOG",
        "VEOG",
        "HEOG",
        "REOG",
        "LEOG",
        "REYE",
        "LEYE",
        "EYE"
    ]

    eog = []

    for label in labels:

        upper = label.upper()

        if any(
            keyword in upper
            for keyword in eog_keywords
        ):

            eog.append(label)

    return eog


def detect_eeg_channels(labels, eog_labels):
    return [
        label
        for label in labels
        if label not in eog_labels
    ]


def create_raw(data, labels, srate):
    """
    Create MNE RawArray.

    We intentionally DO NOT normalize the signal here.
    """

    ch_types = []

    eog_labels = detect_eog_channels(
        labels
    )

    for label in labels:

        if label in eog_labels:

            ch_types.append("eog")

        else:

            ch_types.append("eeg")

    info = mne.create_info(
        ch_names=labels,
        sfreq=srate,
        ch_types=ch_types
    )

    raw = mne.io.RawArray(
        data,
        info,
        verbose=False
    )

    return raw, eog_labels


def preprocess_raw(raw):
    """
    Filtering only.
    """

    print(
        f"Filtering {LOW_CUT}-{HIGH_CUT} Hz..."
    )

    raw.filter(
        l_freq=LOW_CUT,
        h_freq=HIGH_CUT,
        picks="eeg",
        method="fir",
        phase="zero",
        verbose=False
    )

    # Apply notch only if Nyquist allows it
    if raw.info["sfreq"] / 2 > NOTCH_FREQ:

        print(
            f"Notch filter: {NOTCH_FREQ} Hz"
        )

        raw.notch_filter(
            freqs=[NOTCH_FREQ],
            picks="eeg",
            method="fir",
            phase="zero",
            verbose=False
        )

    return raw


def run_ica(raw, eog_labels):
    """
    ICA artifact correction.

    ICA is fitted on decimated data to reduce computation.
    """

    print(
        "Running ICA..."
    )

    eeg_picks = mne.pick_types(
        raw.info,
        eeg=True,
        eog=False,
        exclude="bads"
    )

    if len(eeg_picks) < 10:

        print(
            "Too few EEG channels for ICA."
        )

        return raw, None, []

    n_components = min(
        ICA_MAX_COMPONENTS,
        len(eeg_picks)
    )

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        random_state=ICA_RANDOM_STATE,
        max_iter="auto"
    )

    ica.fit(
        raw,
        picks=eeg_picks,
        decim=ICA_DECIM,
        reject=None,
        verbose=False
    )

    excluded = []

    # --------------------------------------------------------
    # EOG COMPONENT DETECTION
    # --------------------------------------------------------

    if len(eog_labels) > 0:

        print(
            "Detecting EOG-related ICA components..."
        )

        for eog in eog_labels:

            try:

                inds, scores = ica.find_bads_eog(
                    raw,
                    ch_name=eog,
                    verbose=False
                )

                # Conservative threshold:
                # only strong candidates
                for idx, score in zip(
                    inds,
                    scores[inds]
                ):

                    if abs(score) >= 0.30:

                        excluded.append(
                            int(idx)
                        )

            except Exception as e:

                print(
                    f"Could not detect EOG "
                    f"for {eog}: {e}"
                )

    excluded = sorted(
        list(set(excluded))
    )

    ica.exclude = excluded

    print(
        f"ICA components: {ica.n_components_}"
    )

    print(
        f"Components marked for exclusion: "
        f"{excluded}"
    )

    # --------------------------------------------------------
    # APPLY ICA
    # --------------------------------------------------------

    if len(excluded) > 0:

        raw_clean = raw.copy()

        ica.apply(
            raw_clean,
            verbose=False
        )

    else:

        raw_clean = raw.copy()

    return (
        raw_clean,
        ica,
        excluded
    )


# ============================================================
# LOAD EVENT FILE
# ============================================================

print("=" * 80)
print("EEG PREPROCESSING + ARTIFACT QC")
print("=" * 80)

if not os.path.exists(EVENT_FILE):

    raise FileNotFoundError(
        f"Event file not found:\n{EVENT_FILE}"
    )

events_df = pd.read_csv(
    EVENT_FILE
)

print(
    f"Events loaded: {len(events_df)}"
)

print(
    f"Event columns: "
    f"{list(events_df.columns)}"
)


# ============================================================
# FIND EEG FILES
# ============================================================

set_files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "sub-*",
            "ses-*",
            "eeg",
            "*_eeg.set"
        )
    )
)

print(
    f"\nEEG SET files found: "
    f"{len(set_files)}"
)

if len(set_files) == 0:

    raise FileNotFoundError(
        "No EEG .set files found."
    )


# ============================================================
# OUTPUT REPORTS
# ============================================================

run_summary = []

channel_summary = []

ica_summary = []


# ============================================================
# PROCESS ALL RUNS
# ============================================================

for file_idx, set_file in enumerate(
    set_files,
    start=1
):

    print("\n")
    print("=" * 80)
    print(
        f"PROCESSING {file_idx}/{len(set_files)}"
    )
    print("=" * 80)

    filename = os.path.basename(
        set_file
    )

    print(filename)

    try:

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        metadata = get_metadata(
            set_file
        )

        n_channels = int(
            metadata["nbchan"]
        )

        n_samples = int(
            metadata["pnts"]
        )

        srate = float(
            metadata["srate"]
        )

        print(
            f"Channels: {n_channels}"
        )

        print(
            f"Samples: {n_samples}"
        )

        print(
            f"Sampling rate: {srate}"
        )

        # ----------------------------------------------------
        # CHANNEL LABELS
        # ----------------------------------------------------

        labels = extract_channel_labels(
            set_file,
            n_channels
        )

        eog_labels = detect_eog_channels(
            labels
        )

        print(
            f"EOG channels: {eog_labels}"
        )

        # ----------------------------------------------------
        # READ FDT
        # ----------------------------------------------------

        data, fdt_file = read_eeg_from_fdt(
            set_file,
            metadata
        )

        print(
            f"Data shape: {data.shape}"
        )

        print(
            f"Raw STD: {np.std(data):.6f}"
        )

        print(
            f"Raw min: {np.min(data):.6f}"
        )

        print(
            f"Raw max: {np.max(data):.6f}"
        )

        # ----------------------------------------------------
        # SIGNAL SCALE FLAG
        # ----------------------------------------------------

        global_std = float(
            np.std(data)
        )

        if global_std < 1:

            scale_flag = "VERY_LOW"

        elif global_std > 100:

            scale_flag = "VERY_HIGH"

        else:

            scale_flag = "NORMAL_RANGE"

        # IMPORTANT:
        # We do NOT rescale.
        #
        # The goal is to determine later whether
        # low/high scale is caused by acquisition
        # units, reference, amplifier settings, etc.

        # ----------------------------------------------------
        # CHANNEL QC
        # ----------------------------------------------------

        channel_df, flat_channels, extreme_channels = (
            robust_channel_qc(
                data,
                labels
            )
        )

        print(
            f"Flat channels: "
            f"{flat_channels}"
        )

        print(
            f"Extreme channels: "
            f"{extreme_channels}"
        )

        # ----------------------------------------------------
        # CREATE MNE RAW
        # ----------------------------------------------------

        raw, eog_labels = create_raw(
            data,
            labels,
            srate
        )

        # ----------------------------------------------------
        # MARK BAD CHANNELS
        # ----------------------------------------------------

        bad_channels = list(
            set(
                flat_channels +
                extreme_channels
            )
        )

        # We flag them but DON'T automatically delete them.
        #
        # This is important for the Brain Perturbation project.

        raw.info["bads"] = bad_channels

        print(
            f"Flagged bad channels: "
            f"{bad_channels}"
        )

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        raw_filtered = preprocess_raw(
            raw.copy()
        )

        # ----------------------------------------------------
        # ICA
        # ----------------------------------------------------

        raw_clean, ica, excluded_components = run_ica(
            raw_filtered,
            eog_labels
        )

        # ----------------------------------------------------
        # SAVE ICA INFORMATION
        # ----------------------------------------------------

        subject = os.path.basename(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        set_file
                    )
                )
            )
        )

        run_name = os.path.splitext(
            filename
        )[0]

        if ica is not None:

            ica_file = os.path.join(
                QC_DIR,
                f"{run_name}-ica.fif"
            )

            ica.save(
                ica_file,
                overwrite=True
            )

        else:

            ica_file = ""

        # ----------------------------------------------------
        # SAVE PREPROCESSED EEG
        # ----------------------------------------------------

        if SAVE_PREPROCESSED:

            output_file = os.path.join(
                OUTPUT_DIR,
                f"{run_name}_preprocessed_raw.fif"
            )

            raw_clean.save(
                output_file,
                overwrite=True
            )

        else:

            output_file = ""

        # ----------------------------------------------------
        # RUN SUMMARY
        # ----------------------------------------------------

        run_summary.append({

            "subject": subject,

            "file": filename,

            "fdt_file": os.path.basename(
                fdt_file
            ),

            "channels": n_channels,

            "samples": n_samples,

            "sampling_rate": srate,

            "duration_sec":
                n_samples / srate,

            "global_std": global_std,

            "scale_flag": scale_flag,

            "flat_channels":
                len(flat_channels),

            "extreme_channels":
                len(extreme_channels),

            "flagged_bad_channels":
                len(bad_channels),

            "eog_channels":
                len(eog_labels),

            "ica_components":
                (
                    int(ica.n_components_)
                    if ica is not None
                    else 0
                ),

            "ica_excluded":
                len(excluded_components),

            "excluded_components":
                ",".join(
                    map(
                        str,
                        excluded_components
                    )
                ),

            "status": "OK",

            "preprocessed_file":
                output_file,

            "ica_file":
                ica_file

        })

        # ----------------------------------------------------
        # CHANNEL REPORT
        # ----------------------------------------------------

        for _, row in channel_df.iterrows():

            row_dict = row.to_dict()

            row_dict.update({

                "subject": subject,

                "file": filename,

                "sampling_rate": srate

            })

            channel_summary.append(
                row_dict
            )

        # ----------------------------------------------------
        # ICA REPORT
        # ----------------------------------------------------

        ica_summary.append({

            "subject": subject,

            "file": filename,

            "ica_components":
                (
                    int(ica.n_components_)
                    if ica is not None
                    else 0
                ),

            "excluded_components":
                ",".join(
                    map(
                        str,
                        excluded_components
                    )
                ),

            "n_excluded":
                len(excluded_components),

            "eog_channels":
                ",".join(
                    eog_labels
                ),

            "status": "OK"

        })

        print(
            "\nSTATUS: OK"
        )

    except Exception as e:

        print(
            "\nERROR:"
        )

        print(e)

        traceback.print_exc()

        run_summary.append({

            "subject":
                "UNKNOWN",

            "file":
                filename,

            "channels":
                np.nan,

            "samples":
                np.nan,

            "sampling_rate":
                np.nan,

            "duration_sec":
                np.nan,

            "global_std":
                np.nan,

            "scale_flag":
                "ERROR",

            "flat_channels":
                np.nan,

            "extreme_channels":
                np.nan,

            "flagged_bad_channels":
                np.nan,

            "eog_channels":
                np.nan,

            "ica_components":
                np.nan,

            "ica_excluded":
                np.nan,

            "excluded_components":
                "",

            "status":
                f"ERROR: {str(e)}",

            "preprocessed_file":
                "",

            "ica_file":
                ""

        })


# ============================================================
# SAVE REPORTS
# ============================================================

print("\n")
print("=" * 80)
print("SAVING REPORTS")
print("=" * 80)


run_df = pd.DataFrame(
    run_summary
)

channel_df_all = pd.DataFrame(
    channel_summary
)

ica_df = pd.DataFrame(
    ica_summary
)


run_report = os.path.join(
    QC_DIR,
    "PREPROCESSING_RUN_SUMMARY.csv"
)

channel_report = os.path.join(
    QC_DIR,
    "PREPROCESSING_CHANNEL_QC.csv"
)

ica_report = os.path.join(
    QC_DIR,
    "ICA_ARTIFACT_REPORT.csv"
)


run_df.to_csv(
    run_report,
    index=False
)

channel_df_all.to_csv(
    channel_report,
    index=False
)

ica_df.to_csv(
    ica_report,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL PREPROCESSING SUMMARY")
print("=" * 80)

print(
    "\nSTATUS:"
)

print(
    run_df["status"].value_counts(
        dropna=False
    )
)

print(
    "\nSAMPLING RATES:"
)

print(
    run_df[
        run_df["status"] == "OK"
    ]["sampling_rate"].value_counts()
    .sort_index()
)

print(
    "\nSCALE FLAGS:"
)

print(
    run_df[
        run_df["status"] == "OK"
    ]["scale_flag"].value_counts()
)

print(
    "\nFILES WITH FLAGGED BAD CHANNELS:"
)

flagged = run_df[
    (run_df["status"] == "OK") &
    (run_df["flagged_bad_channels"] > 0)
]

if len(flagged) > 0:

    print(
        flagged[
            [
                "subject",
                "file",
                "global_std",
                "flat_channels",
                "extreme_channels",
                "flagged_bad_channels"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "None"
    )


print(
    "\nFILES WITH ICA EXCLUSIONS:"
)

ica_excluded = run_df[
    (run_df["status"] == "OK") &
    (run_df["ica_excluded"] > 0)
]

if len(ica_excluded) > 0:

    print(
        ica_excluded[
            [
                "subject",
                "file",
                "ica_components",
                "ica_excluded",
                "excluded_components"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "None"
    )


print("\n")
print("=" * 80)
print("REPORTS")
print("=" * 80)

print(
    f"\nRun report:\n{run_report}"
)

print(
    f"\nChannel report:\n{channel_report}"
)

print(
    f"\nICA report:\n{ica_report}"
)

print(
    f"\nPreprocessed EEG:\n{OUTPUT_DIR}"
)

print("\n")
print("=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    """
Original .SET files were NOT modified.
Original .FDT files were NOT modified.
No subjects were deleted.
No trials were deleted.
No channels were permanently deleted.
No automatic signal normalization was performed.

Suspicious scale differences were preserved and flagged.

The preprocessing stage is now ready for epoch-level analysis.
"""
)

print("=" * 80)
print("DONE")
print("=" * 80)