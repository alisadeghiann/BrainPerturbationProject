from pathlib import Path
import numpy as np
import pandas as pd
import h5py


# ============================================================
# CONFIG
# ============================================================

PROJECT = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

DATA_DIR = PROJECT / "data"
QC_DIR = PROJECT / "qc"

QC_DIR.mkdir(exist_ok=True)


# ============================================================
# THRESHOLDS
# ============================================================

# Channel STD relative to median EEG channel STD
HIGH_STD_RATIO = 5.0
REVIEW_STD_RATIO = 2.5

# Percentage of samples with large amplitude
CLIP_THRESHOLD = 499.0
HIGH_AMPLITUDE_THRESHOLD = 200.0

HIGH_CLIP_PERCENT = 1.0
REVIEW_CLIP_PERCENT = 0.1

HIGH_AMPLITUDE_PERCENT = 5.0
REVIEW_AMPLITUDE_PERCENT = 1.0

# Very low variance
LOW_STD_RATIO = 0.10


# ============================================================
# HELPER: DECODE MATLAB STRING
# ============================================================

def decode_hdf5_string(h5, ref):

    try:

        obj = h5[ref]

        arr = np.asarray(obj[()]).squeeze()

        if arr.dtype.kind in "iu":

            return "".join(
                chr(int(x))
                for x in arr.flatten()
                if int(x) != 0
            )

        return str(arr)

    except Exception:

        return str(ref)


# ============================================================
# READ CHANNEL NAMES
# ============================================================

def get_channel_names(h5):

    labels = h5["chanlocs"]["labels"]

    names = []

    for i in range(labels.shape[0]):

        names.append(
            decode_hdf5_string(
                h5,
                labels[i, 0]
            )
        )

    return [
        x.strip()
        for x in names
    ]


# ============================================================
# LOAD ONE EEGLAB FILE
# ============================================================

def load_eeglab_v73(set_file):

    fdt_file = set_file.with_suffix(".fdt")

    with h5py.File(set_file, "r") as h5:

        nbchan = int(
            np.asarray(
                h5["nbchan"][()]
            ).squeeze()
        )

        pnts = int(
            np.asarray(
                h5["pnts"][()]
            ).squeeze()
        )

        srate = float(
            np.asarray(
                h5["srate"][()]
            ).squeeze()
        )

        trials = int(
            np.asarray(
                h5["trials"][()]
            ).squeeze()
        )

        names = get_channel_names(h5)

    if not fdt_file.exists():

        raise FileNotFoundError(
            f"Missing FDT: {fdt_file}"
        )

    expected_values = nbchan * pnts * trials

    data = np.fromfile(
        fdt_file,
        dtype="<f4"
    )

    if data.size != expected_values:

        raise ValueError(
            f"Unexpected FDT size: "
            f"{data.size} "
            f"expected {expected_values}"
        )

    if trials != 1:

        raise ValueError(
            "This QC script expects continuous "
            "single-trial recordings."
        )

    data = data.reshape(
        (nbchan, pnts),
        order="F"
    )

    return data, names, srate


# ============================================================
# QC ONE RUN
# ============================================================

def qc_one_run(set_file):

    data, names, srate = load_eeglab_v73(
        set_file
    )

    subject = set_file.parts[
        set_file.parts.index("data") + 1
    ]

    # --------------------------------------------------------
    # Identify EEG channels only
    # --------------------------------------------------------

    eeg_indices = [
        i
        for i, name in enumerate(names)
        if name.upper() not in [
            "LEYE",
            "REYE",
            "HEOG",
            "VEOG",
            "EOG"
        ]
    ]

    eeg_names = [
        names[i]
        for i in eeg_indices
    ]

    eeg_data = data[
        eeg_indices
    ].astype(np.float64)

    # --------------------------------------------------------
    # Basic recording information
    # --------------------------------------------------------

    duration = data.shape[1] / srate

    channel_stds = np.std(
        eeg_data,
        axis=1
    )

    median_std = np.median(
        channel_stds
    )

    # --------------------------------------------------------
    # Per-channel QC
    # --------------------------------------------------------

    rows = []

    for local_i, channel_index in enumerate(
        eeg_indices
    ):

        name = names[channel_index]

        signal = data[
            channel_index
        ].astype(np.float64)

        std = np.std(signal)

        mean = np.mean(signal)

        minimum = np.min(signal)

        maximum = np.max(signal)

        std_ratio = (
            std / median_std
            if median_std > 0
            else np.inf
        )

        clip_percent = (
            np.mean(
                np.abs(signal)
                >= CLIP_THRESHOLD
            )
            * 100
        )

        high_amp_percent = (
            np.mean(
                np.abs(signal)
                > HIGH_AMPLITUDE_THRESHOLD
            )
            * 100
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        reasons = []

        # Extremely high variance
        if std_ratio >= HIGH_STD_RATIO:

            reasons.append(
                "HIGH_STD"
            )

        elif std_ratio >= REVIEW_STD_RATIO:

            reasons.append(
                "ELEVATED_STD"
            )

        # Very low variance
        if std_ratio <= LOW_STD_RATIO:

            reasons.append(
                "LOW_STD"
            )

        # Clipping
        if clip_percent >= HIGH_CLIP_PERCENT:

            reasons.append(
                "HIGH_CLIPPING"
            )

        elif clip_percent >= REVIEW_CLIP_PERCENT:

            reasons.append(
                "CLIPPING_REVIEW"
            )

        # Large amplitude
        if high_amp_percent >= HIGH_AMPLITUDE_PERCENT:

            reasons.append(
                "HIGH_AMPLITUDE"
            )

        elif high_amp_percent >= REVIEW_AMPLITUDE_PERCENT:

            reasons.append(
                "AMPLITUDE_REVIEW"
            )

        # Final channel status
        if any(
            x in reasons
            for x in [
                "HIGH_STD",
                "LOW_STD",
                "HIGH_CLIPPING",
                "HIGH_AMPLITUDE"
            ]
        ):

            status = "BAD"

        elif len(reasons) > 0:

            status = "REVIEW"

        else:

            status = "PASS"

        rows.append({

            "subject": subject,

            "run": set_file.stem.split(
                "_run-"
            )[1].split(
                "_"
            )[0],

            "file": set_file.name,

            "channel": name,

            "channel_index": channel_index + 1,

            "std": std,

            "median_eeg_std": median_std,

            "std_ratio": std_ratio,

            "mean": mean,

            "min": minimum,

            "max": maximum,

            "clip_percent": clip_percent,

            "high_amplitude_percent":
                high_amp_percent,

            "status": status,

            "reasons": ";".join(
                reasons
            )

        })

    return rows, {

        "subject": subject,

        "file": set_file.name,

        "channels_total": len(names),

        "eeg_channels": len(eeg_indices),

        "srate": srate,

        "duration_seconds": duration,

        "median_eeg_std": median_std

    }


# ============================================================
# FIND ALL SET FILES
# ============================================================

set_files = sorted(
    DATA_DIR.rglob("*.set")
)

print("=" * 70)
print("AUTOMATED EEG QC - ALL RUNS")
print("=" * 70)

print(
    f"\nData directory:"
)

print(
    DATA_DIR
)

print(
    f"\nSET files found: "
    f"{len(set_files)}"
)

if len(set_files) != 83:

    print(
        "\nWARNING:"
    )

    print(
        f"Expected 83 files but found "
        f"{len(set_files)}."
    )


# ============================================================
# PROCESS
# ============================================================

all_channel_rows = []

all_run_rows = []

errors = []


for counter, set_file in enumerate(
    set_files,
    start=1
):

    print(
        f"\n[{counter}/{len(set_files)}] "
        f"{set_file.name}"
    )

    try:

        channel_rows, run_info = qc_one_run(
            set_file
        )

        all_channel_rows.extend(
            channel_rows
        )

        all_run_rows.append(
            run_info
        )

        bad_count = sum(
            x["status"] == "BAD"
            for x in channel_rows
        )

        review_count = sum(
            x["status"] == "REVIEW"
            for x in channel_rows
        )

        print(
            f"    EEG channels: "
            f"{run_info['eeg_channels']}"
        )

        print(
            f"    BAD channels: "
            f"{bad_count}"
        )

        print(
            f"    REVIEW channels: "
            f"{review_count}"
        )

    except Exception as e:

        print(
            f"    ERROR: {e}"
        )

        errors.append({

            "file": str(set_file),

            "error": str(e)

        })


# ============================================================
# SAVE CHANNEL LEVEL RESULTS
# ============================================================

channel_df = pd.DataFrame(
    all_channel_rows
)

channel_output = (
    QC_DIR /
    "automated_channel_qc_83runs.csv"
)

channel_df.to_csv(
    channel_output,
    index=False
)


# ============================================================
# SAVE RUN LEVEL RESULTS
# ============================================================

run_df = pd.DataFrame(
    all_run_rows
)

run_output = (
    QC_DIR /
    "automated_run_qc_83runs.csv"
)

run_df.to_csv(
    run_output,
    index=False
)


# ============================================================
# BAD / REVIEW SUMMARY
# ============================================================

if len(channel_df) > 0:

    bad_df = channel_df[
        channel_df["status"] == "BAD"
    ].copy()

    review_df = channel_df[
        channel_df["status"] == "REVIEW"
    ].copy()

else:

    bad_df = pd.DataFrame()
    review_df = pd.DataFrame()


# ============================================================
# CHANNEL SUMMARY
# ============================================================

summary_output = (
    QC_DIR /
    "automated_qc_summary.txt"
)


with open(
    summary_output,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        "AUTOMATED EEG QC SUMMARY\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"SET files found: "
        f"{len(set_files)}\n"
    )

    f.write(
        f"Runs successfully processed: "
        f"{len(all_run_rows)}\n"
    )

    f.write(
        f"Runs with errors: "
        f"{len(errors)}\n\n"
    )

    f.write(
        "CHANNEL STATUS COUNTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    if len(channel_df) > 0:

        counts = (
            channel_df["status"]
            .value_counts()
        )

        for status, count in counts.items():

            f.write(
                f"{status}: "
                f"{count}\n"
            )

    f.write("\n")

    f.write(
        "BAD CHANNELS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    if len(bad_df) > 0:

        f.write(
            bad_df[
                [
                    "subject",
                    "run",
                    "channel",
                    "std",
                    "std_ratio",
                    "clip_percent",
                    "high_amplitude_percent",
                    "reasons"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        f.write(
            "None\n"
        )

    f.write("\n\n")

    f.write(
        "REVIEW CHANNELS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    if len(review_df) > 0:

        f.write(
            review_df[
                [
                    "subject",
                    "run",
                    "channel",
                    "std",
                    "std_ratio",
                    "clip_percent",
                    "high_amplitude_percent",
                    "reasons"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        f.write(
            "None\n"
        )

    f.write("\n\n")

    if errors:

        f.write(
            "ERRORS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for error in errors:

            f.write(
                f"{error['file']}\n"
            )

            f.write(
                f"    {error['error']}\n"
            )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("QC COMPLETE")
print("=" * 70)

print(
    f"\nRuns found: "
    f"{len(set_files)}"
)

print(
    f"Runs processed: "
    f"{len(all_run_rows)}"
)

print(
    f"Errors: "
    f"{len(errors)}"
)


if len(channel_df) > 0:

    print(
        "\nCHANNEL STATUS:"
    )

    print(
        channel_df[
            "status"
        ].value_counts()
    )

    print(
        "\nBAD CHANNELS:"
    )

    if len(bad_df) > 0:

        print(
            bad_df[
                [
                    "subject",
                    "run",
                    "channel",
                    "std_ratio",
                    "clip_percent",
                    "reasons"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "None"
        )

    print(
        "\nREVIEW CHANNELS:"
    )

    if len(review_df) > 0:

        print(
            review_df[
                [
                    "subject",
                    "run",
                    "channel",
                    "std_ratio",
                    "clip_percent",
                    "reasons"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "None"
        )


print(
    "\nSaved:"
)

print(
    channel_output
)

print(
    run_output
)

print(
    summary_output
)

print(
    "\nDO NOT MODIFY RAW DATA."
)

print(
    "Use the QC results for the next preprocessing step."
)

print(
    "=" * 70
)