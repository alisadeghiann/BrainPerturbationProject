import os
import glob
import numpy as np
import pandas as pd
import h5py


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "qc"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIND ALL SET FILES
# ============================================================

set_files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "sub-*",
            "ses-*",
            "eeg",
            "*.set"
        )
    )
)

print("=" * 80)
print("BAD CHANNEL DETECTION")
print("=" * 80)

print(
    f"\nNumber of EEG files: {len(set_files)}"
)

if len(set_files) == 0:

    raise FileNotFoundError(
        "No .set files found."
    )


# ============================================================
# READ HDF5 STRING
# ============================================================

def read_hdf5_string(dataset):

    arr = np.array(
        dataset
    ).astype(
        np.uint16
    ).flatten()

    return "".join(
        chr(int(x))
        for x in arr
        if int(x) != 0
    )


# ============================================================
# READ SET METADATA
# ============================================================

def read_set_metadata(
    set_file
):

    with h5py.File(
        set_file,
        "r"
    ) as f:

        nbchan = int(
            np.array(
                f["nbchan"]
            ).squeeze()
        )

        pnts = int(
            np.array(
                f["pnts"]
            ).squeeze()
        )

        trials = int(
            np.array(
                f["trials"]
            ).squeeze()
        )

        srate = float(
            np.array(
                f["srate"]
            ).squeeze()
        )

        datfile = read_hdf5_string(
            f["datfile"]
        )

    return (
        nbchan,
        pnts,
        trials,
        srate,
        datfile
    )


# ============================================================
# FIND FDT
# ============================================================

def find_fdt(
    set_file,
    datfile
):

    directory = os.path.dirname(
        set_file
    )

    candidate = os.path.join(
        directory,
        datfile
    )

    if os.path.exists(
        candidate
    ):

        return candidate

    candidate = (
        os.path.splitext(
            set_file
        )[0]
        + ".fdt"
    )

    if os.path.exists(
        candidate
    ):

        return candidate

    raise FileNotFoundError(
        f"FDT not found:\n{set_file}"
    )


# ============================================================
# READ FDT
# ============================================================

def read_fdt(
    fdt_file,
    nbchan,
    pnts,
    trials
):

    expected_values = (
        nbchan
        * pnts
        * trials
    )

    file_size = os.path.getsize(
        fdt_file
    )

    expected_float32 = (
        expected_values * 4
    )

    expected_float64 = (
        expected_values * 8
    )

    if file_size == expected_float32:

        dtype = np.float32

    elif file_size == expected_float64:

        dtype = np.float64

    else:

        raise ValueError(
            "FDT size does not match "
            "expected dimensions."
        )

    data = np.memmap(
        fdt_file,
        dtype=dtype,
        mode="r",
        shape=(
            nbchan,
            pnts * trials
        ),
        order="F"
    )

    return data


# ============================================================
# DETECT BAD CHANNELS
# ============================================================

def detect_bad_channels(
    data
):

    # --------------------------------------------------------
    # Channel statistics
    # --------------------------------------------------------

    channel_mean = np.mean(
        data,
        axis=1
    )

    channel_std = np.std(
        data,
        axis=1
    )

    channel_min = np.min(
        data,
        axis=1
    )

    channel_max = np.max(
        data,
        axis=1
    )

    channel_ptp = (
        channel_max
        - channel_min
    )

    # --------------------------------------------------------
    # Robust reference values
    # --------------------------------------------------------

    median_std = np.median(
        channel_std
    )

    median_ptp = np.median(
        channel_ptp
    )

    # --------------------------------------------------------
    # Flat channels
    # --------------------------------------------------------

    flat_threshold = 1e-12

    flat = (
        channel_std
        < flat_threshold
    )

    # --------------------------------------------------------
    # Very high variance
    # --------------------------------------------------------

    high_std = (
        channel_std
        > median_std * 10
    )

    # --------------------------------------------------------
    # Very low variance
    # --------------------------------------------------------

    low_std = (
        channel_std
        < median_std * 0.1
    )

    # --------------------------------------------------------
    # Very high peak-to-peak
    # --------------------------------------------------------

    high_ptp = (
        channel_ptp
        > median_ptp * 10
    )

    # --------------------------------------------------------
    # Combine criteria
    # --------------------------------------------------------

    bad = (
        flat
        | high_std
        | low_std
        | high_ptp
    )

    return {
        "mean": channel_mean,
        "std": channel_std,
        "min": channel_min,
        "max": channel_max,
        "ptp": channel_ptp,
        "flat": flat,
        "high_std": high_std,
        "low_std": low_std,
        "high_ptp": high_ptp,
        "bad": bad
    }


# ============================================================
# PROCESS FILES
# ============================================================

all_results = []

summary_results = []


for file_number, set_file in enumerate(
    set_files,
    start=1
):

    print()
    print("=" * 80)

    print(
        f"PROCESSING "
        f"{file_number}/{len(set_files)}"
    )

    print("=" * 80)

    filename = os.path.basename(
        set_file
    )

    path_parts = set_file.split(
        os.sep
    )

    subject = next(
        (
            p
            for p in path_parts
            if p.startswith("sub-")
        ),
        "unknown"
    )

    print(
        f"Subject: {subject}"
    )

    print(
        f"File: {filename}"
    )

    try:

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        (
            nbchan,
            pnts,
            trials,
            srate,
            datfile
        ) = read_set_metadata(
            set_file
        )

        # ----------------------------------------------------
        # FDT
        # ----------------------------------------------------

        fdt_file = find_fdt(
            set_file,
            datfile
        )

        # ----------------------------------------------------
        # EEG
        # ----------------------------------------------------

        data = read_fdt(
            fdt_file,
            nbchan,
            pnts,
            trials
        )

        print(
            f"Data shape: {data.shape}"
        )

        # ----------------------------------------------------
        # Detect bad channels
        # ----------------------------------------------------

        qc = detect_bad_channels(
            data
        )

        bad_indices = np.where(
            qc["bad"]
        )[0]

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Median STD: "
            f"{np.median(qc['std']):.4f}"
        )

        print(
            f"Median PTP: "
            f"{np.median(qc['ptp']):.4f}"
        )

        print(
            f"Bad channels: "
            f"{len(bad_indices)}"
        )

        if len(bad_indices) > 0:

            print(
                "Bad channel indices:"
            )

            print(
                (
                    bad_indices + 1
                ).tolist()
            )

        else:

            print(
                "No bad channels detected."
            )

        # ----------------------------------------------------
        # Channel-level results
        # ----------------------------------------------------

        for ch in range(
            nbchan
        ):

            reasons = []

            if qc["flat"][ch]:

                reasons.append(
                    "flat"
                )

            if qc["high_std"][ch]:

                reasons.append(
                    "high_std"
                )

            if qc["low_std"][ch]:

                reasons.append(
                    "low_std"
                )

            if qc["high_ptp"][ch]:

                reasons.append(
                    "high_ptp"
                )

            status = (
                "BAD"
                if qc["bad"][ch]
                else "GOOD"
            )

            all_results.append({

                "subject": subject,

                "file": filename,

                "channel_index": ch + 1,

                "mean": float(
                    qc["mean"][ch]
                ),

                "std": float(
                    qc["std"][ch]
                ),

                "min": float(
                    qc["min"][ch]
                ),

                "max": float(
                    qc["max"][ch]
                ),

                "ptp": float(
                    qc["ptp"][ch]
                ),

                "flat": bool(
                    qc["flat"][ch]
                ),

                "high_std": bool(
                    qc["high_std"][ch]
                ),

                "low_std": bool(
                    qc["low_std"][ch]
                ),

                "high_ptp": bool(
                    qc["high_ptp"][ch]
                ),

                "status": status,

                "reason": ",".join(
                    reasons
                )

            })

        # ----------------------------------------------------
        # File-level summary
        # ----------------------------------------------------

        summary_results.append({

            "subject": subject,

            "file": filename,

            "n_channels": nbchan,

            "n_samples": pnts,

            "sampling_rate": srate,

            "duration_seconds":
                pnts / srate,

            "n_bad_channels":
                len(bad_indices),

            "bad_channel_indices":
                str(
                    (
                        bad_indices + 1
                    ).tolist()
                )

        })

    except Exception as e:

        print(
            f"ERROR: {repr(e)}"
        )

        summary_results.append({

            "subject": subject,

            "file": filename,

            "error": repr(e)

        })


# ============================================================
# SAVE CHANNEL-LEVEL REPORT
# ============================================================

channel_df = pd.DataFrame(
    all_results
)

channel_report = os.path.join(
    OUTPUT_DIR,
    "bad_channel_detection.csv"
)

channel_df.to_csv(
    channel_report,
    index=False
)


# ============================================================
# SAVE FILE-LEVEL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_results
)

summary_report = os.path.join(
    OUTPUT_DIR,
    "bad_channel_summary.csv"
)

summary_df.to_csv(
    summary_report,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("BAD CHANNEL DETECTION COMPLETE")
print("=" * 80)

print()
print(
    "Channel-level report:"
)

print(
    channel_report
)

print()
print(
    "File-level summary:"
)

print(
    summary_report
)

print()
print(
    "Total files:",
    len(set_files)
)

print()
print(
    "Files with detected bad channels:"
)

if "n_bad_channels" in summary_df.columns:

    print(
        (
            summary_df[
                summary_df[
                    "n_bad_channels"
                ] > 0
            ]
        )[
            [
                "subject",
                "file",
                "n_bad_channels",
                "bad_channel_indices"
            ]
        ].to_string(
            index=False
        )
    )

print()
print("DONE.")