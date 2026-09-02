from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import re
import os

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

DATA_DIR = PROJECT_DIR / "data"

QC_DIR = PROJECT_DIR / "qc"
QC_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = QC_DIR / "raw_qc_results.csv"


# ============================================================
# HDF5 HELPERS
# ============================================================

def decode_hdf5_string(h5file, obj):
    """
    Decode EEGLAB MATLAB v7.3 HDF5 string/reference objects.
    """

    try:

        # HDF5 reference
        if isinstance(obj, h5py.Reference):

            target = h5file[obj]

            if hasattr(target, "shape"):
                arr = target[()]
            else:
                arr = target

        else:
            arr = obj

        arr = np.asarray(arr)

        # Numeric ASCII
        if np.issubdtype(arr.dtype, np.number):

            flat = arr.flatten()

            if len(flat) > 0:
                try:
                    return "".join(
                        chr(int(x))
                        for x in flat
                        if int(x) != 0
                    )
                except Exception:
                    pass

        # Bytes
        if arr.dtype.kind == "S":
            return b"".join(arr.flatten()).decode(
                "utf-8",
                errors="ignore"
            )

        # Unicode
        if arr.dtype.kind == "U":
            return "".join(arr.flatten())

        return str(arr)

    except Exception:

        return ""


def get_scalar(h5file, key):

    try:

        value = h5file[key][()]

        value = np.asarray(value).squeeze()

        if value.size == 1:
            return float(value)

        return value

    except Exception:

        return np.nan


# ============================================================
# FDT READER
# ============================================================

def read_fdt(fdt_path, n_channels, n_samples):

    expected_values = n_channels * n_samples

    expected_bytes = expected_values * 4

    actual_bytes = os.path.getsize(fdt_path)

    if actual_bytes != expected_bytes:

        raise ValueError(
            f"FDT size mismatch: "
            f"expected {expected_bytes:,} bytes, "
            f"found {actual_bytes:,}"
        )

    data = np.fromfile(
        fdt_path,
        dtype="<f4"
    )

    data = data.reshape(
        (n_channels, n_samples),
        order="F"
    )

    return data


# ============================================================
# SIGNAL QC
# ============================================================

def signal_qc(data):

    n_channels, n_samples = data.shape

    channel_mean = np.mean(data, axis=1)

    channel_std = np.std(data, axis=1)

    channel_ptp = (
        np.max(data, axis=1)
        -
        np.min(data, axis=1)
    )

    channel_min = np.min(data, axis=1)

    channel_max = np.max(data, axis=1)

    nan_count = np.isnan(data).sum()

    inf_count = np.isinf(data).sum()

    # --------------------------------------------------------
    # Flat channels
    # --------------------------------------------------------

    flat_channels = np.where(
        channel_std < 1e-6
    )[0]

    # --------------------------------------------------------
    # Extremely large amplitude
    # --------------------------------------------------------

    extreme_channels = np.where(
        channel_ptp > 5000
    )[0]

    # --------------------------------------------------------
    # Very low variance relative to recording
    # --------------------------------------------------------

    median_std = np.median(channel_std)

    if median_std > 0:

        low_variance_channels = np.where(
            channel_std < median_std * 0.05
        )[0]

    else:

        low_variance_channels = np.array([])

    # --------------------------------------------------------
    # Very high variance
    # --------------------------------------------------------

    if median_std > 0:

        high_variance_channels = np.where(
            channel_std > median_std * 10
        )[0]

    else:

        high_variance_channels = np.array([])

    # --------------------------------------------------------
    # Recording-level statistics
    # --------------------------------------------------------

    result = {

        "nan_count": int(nan_count),

        "inf_count": int(inf_count),

        "mean_amplitude": float(
            np.mean(data)
        ),

        "std_amplitude": float(
            np.std(data)
        ),

        "global_min": float(
            np.min(data)
        ),

        "global_max": float(
            np.max(data)
        ),

        "median_channel_std": float(
            np.median(channel_std)
        ),

        "max_channel_std": float(
            np.max(channel_std)
        ),

        "median_channel_ptp": float(
            np.median(channel_ptp)
        ),

        "max_channel_ptp": float(
            np.max(channel_ptp)
        ),

        "flat_channel_count": int(
            len(flat_channels)
        ),

        "extreme_channel_count": int(
            len(extreme_channels)
        ),

        "low_variance_channel_count": int(
            len(low_variance_channels)
        ),

        "high_variance_channel_count": int(
            len(high_variance_channels)
        ),

    }

    return result


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(set_path):

    result = {

        "file": set_path.name,

        "subject": "",

        "session": "",

        "task": "",

        "run": "",

        "set_path": str(set_path),

        "fdt_path": "",

        "set_exists": True,

        "fdt_exists": False,

        "fdt_size_ok": False,

        "hdf5_ok": False,

        "read_ok": False,

        "status": "ERROR",

    }

    try:

        # ----------------------------------------------------
        # Parse BIDS filename
        # ----------------------------------------------------

        name = set_path.name

        match = re.search(
            r"(sub-\d+)_"
            r"(ses-[^_]+)_"
            r"task-([^_]+)_"
            r"run-(\d+)_eeg\.set",
            name
        )

        if match:

            result["subject"] = match.group(1)

            result["session"] = match.group(2)

            result["task"] = match.group(3)

            result["run"] = match.group(4)

        # ----------------------------------------------------
        # FDT
        # ----------------------------------------------------

        fdt_path = set_path.with_suffix(".fdt")

        result["fdt_path"] = str(fdt_path)

        result["fdt_exists"] = fdt_path.exists()

        if not fdt_path.exists():

            result["status"] = "MISSING_FDT"

            return result

        # ----------------------------------------------------
        # Open HDF5 SET
        # ----------------------------------------------------

        with h5py.File(
            set_path,
            "r"
        ) as h5:

            result["hdf5_ok"] = True

            # -----------------------------------------------
            # Metadata
            # -----------------------------------------------

            nbchan = int(
                get_scalar(
                    h5,
                    "nbchan"
                )
            )

            pnts = int(
                get_scalar(
                    h5,
                    "pnts"
                )
            )

            srate = float(
                get_scalar(
                    h5,
                    "srate"
                )
            )

            trials = int(
                get_scalar(
                    h5,
                    "trials"
                )
            )

            xmin = float(
                get_scalar(
                    h5,
                    "xmin"
                )
            )

            xmax = float(
                get_scalar(
                    h5,
                    "xmax"
                )
            )

            # -----------------------------------------------
            # Events
            # -----------------------------------------------

            event_count = 0

            if "event" in h5:

                if "latency" in h5["event"]:

                    event_count = (
                        h5["event"]["latency"].shape[0]
                    )

            # -----------------------------------------------
            # Channel locations
            # -----------------------------------------------

            channel_location_count = 0

            if "chanlocs" in h5:

                if "labels" in h5["chanlocs"]:

                    channel_location_count = (
                        h5["chanlocs"]["labels"].shape[0]
                    )

        # ----------------------------------------------------
        # Expected FDT size
        # ----------------------------------------------------

        expected_bytes = (
            nbchan
            *
            pnts
            *
            4
        )

        actual_bytes = (
            os.path.getsize(fdt_path)
        )

        result["fdt_size_bytes"] = actual_bytes

        result["expected_fdt_size_bytes"] = (
            expected_bytes
        )

        result["fdt_size_ok"] = (
            actual_bytes == expected_bytes
        )

        if not result["fdt_size_ok"]:

            result["status"] = "FDT_SIZE_MISMATCH"

            return result

        # ----------------------------------------------------
        # Read EEG
        # ----------------------------------------------------

        data = read_fdt(
            fdt_path,
            nbchan,
            pnts
        )

        # ----------------------------------------------------
        # Basic metadata
        # ----------------------------------------------------

        result["channels"] = nbchan

        result["samples"] = pnts

        result["sampling_rate"] = srate

        result["trials"] = trials

        result["xmin"] = xmin

        result["xmax"] = xmax

        result["duration_seconds"] = (
            pnts / srate
        )

        result["events"] = event_count

        result["channel_locations"] = (
            channel_location_count
        )

        # ----------------------------------------------------
        # Signal QC
        # ----------------------------------------------------

        qc = signal_qc(data)

        result.update(qc)

        result["read_ok"] = True

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if (
            result["nan_count"] > 0
            or
            result["inf_count"] > 0
            or
            result["flat_channel_count"] > 0
            or
            result["fdt_size_ok"] is False
        ):

            result["status"] = "FAIL"

        elif (
            result["high_variance_channel_count"] > 0
            or
            result["low_variance_channel_count"] > 0
            or
            result["extreme_channel_count"] > 0
        ):

            result["status"] = "REVIEW"

        else:

            result["status"] = "PASS"

        return result

    except Exception as e:

        result["error"] = str(e)

        result["status"] = "ERROR"

        return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "BRAIN PERTURBATION PROJECT"
    )

    print(
        "RAW EEG QUALITY CONTROL"
    )

    print("=" * 70)

    set_files = sorted(
        DATA_DIR.rglob("*_eeg.set")
    )

    print(
        f"\nFound {len(set_files)} SET files."
    )

    if len(set_files) == 0:

        print(
            "ERROR: No EEG SET files found."
        )

        return

    results = []

    for i, set_path in enumerate(
        set_files,
        start=1
    ):

        print(
            f"[{i:03d}/{len(set_files):03d}] "
            f"{set_path.name}"
        )

        result = process_file(
            set_path
        )

        results.append(result)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    # Sort
    df = df.sort_values(
        [
            "subject",
            "run"
        ]
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_file = (
        QC_DIR /
        "raw_qc_summary.txt"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "BRAIN PERTURBATION PROJECT\n"
        )

        f.write(
            "RAW EEG QC SUMMARY\n"
        )

        f.write(
            "=" * 50 + "\n\n"
        )

        f.write(
            f"Total SET files: "
            f"{len(df)}\n"
        )

        f.write(
            f"PASS: "
            f"{(df['status'] == 'PASS').sum()}\n"
        )

        f.write(
            f"REVIEW: "
            f"{(df['status'] == 'REVIEW').sum()}\n"
        )

        f.write(
            f"FAIL: "
            f"{(df['status'] == 'FAIL').sum()}\n"
        )

        f.write(
            f"ERROR: "
            f"{(df['status'] == 'ERROR').sum()}\n"
        )

        f.write(
            f"MISSING FDT: "
            f"{(df['status'] == 'MISSING_FDT').sum()}\n"
        )

        f.write(
            f"FDT SIZE MISMATCH: "
            f"{(df['status'] == 'FDT_SIZE_MISMATCH').sum()}\n"
        )

    print("\n" + "=" * 70)

    print("QC COMPLETE")

    print("=" * 70)

    print(
        f"Results saved to:\n"
        f"{OUTPUT_CSV}"
    )

    print(
        f"\nSummary saved to:\n"
        f"{summary_file}"
    )

    print("\nSTATUS COUNTS:")

    print(
        df["status"].value_counts()
    )


if __name__ == "__main__":

    main()