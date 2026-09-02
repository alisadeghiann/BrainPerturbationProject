from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import os
import re

PROJECT = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")
DATA = PROJECT / "data"
QC = PROJECT / "qc"
QC.mkdir(exist_ok=True)

OUTPUT = QC / "raw_qc_eeg_only.csv"


def scalar(h5, key):
    return int(np.asarray(h5[key][()]).squeeze())


def decode_label(h5, ref):
    try:
        obj = h5[ref]
        arr = np.asarray(obj[()]).squeeze()

        if arr.dtype.kind in "iu":
            return "".join(
                chr(int(x))
                for x in arr.flatten()
                if int(x) != 0
            )

        if arr.dtype.kind == "S":
            return b"".join(arr.flatten()).decode(
                "utf-8", errors="ignore"
            )

        return str(arr)

    except Exception:
        return "UNKNOWN"


def process_file(set_file):

    result = {
        "file": set_file.name,
        "subject": "",
        "run": "",
        "channels_total": 0,
        "eeg_channels": 0,
        "eog_channels": 0,
        "sampling_rate": np.nan,
        "duration_seconds": np.nan,
        "events": 0,
        "nan_count": 0,
        "inf_count": 0,
        "median_eeg_std": np.nan,
        "max_eeg_std": np.nan,
        "high_variance_eeg": 0,
        "low_variance_eeg": 0,
        "status": "ERROR",
        "error": ""
    }

    try:

        name = set_file.name

        match = re.search(
            r"(sub-\d+).*run-(\d+)_eeg\.set",
            name
        )

        if match:
            result["subject"] = match.group(1)
            result["run"] = match.group(2)

        fdt = set_file.with_suffix(".fdt")

        with h5py.File(set_file, "r") as h5:

            nbchan = scalar(h5, "nbchan")
            pnts = scalar(h5, "pnts")

            srate = float(
                np.asarray(
                    h5["srate"][()]
                ).squeeze()
            )

            result["channels_total"] = nbchan
            result["sampling_rate"] = srate
            result["duration_seconds"] = pnts / srate

            # ------------------------------------------------
            # Identify EEG / EOG channels
            # ------------------------------------------------

            labels = h5["chanlocs"]["labels"]
            types = h5["chanlocs"]["type"]

            eeg_indices = []
            eog_indices = []

            for i in range(labels.shape[0]):

                label = decode_label(
                    h5,
                    labels[i, 0]
                ).upper()

                ch_type = decode_label(
                    h5,
                    types[i, 0]
                ).upper()

                if ch_type == "EEG":
                    eeg_indices.append(i)

                elif ch_type == "EOG":
                    eog_indices.append(i)

            result["eeg_channels"] = len(eeg_indices)
            result["eog_channels"] = len(eog_indices)

            # ------------------------------------------------
            # Events
            # ------------------------------------------------

            if "event" in h5:
                if "latency" in h5["event"]:
                    result["events"] = (
                        h5["event"]["latency"].shape[0]
                    )

        # ----------------------------------------------------
        # Read FDT
        # ----------------------------------------------------

        data = np.fromfile(
            fdt,
            dtype="<f4"
        ).reshape(
            (nbchan, pnts),
            order="F"
        )

        # ONLY EEG CHANNELS
        eeg_data = data[eeg_indices, :]

        result["nan_count"] = int(
            np.isnan(eeg_data).sum()
        )

        result["inf_count"] = int(
            np.isinf(eeg_data).sum()
        )

        std = np.std(
            eeg_data,
            axis=1
        )

        median_std = np.median(std)

        result["median_eeg_std"] = float(
            median_std
        )

        result["max_eeg_std"] = float(
            np.max(std)
        )

        high = np.where(
            std > median_std * 10
        )[0]

        low = np.where(
            std < median_std * 0.05
        )[0]

        result["high_variance_eeg"] = len(high)
        result["low_variance_eeg"] = len(low)

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if (
            result["nan_count"] > 0
            or
            result["inf_count"] > 0
        ):
            result["status"] = "FAIL"

        elif (
            len(high) > 0
            or
            len(low) > 0
        ):
            result["status"] = "REVIEW"

        else:
            result["status"] = "PASS"

        return result

    except Exception as e:

        result["error"] = str(e)

        return result


# ============================================================
# MAIN
# ============================================================

files = sorted(
    DATA.rglob("*_eeg.set")
)

print("=" * 70)
print("EEG-ONLY QUALITY CONTROL")
print("=" * 70)

print(
    f"\nFound {len(files)} recordings.\n"
)

results = []

for i, f in enumerate(files, 1):

    print(
        f"[{i:03d}/{len(files):03d}] "
        f"{f.name}"
    )

    results.append(
        process_file(f)
    )

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)

print("\nSTATUS:")

print(
    df["status"].value_counts()
)

print(
    f"\nSaved:\n{OUTPUT}"
)