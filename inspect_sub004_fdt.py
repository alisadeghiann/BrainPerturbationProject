import os
import numpy as np
import h5py

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-004\ses-01\eeg"

RUNS = [1, 2, 3, 4]

# EEGLAB float32 data
DTYPE = np.float32

# EEG channels only: first 69 channels
EEG_CHANNELS = 69

# ============================================================
# HELPER
# ============================================================

def read_uint16_string(dataset):
    """
    Read MATLAB/HDF5 uint16/uint8 character array and convert to string.
    """
    arr = np.array(dataset).squeeze()

    if arr.dtype.kind in ("u", "i"):
        return "".join(chr(int(x)) for x in arr if int(x) != 0)

    return str(arr)


def load_set_metadata(set_file):
    """
    Read only metadata needed from the EEGLAB .set HDF5 file.
    """
    with h5py.File(set_file, "r") as f:

        nbchan = int(np.array(f["nbchan"]).squeeze())
        pnts = int(np.array(f["pnts"]).squeeze())
        srate = float(np.array(f["srate"]).squeeze())
        trials = int(np.array(f["trials"]).squeeze())

        return nbchan, pnts, srate, trials


def load_channel_names(set_file, expected_channels):
    """
    Read channel labels from chanlocs.
    """

    names = []

    with h5py.File(set_file, "r") as f:

        chanlocs = f["chanlocs"]

        labels = chanlocs["labels"]

        for i in range(expected_channels):

            ref = labels[i, 0]

            # HDF5 object reference
            if isinstance(ref, h5py.h5r.Reference):

                obj = f[ref]

                arr = np.array(obj).squeeze()

                if arr.dtype.kind in ("u", "i"):
                    name = "".join(
                        chr(int(x))
                        for x in arr
                        if int(x) != 0
                    )
                else:
                    name = str(arr)

            else:
                name = str(ref)

            names.append(name)

    return names


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("SUB-004 REAL FDT INSPECTION")
print("=" * 70)

all_results = []

for run in RUNS:

    print()
    print("=" * 70)
    print(f"RUN {run}")
    print("=" * 70)

    base = f"sub-004_ses-01_task-WorkingMemory_run-{run}_eeg"

    set_file = os.path.join(DATA_DIR, base + ".set")
    fdt_file = os.path.join(DATA_DIR, base + ".fdt")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(set_file):
        print("ERROR: SET file not found:")
        print(set_file)
        continue

    if not os.path.exists(fdt_file):
        print("ERROR: FDT file not found:")
        print(fdt_file)
        continue

    print("SET:", set_file)
    print("FDT:", fdt_file)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    nbchan, pnts, srate, trials = load_set_metadata(set_file)

    print()
    print("SET METADATA")
    print("-" * 70)
    print("Channels:", nbchan)
    print("Points:", pnts)
    print("Sampling rate:", srate)
    print("Trials:", trials)
    print("Expected EEG channels:", EEG_CHANNELS)

    # --------------------------------------------------------
    # Channel names
    # --------------------------------------------------------

    channel_names = load_channel_names(
        set_file,
        nbchan
    )

    # --------------------------------------------------------
    # FDT size
    # --------------------------------------------------------

    file_size = os.path.getsize(fdt_file)

    expected_values = nbchan * pnts * trials
    expected_bytes = expected_values * np.dtype(DTYPE).itemsize

    print()
    print("FDT SIZE CHECK")
    print("-" * 70)
    print("Actual FDT bytes:   ", file_size)
    print("Expected FDT bytes: ", expected_bytes)

    if file_size == expected_bytes:
        print("FDT SIZE STATUS: OK")
    else:
        print("FDT SIZE STATUS: WARNING")
        print(
            "Difference:",
            file_size - expected_bytes,
            "bytes"
        )

    # --------------------------------------------------------
    # Memory-map FDT
    # --------------------------------------------------------

    print()
    print("Reading REAL EEG from FDT...")

    try:

        raw = np.memmap(
            fdt_file,
            dtype=DTYPE,
            mode="r",
            shape=(nbchan, pnts * trials),
            order="F"
        )

    except Exception as e:

        print("ERROR reading FDT:")
        print(e)
        continue

    print("FDT shape:", raw.shape)
    print("FDT dtype:", raw.dtype)

    # --------------------------------------------------------
    # Basic global information
    # --------------------------------------------------------

    print()
    print("GLOBAL FDT CHECK")
    print("-" * 70)

    print("Global min:", float(np.min(raw)))
    print("Global max:", float(np.max(raw)))

    # --------------------------------------------------------
    # Channel QC
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CHANNEL-LEVEL FDT QC")
    print("=" * 70)

    print(
        f"{'Idx':>4} "
        f"{'Channel':<8} "
        f"{'STD':>12} "
        f"{'Mean':>12} "
        f"{'Min':>12} "
        f"{'Max':>12} "
        f"{'>100%':>9} "
        f"{'>200%':>9} "
        f"{'Clip%':>9}"
    )

    print("-" * 105)

    for ch in range(EEG_CHANNELS):

        signal = np.asarray(raw[ch], dtype=np.float64)

        name = channel_names[ch]

        mean = np.mean(signal)
        std = np.std(signal)
        minimum = np.min(signal)
        maximum = np.max(signal)

        high100 = np.mean(np.abs(signal) > 100) * 100
        high200 = np.mean(np.abs(signal) > 200) * 100

        clipping = np.mean(
            (signal <= -499) |
            (signal >= 499)
        ) * 100

        result = {
            "run": run,
            "channel_index": ch + 1,
            "channel": name,
            "std": std,
            "mean": mean,
            "min": minimum,
            "max": maximum,
            "high_amplitude_100_percent": high100,
            "high_amplitude_200_percent": high200,
            "clip_percent": clipping,
        }

        all_results.append(result)

        print(
            f"{ch+1:4d} "
            f"{name:<8} "
            f"{std:12.3f} "
            f"{mean:12.3f} "
            f"{minimum:12.3f} "
            f"{maximum:12.3f} "
            f"{high100:9.3f} "
            f"{high200:9.3f} "
            f"{clipping:9.3f}"
        )

    # --------------------------------------------------------
    # EEG median STD
    # --------------------------------------------------------

    eeg_stds = np.array([
        r["std"]
        for r in all_results
        if r["run"] == run
    ])

    median_std = np.median(eeg_stds)

    print()
    print("=" * 70)
    print(f"RUN {run} SUMMARY")
    print("=" * 70)

    print("Median EEG STD:", round(float(median_std), 3))

    # --------------------------------------------------------
    # Find suspicious channels
    # --------------------------------------------------------

    print()
    print("POTENTIALLY SUSPICIOUS CHANNELS")
    print("-" * 70)

    found = False

    for r in all_results:

        if r["run"] != run:
            continue

        ratio = r["std"] / median_std

        if (
            ratio > 5
            or r["clip_percent"] > 1
            or r["high_amplitude_200_percent"] > 1
        ):

            found = True

            print(
                f"{r['channel']:<6} "
                f"STD={r['std']:.3f} "
                f"ratio={ratio:.2f} "
                f">200={r['high_amplitude_200_percent']:.3f}% "
                f"clip={r['clip_percent']:.3f}%"
            )

    if not found:
        print("No obvious suspicious channels by these thresholds.")

    # --------------------------------------------------------
    # Close memmap
    # --------------------------------------------------------

    del raw


# ============================================================
# SAVE CSV
# ============================================================

import csv

output_file = r"C:\Users\Ali\Desktop\BrainPerturbationProject\qc\sub004_fdt_channel_inspection.csv"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

if all_results:

    fieldnames = list(all_results[0].keys())

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print()
    print("Total channel records:", len(all_results))
    print("Expected:", len(RUNS) * EEG_CHANNELS)
    print()
    print("Saved:")
    print(output_file)

else:

    print()
    print("NO RESULTS GENERATED.")