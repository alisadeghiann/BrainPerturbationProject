from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

SET_FILE = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-018\ses-01\eeg\sub-018_ses-01_task-WorkingMemory_run-1_eeg.set"
)

OUTPUT_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\qc\po4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK FILES
# ============================================================

FDT_FILE = SET_FILE.with_suffix(".fdt")

print("=" * 70)
print("PO4 INVESTIGATION")
print("=" * 70)

print("\nSET file:")
print(SET_FILE)

print("\nFDT file:")
print(FDT_FILE)

if not SET_FILE.exists():
    raise FileNotFoundError(
        f"\nSET file not found:\n{SET_FILE}"
    )

if not FDT_FILE.exists():
    raise FileNotFoundError(
        f"\nFDT file not found:\n{FDT_FILE}"
    )

print("\nFiles found successfully.")


# ============================================================
# DECODE EEGLAB LABEL
# ============================================================

def decode_label(h5, ref):

    obj = h5[ref]

    arr = np.asarray(
        obj[()]
    ).squeeze()

    # MATLAB uint16 / integer character codes
    if arr.dtype.kind in "iu":

        return "".join(
            chr(int(x))
            for x in arr.flatten()
            if int(x) != 0
        )

    # Byte strings
    if arr.dtype.kind == "S":

        return b"".join(
            arr.flatten()
        ).decode(
            "utf-8",
            errors="ignore"
        )

    return str(arr)


# ============================================================
# READ SET FILE
# ============================================================

print("\nReading SET file...")

with h5py.File(
    SET_FILE,
    "r"
) as h5:

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

    labels = h5["chanlocs"]["labels"]

    channel_names = []

    for i in range(
        labels.shape[0]
    ):

        label = decode_label(
            h5,
            labels[i, 0]
        )

        channel_names.append(
            label
        )


# ============================================================
# BASIC INFORMATION
# ============================================================

duration = pnts / srate

print("\nDATA INFORMATION")
print("-" * 70)

print(
    f"Channels:       {nbchan}"
)

print(
    f"Samples:        {pnts}"
)

print(
    f"Sampling rate:  {srate} Hz"
)

print(
    f"Duration:       {duration:.3f} seconds"
)


# ============================================================
# READ FDT DATA
# ============================================================

print("\nReading FDT data...")

expected_values = nbchan * pnts

file_size = FDT_FILE.stat().st_size

expected_bytes = expected_values * 4

print(
    f"FDT size:       {file_size:,} bytes"
)

print(
    f"Expected size:  {expected_bytes:,} bytes"
)

if file_size != expected_bytes:

    raise ValueError(
        "\nFDT file size does not match "
        "nbchan × pnts × float32."
    )

data = np.fromfile(
    FDT_FILE,
    dtype="<f4"
)

data = data.reshape(
    (nbchan, pnts),
    order="F"
)

print(
    f"Data shape:     {data.shape}"
)

print(
    f"Data dtype:     {data.dtype}"
)


# ============================================================
# FIND TARGET CHANNELS
# ============================================================

targets = [
    "PO3",
    "POZ",
    "PO4",
    "PO8",
    "O1",
    "OZ",
    "O2"
]

indices = {}

for target in targets:

    matches = [
        i
        for i, name in enumerate(channel_names)
        if name.strip().upper() == target
    ]

    if len(matches) > 0:

        indices[target] = matches[0]


# ============================================================
# PRINT CHANNEL INFORMATION
# ============================================================

print("\nTARGET CHANNELS")
print("-" * 70)

for target in targets:

    if target in indices:

        idx = indices[target]

        print(
            f"{target:5s} -> "
            f"channel {idx + 1}"
        )

    else:

        print(
            f"{target:5s} -> NOT FOUND"
        )


# ============================================================
# CHECK PO4 EXISTS
# ============================================================

if "PO4" not in indices:

    raise RuntimeError(
        "\nPO4 channel was not found."
    )


# ============================================================
# CHANNEL STATISTICS
# ============================================================

print("\nCHANNEL STATISTICS")
print("-" * 70)

statistics = {}

for target, idx in indices.items():

    signal = data[idx]

    std = float(
        np.std(signal)
    )

    mean = float(
        np.mean(signal)
    )

    minimum = float(
        np.min(signal)
    )

    maximum = float(
        np.max(signal)
    )

    ptp = float(
        np.ptp(signal)
    )

    statistics[target] = {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "ptp": ptp
    }

    print(
        f"{target:5s} | "
        f"STD={std:10.3f} | "
        f"MIN={minimum:10.3f} | "
        f"MAX={maximum:10.3f} | "
        f"PTP={ptp:10.3f}"
    )


# ============================================================
# PO4 VS NEIGHBORS
# ============================================================

po4_std = statistics["PO4"]["std"]

neighbor_names = [
    "PO3",
    "POZ",
    "PO8"
]

neighbor_stds = [
    statistics[name]["std"]
    for name in neighbor_names
]

median_neighbor_std = float(
    np.median(neighbor_stds)
)

ratio = (
    po4_std /
    median_neighbor_std
)


print("\nPO4 ANALYSIS")
print("-" * 70)

print(
    f"PO4 STD:                    {po4_std:.3f}"
)

print(
    f"Median neighboring STD:     {median_neighbor_std:.3f}"
)

print(
    f"PO4 / neighbor ratio:       {ratio:.3f}"
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\nPRELIMINARY INTERPRETATION")
print("-" * 70)

if ratio < 2:

    print(
        "PO4 variance is relatively close "
        "to neighboring channels."
    )

elif ratio < 5:

    print(
        "PO4 variance is elevated compared "
        "with neighboring channels."
    )

else:

    print(
        "PO4 variance is substantially higher "
        "than neighboring channels."
    )

print(
    "\nIMPORTANT: This is NOT a final decision "
    "to remove PO4."
)


# ============================================================
# PLOT 1
# FIRST 20 SECONDS
# ============================================================

plot_duration = 20

n = min(
    int(plot_duration * srate),
    pnts
)

time = (
    np.arange(n) /
    srate
)

print(
    f"\nCreating first {plot_duration} seconds plot..."
)

plt.figure(
    figsize=(14, 9)
)

offset = 0

for target in targets:

    if target not in indices:
        continue

    idx = indices[target]

    signal = data[
        idx,
        :n
    ].astype(
        np.float64
    )

    signal -= np.mean(
        signal
    )

    scale = max(
        np.std(signal) * 5,
        1
    )

    plt.plot(
        time,
        signal / scale + offset,
        label=target
    )

    offset += 2


plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Normalized amplitude"
)

plt.title(
    "Posterior EEG Channels - First 20 Seconds"
)

plt.legend(
    loc="upper right"
)

plt.tight_layout()

plot1 = (
    OUTPUT_DIR /
    "posterior_first20s.png"
)

plt.savefig(
    plot1,
    dpi=200
)

plt.close()

print(
    f"Saved: {plot1}"
)


# ============================================================
# PLOT 2
# PO4 VS NEIGHBORS
# ============================================================

print(
    "\nCreating PO4 comparison plot..."
)

comparison = [
    "PO3",
    "POZ",
    "PO4",
    "PO8"
]

plt.figure(
    figsize=(14, 8)
)

for target in comparison:

    if target not in indices:
        continue

    idx = indices[target]

    signal = data[
        idx,
        :n
    ].astype(
        np.float64
    )

    signal -= np.mean(
        signal
    )

    plt.plot(
        time,
        signal,
        label=target
    )


plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)

plt.title(
    "PO4 Compared With Neighboring Posterior Channels"
)

plt.legend()

plt.tight_layout()

plot2 = (
    OUTPUT_DIR /
    "po4_vs_neighbors.png"
)

plt.savefig(
    plot2,
    dpi=200
)

plt.close()

print(
    f"Saved: {plot2}"
)


# ============================================================
# PLOT 3
# PO4 ALONE
# ============================================================

print(
    "\nCreating PO4 detailed plot..."
)

po4_signal = data[
    indices["PO4"],
    :n
]

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    time,
    po4_signal
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)

plt.title(
    "PO4 - First 20 Seconds"
)

plt.tight_layout()

plot3 = (
    OUTPUT_DIR /
    "PO4_first20s.png"
)

plt.savefig(
    plot3,
    dpi=200
)

plt.close()

print(
    f"Saved: {plot3}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PO4 INVESTIGATION COMPLETE")
print("=" * 70)

print(
    "\nPO4 STD:",
    round(po4_std, 3)
)

print(
    "Median neighbor STD:",
    round(median_neighbor_std, 3)
)

print(
    "PO4 / neighbor ratio:",
    round(ratio, 3)
)

print("\nPlots saved in:")

print(
    OUTPUT_DIR
)

print("\nDO NOT remove PO4 yet.")
print(
    "Send the terminal output and the plots "
    "for the next QC decision."
)

print("=" * 70)