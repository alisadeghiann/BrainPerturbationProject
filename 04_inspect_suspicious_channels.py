import os
import numpy as np
import h5py
import matplotlib.pyplot as plt


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
    "qc",
    "suspicious_channels"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TARGET FILE
# ============================================================

subject = "sub-018"

run = "run-1"

filename = (
    "sub-018_ses-01_task-WorkingMemory_run-1_eeg.set"
)

set_file = os.path.join(
    DATA_DIR,
    subject,
    "ses-01",
    "eeg",
    filename
)


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(
    set_file
):

    raise FileNotFoundError(
        f"SET file not found:\n{set_file}"
    )


# ============================================================
# READ HDF5 STRING
# ============================================================

def read_hdf5_string(
    dataset
):

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
# READ METADATA
# ============================================================

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


# ============================================================
# FIND FDT
# ============================================================

fdt_file = os.path.join(
    os.path.dirname(
        set_file
    ),
    datfile
)

if not os.path.exists(
    fdt_file
):

    fdt_file = (
        os.path.splitext(
            set_file
        )[0]
        + ".fdt"
    )


if not os.path.exists(
    fdt_file
):

    raise FileNotFoundError(
        f"FDT file not found:\n{fdt_file}"
    )


# ============================================================
# READ FDT
# ============================================================

expected_values = (
    nbchan
    * pnts
    * trials
)

file_size = os.path.getsize(
    fdt_file
)

if file_size == expected_values * 4:

    dtype = np.float32

elif file_size == expected_values * 8:

    dtype = np.float64

else:

    raise ValueError(
        "FDT size does not match expected data size."
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


# ============================================================
# TARGET CHANNEL
# ============================================================

channel_index = 64

channel_name = "PO4"

channel = data[
    channel_index - 1
]


# ============================================================
# BASIC STATISTICS
# ============================================================

mean_value = np.mean(
    channel
)

std_value = np.std(
    channel
)

min_value = np.min(
    channel
)

max_value = np.max(
    channel
)

ptp_value = (
    max_value
    - min_value
)

percentile_1 = np.percentile(
    channel,
    1
)

percentile_99 = np.percentile(
    channel,
    99
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 80)

print(
    "SUSPICIOUS CHANNEL INSPECTION"
)

print("=" * 80)

print()

print(
    f"Subject: {subject}"
)

print(
    f"Run: {run}"
)

print(
    f"Channel: {channel_index}"
)

print(
    f"Name: {channel_name}"
)

print()

print(
    f"Sampling rate: {srate} Hz"
)

print(
    f"Samples: {len(channel)}"
)

print(
    f"Duration: {len(channel) / srate:.2f} sec"
)

print()

print(
    f"Mean: {mean_value:.6f}"
)

print(
    f"STD: {std_value:.6f}"
)

print(
    f"Min: {min_value:.6f}"
)

print(
    f"Max: {max_value:.6f}"
)

print(
    f"Peak-to-peak: {ptp_value:.6f}"
)

print(
    f"1st percentile: {percentile_1:.6f}"
)

print(
    f"99th percentile: {percentile_99:.6f}"
)


# ============================================================
# COMPARE WITH OTHER EEG CHANNELS
# ============================================================

eeg_data = data[
    :69
]

channel_stds = np.std(
    eeg_data,
    axis=1
)

median_std = np.median(
    channel_stds
)

print()

print(
    "=" * 80
)

print(
    "COMPARISON WITH OTHER EEG CHANNELS"
)

print(
    "=" * 80
)

print()

print(
    f"Median EEG channel STD: "
    f"{median_std:.6f}"
)

print(
    f"PO4 / median STD: "
    f"{std_value / median_std:.2f}"
)


# ============================================================
# RANK CHANNELS BY STD
# ============================================================

sorted_indices = np.argsort(
    channel_stds
)[::-1]

print()

print(
    "TOP 10 CHANNELS BY STD:"
)

print()

for rank, idx in enumerate(
    sorted_indices[:10],
    start=1
):

    print(
        f"{rank:2d}. "
        f"Channel {idx + 1:2d} "
        f"STD = "
        f"{channel_stds[idx]:.6f}"
    )


# ============================================================
# PLOT 1 — FULL SIGNAL
# ============================================================

time = (
    np.arange(
        len(channel)
    )
    / srate
)

plt.figure(
    figsize=(16, 5)
)

plt.plot(
    time,
    channel
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)

plt.title(
    "sub-018 Run-1 — PO4 Full Signal"
)

plt.tight_layout()

full_plot = os.path.join(
    OUTPUT_DIR,
    "sub-018_run-1_PO4_full.png"
)

plt.savefig(
    full_plot,
    dpi=150
)

plt.close()


# ============================================================
# PLOT 2 — FIRST 30 SECONDS
# ============================================================

seconds = 30

samples = int(
    seconds * srate
)

samples = min(
    samples,
    len(channel)
)

plt.figure(
    figsize=(16, 5)
)

plt.plot(
    time[:samples],
    channel[:samples]
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)

plt.title(
    "sub-018 Run-1 — PO4 First 30 Seconds"
)

plt.tight_layout()

first_plot = os.path.join(
    OUTPUT_DIR,
    "sub-018_run-1_PO4_first30s.png"
)

plt.savefig(
    first_plot,
    dpi=150
)

plt.close()


# ============================================================
# PLOT 3 — LAST 30 SECONDS
# ============================================================

start = max(
    0,
    len(channel) - samples
)

plt.figure(
    figsize=(16, 5)
)

plt.plot(
    time[start:],
    channel[start:]
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Amplitude"
)

plt.title(
    "sub-018 Run-1 — PO4 Last 30 Seconds"
)

plt.tight_layout()

last_plot = os.path.join(
    OUTPUT_DIR,
    "sub-018_run-1_PO4_last30s.png"
)

plt.savefig(
    last_plot,
    dpi=150
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print()

print(
    "=" * 80
)

print(
    "INSPECTION COMPLETE"
)

print(
    "=" * 80
)

print()

print(
    "Plots saved to:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "DONE."
)