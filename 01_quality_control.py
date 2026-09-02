import os
import glob
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

QC_DIR = os.path.join(PROJECT_ROOT, "qc")

PLOTS_DIR = os.path.join(QC_DIR, "plots")

os.makedirs(QC_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# FIND ALL EEG .SET FILES
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
print("EEG QUALITY CONTROL")
print("=" * 80)

print()
print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")
print(f"Number of .set files found: {len(set_files)}")

if len(set_files) == 0:
    raise FileNotFoundError(
        "No .set files were found."
    )


# ============================================================
# READ MATLAB STRING FROM HDF5
# ============================================================

def read_hdf5_string(dataset):

    arr = np.array(dataset).astype(np.uint16).flatten()

    return "".join(
        chr(int(x))
        for x in arr
        if int(x) != 0
    )


# ============================================================
# READ SET METADATA
# ============================================================

def read_set_metadata(set_file):

    with h5py.File(set_file, "r") as f:

        nbchan = int(
            np.array(f["nbchan"]).squeeze()
        )

        pnts = int(
            np.array(f["pnts"]).squeeze()
        )

        trials = int(
            np.array(f["trials"]).squeeze()
        )

        srate = float(
            np.array(f["srate"]).squeeze()
        )

        xmin = float(
            np.array(f["xmin"]).squeeze()
        )

        xmax = float(
            np.array(f["xmax"]).squeeze()
        )

        datfile = read_hdf5_string(
            f["datfile"]
        )

        filename = read_hdf5_string(
            f["filename"]
        )

        filepath = read_hdf5_string(
            f["filepath"]
        )

    return {
        "nbchan": nbchan,
        "pnts": pnts,
        "trials": trials,
        "srate": srate,
        "xmin": xmin,
        "xmax": xmax,
        "datfile": datfile,
        "filename": filename,
        "filepath": filepath
    }


# ============================================================
# FIND FDT FILE
# ============================================================

def find_fdt_file(set_file, datfile):

    eeg_directory = os.path.dirname(
        set_file
    )

    # First: use datfile name directly
    candidate = os.path.join(
        eeg_directory,
        datfile
    )

    if os.path.exists(candidate):
        return candidate

    # Second: replace .set with .fdt
    candidate = os.path.splitext(
        set_file
    )[0] + ".fdt"

    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"FDT file not found for:\n{set_file}\n"
        f"Expected:\n{candidate}"
    )


# ============================================================
# READ EEG FROM FDT
# ============================================================

def read_fdt_data(
    fdt_file,
    nbchan,
    pnts,
    trials
):

    print(
        f"Reading FDT data..."
    )

    file_size = os.path.getsize(
        fdt_file
    )

    print(
        f"FDT file size: "
        f"{file_size / (1024 ** 2):.2f} MB"
    )

    expected_values = (
        nbchan
        * pnts
        * trials
    )

    expected_bytes_float32 = (
        expected_values * 4
    )

    expected_bytes_float64 = (
        expected_values * 8
    )

    print(
        f"Expected float32 size: "
        f"{expected_bytes_float32 / (1024 ** 2):.2f} MB"
    )

    print(
        f"Expected float64 size: "
        f"{expected_bytes_float64 / (1024 ** 2):.2f} MB"
    )

    # --------------------------------------------------------
    # EEGLAB .fdt is normally float32
    # --------------------------------------------------------

    if file_size == expected_bytes_float32:

        dtype = np.float32

        print(
            "Detected FDT dtype: float32"
        )

    elif file_size == expected_bytes_float64:

        dtype = np.float64

        print(
            "Detected FDT dtype: float64"
        )

    else:

        raise ValueError(
            "\nFDT file size does not match "
            "expected EEG dimensions.\n"
            f"File size: {file_size} bytes\n"
            f"Expected float32: "
            f"{expected_bytes_float32} bytes\n"
            f"Expected float64: "
            f"{expected_bytes_float64} bytes\n"
        )

    # --------------------------------------------------------
    # Memory-map FDT
    # --------------------------------------------------------

    data = np.memmap(
        fdt_file,
        dtype=dtype,
        mode="r",
        shape=(nbchan, pnts * trials),
        order="F"
    )

    return data


# ============================================================
# CALCULATE QC METRICS
# ============================================================

def calculate_qc_metrics(
    data,
    srate
):

    metrics = {}

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    metrics["global_mean"] = float(
        np.mean(data)
    )

    metrics["global_std"] = float(
        np.std(data)
    )

    metrics["global_min"] = float(
        np.min(data)
    )

    metrics["global_max"] = float(
        np.max(data)
    )

    # --------------------------------------------------------
    # Per-channel statistics
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
        channel_max - channel_min
    )

    metrics["median_channel_std"] = float(
        np.median(channel_std)
    )

    metrics["max_channel_std"] = float(
        np.max(channel_std)
    )

    metrics["median_channel_ptp"] = float(
        np.median(channel_ptp)
    )

    metrics["max_channel_ptp"] = float(
        np.max(channel_ptp)
    )

    # --------------------------------------------------------
    # Flat channels
    # --------------------------------------------------------

    flat_threshold = 1e-12

    flat_channels = np.where(
        channel_std < flat_threshold
    )[0]

    metrics["n_flat_channels"] = int(
        len(flat_channels)
    )

    metrics["flat_channels"] = (
        flat_channels.tolist()
    )

    # --------------------------------------------------------
    # Extremely high amplitude channels
    # --------------------------------------------------------

    median_std = np.median(
        channel_std
    )

    if median_std > 0:

        high_amp_threshold = (
            median_std * 10
        )

        high_amp_channels = np.where(
            channel_std > high_amp_threshold
        )[0]

    else:

        high_amp_channels = np.array(
            [],
            dtype=int
        )

    metrics[
        "n_high_amplitude_channels"
    ] = int(
        len(high_amp_channels)
    )

    metrics[
        "high_amplitude_channels"
    ] = high_amp_channels.tolist()

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    metrics["duration_seconds"] = (
        data.shape[1] / srate
    )

    # --------------------------------------------------------
    # Channel means
    # --------------------------------------------------------

    metrics[
        "median_channel_mean"
    ] = float(
        np.median(channel_mean)
    )

    return metrics


# ============================================================
# CREATE QC PLOTS
# ============================================================

def create_qc_plots(
    data,
    srate,
    subject,
    filename
):

    base_name = os.path.splitext(
        filename
    )[0]

    # ========================================================
    # 1. CHANNEL STANDARD DEVIATION
    # ========================================================

    channel_std = np.std(
        data,
        axis=1
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        np.arange(1, len(channel_std) + 1),
        channel_std
    )

    plt.xlabel(
        "Channel"
    )

    plt.ylabel(
        "Standard deviation"
    )

    plt.title(
        f"Channel Variability\n"
        f"{subject} - {filename}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_file = os.path.join(
        PLOTS_DIR,
        f"{base_name}_channel_variability.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    # ========================================================
    # 2. EEG TRACES
    # ========================================================

    max_points = 20000

    step = max(
        1,
        data.shape[1] // max_points
    )

    plot_data = data[
        :,
        ::step
    ]

    times = (
        np.arange(
            plot_data.shape[1]
        )
        * step
        / srate
    )

    n_plot_channels = min(
        10,
        data.shape[0]
    )

    plt.figure(
        figsize=(14, 10)
    )

    offset = 0

    for ch in range(
        n_plot_channels
    ):

        signal = np.asarray(
            plot_data[ch]
        )

        signal_std = np.std(
            signal
        )

        if signal_std > 0:

            signal = (
                signal / signal_std
            )

        signal = (
            signal + offset
        )

        plt.plot(
            times,
            signal
        )

        offset += 5

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Normalized amplitude"
    )

    plt.title(
        f"First {n_plot_channels} EEG Channels\n"
        f"{subject} - {filename}"
    )

    plt.tight_layout()

    output_file = os.path.join(
        PLOTS_DIR,
        f"{base_name}_raw_traces.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()


# ============================================================
# PROCESS ALL EEG FILES
# ============================================================

results = []


for file_number, set_file in enumerate(
    set_files,
    start=1
):

    print()
    print("=" * 80)

    print(
        f"PROCESSING FILE "
        f"{file_number}/{len(set_files)}"
    )

    print("=" * 80)

    print(
        f"File: {set_file}"
    )

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    path_parts = set_file.split(
        os.sep
    )

    subject = next(
        (
            part
            for part in path_parts
            if part.startswith("sub-")
        ),
        "unknown"
    )

    filename = os.path.basename(
        set_file
    )

    print(
        f"Subject: {subject}"
    )

    try:

        # ====================================================
        # READ SET METADATA
        # ====================================================

        metadata = read_set_metadata(
            set_file
        )

        nbchan = metadata[
            "nbchan"
        ]

        pnts = metadata[
            "pnts"
        ]

        trials = metadata[
            "trials"
        ]

        srate = metadata[
            "srate"
        ]

        datfile = metadata[
            "datfile"
        ]

        print()
        print(
            f"Channels: {nbchan}"
        )

        print(
            f"Samples: {pnts}"
        )

        print(
            f"Trials: {trials}"
        )

        print(
            f"Sampling rate: {srate} Hz"
        )

        print(
            f"FDT file: {datfile}"
        )

        # ====================================================
        # FIND FDT
        # ====================================================

        fdt_file = find_fdt_file(
            set_file,
            datfile
        )

        print(
            f"FDT path: {fdt_file}"
        )

        # ====================================================
        # READ REAL EEG
        # ====================================================

        data = read_fdt_data(
            fdt_file,
            nbchan,
            pnts,
            trials
        )

        print(
            f"EEG data shape: "
            f"{data.shape}"
        )

        # ====================================================
        # QC
        # ====================================================

        metrics = calculate_qc_metrics(
            data,
            srate
        )

        print()
        print(
            f"Duration: "
            f"{metrics['duration_seconds']:.2f} sec"
        )

        print(
            f"Global mean: "
            f"{metrics['global_mean']:.4f}"
        )

        print(
            f"Global std: "
            f"{metrics['global_std']:.4f}"
        )

        print(
            f"Median channel std: "
            f"{metrics['median_channel_std']:.4f}"
        )

        print(
            f"Max channel std: "
            f"{metrics['max_channel_std']:.4f}"
        )

        print(
            f"Flat channels: "
            f"{metrics['n_flat_channels']}"
        )

        print(
            f"High amplitude channels: "
            f"{metrics['n_high_amplitude_channels']}"
        )

        # ====================================================
        # PLOTS
        # ====================================================

        create_qc_plots(
            data=data,
            srate=srate,
            subject=subject,
            filename=filename
        )

        # ====================================================
        # SAVE RESULT
        # ====================================================

        results.append({

            "subject": subject,

            "file": filename,

            "fdt_file": os.path.basename(
                fdt_file
            ),

            "n_channels": nbchan,

            "n_samples": pnts,

            "trials": trials,

            "sampling_rate": srate,

            "duration_seconds":
                metrics[
                    "duration_seconds"
                ],

            "global_mean":
                metrics[
                    "global_mean"
                ],

            "global_std":
                metrics[
                    "global_std"
                ],

            "global_min":
                metrics[
                    "global_min"
                ],

            "global_max":
                metrics[
                    "global_max"
                ],

            "median_channel_std":
                metrics[
                    "median_channel_std"
                ],

            "max_channel_std":
                metrics[
                    "max_channel_std"
                ],

            "median_channel_ptp":
                metrics[
                    "median_channel_ptp"
                ],

            "max_channel_ptp":
                metrics[
                    "max_channel_ptp"
                ],

            "n_flat_channels":
                metrics[
                    "n_flat_channels"
                ],

            "flat_channels":
                str(
                    metrics[
                        "flat_channels"
                    ]
                ),

            "n_high_amplitude_channels":
                metrics[
                    "n_high_amplitude_channels"
                ],

            "high_amplitude_channels":
                str(
                    metrics[
                        "high_amplitude_channels"
                    ]
                )

        })

    except Exception as e:

        print()
        print(
            "ERROR PROCESSING FILE"
        )

        print(
            repr(e)
        )

        results.append({

            "subject": subject,

            "file": filename,

            "error": repr(e)

        })


# ============================================================
# SAVE QC SUMMARY
# ============================================================

df = pd.DataFrame(
    results
)

csv_path = os.path.join(
    QC_DIR,
    "quality_control_summary.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("QUALITY CONTROL FINISHED")
print("=" * 80)

print()
print(
    f"Files processed: {len(results)}"
)

print()
print(
    f"QC summary saved to:"
)

print(
    csv_path
)

print()
print(
    f"Plots saved to:"
)

print(
    PLOTS_DIR
)

print()
print("QC COMPLETE.")