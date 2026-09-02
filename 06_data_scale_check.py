import os
import glob
import numpy as np
import h5py
import pandas as pd


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
print("EEG DATA SCALE CHECK")
print("=" * 80)

print()
print(
    f"Total .set files found: {len(set_files)}"
)

if len(set_files) == 0:

    raise FileNotFoundError(
        "No .set files found."
    )


# ============================================================
# HELPER FUNCTION
# ============================================================

def read_set_metadata(set_file):

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

        datfile_raw = np.array(
            f["datfile"]
        ).flatten()

        datfile = "".join(
            chr(int(x))
            for x in datfile_raw
            if int(x) != 0
        )

    return (
        nbchan,
        pnts,
        trials,
        srate,
        datfile
    )


# ============================================================
# PROCESS FILES
# ============================================================

results = []


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
            part
            for part in path_parts
            if part.startswith("sub-")
        ),
        "unknown"
    )

    try:

        # ----------------------------------------------------
        # READ METADATA
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
        # FIND FDT
        # ----------------------------------------------------

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

            print(
                "FDT file not found."
            )

            continue

        # ----------------------------------------------------
        # EXPECTED FDT SIZE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # DETECT DTYPE
        # ----------------------------------------------------

        if file_size == expected_float32:

            dtype = np.float32

        elif file_size == expected_float64:

            dtype = np.float64

        else:

            print(
                "WARNING: FDT size mismatch."
            )

            print(
                f"Actual size: "
                f"{file_size}"
            )

            print(
                f"Expected float32: "
                f"{expected_float32}"
            )

            print(
                f"Expected float64: "
                f"{expected_float64}"
            )

            continue

        # ----------------------------------------------------
        # READ DATA
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ONLY EEG CHANNELS
        #
        # First 69 channels are EEG.
        # Channels 70 and 71 are EOG.
        # ----------------------------------------------------

        eeg_data = data[
            :69
        ]

        # ----------------------------------------------------
        # CHANNEL STATISTICS
        # ----------------------------------------------------

        channel_means = np.mean(
            eeg_data,
            axis=1
        )

        channel_stds = np.std(
            eeg_data,
            axis=1
        )

        channel_mins = np.min(
            eeg_data,
            axis=1
        )

        channel_maxs = np.max(
            eeg_data,
            axis=1
        )

        channel_ptp = (
            channel_maxs
            - channel_mins
        )

        # ----------------------------------------------------
        # FILE-LEVEL STATISTICS
        # ----------------------------------------------------

        global_mean = float(
            np.mean(
                eeg_data
            )
        )

        global_std = float(
            np.std(
                eeg_data
            )
        )

        median_std = float(
            np.median(
                channel_stds
            )
        )

        mean_std = float(
            np.mean(
                channel_stds
            )
        )

        max_std = float(
            np.max(
                channel_stds
            )
        )

        min_std = float(
            np.min(
                channel_stds
            )
        )

        median_ptp = float(
            np.median(
                channel_ptp
            )
        )

        max_ptp = float(
            np.max(
                channel_ptp
            )
        )

        global_min = float(
            np.min(
                eeg_data
            )
        )

        global_max = float(
            np.max(
                eeg_data
            )
        )

        # ----------------------------------------------------
        # PERCENTILES
        # ----------------------------------------------------

        p01 = float(
            np.percentile(
                eeg_data,
                1
            )
        )

        p99 = float(
            np.percentile(
                eeg_data,
                99
            )
        )

        # ----------------------------------------------------
        # EXTREME VALUE COUNT
        # ----------------------------------------------------

        extreme_500 = int(
            np.sum(
                np.abs(
                    eeg_data
                ) >= 499.9
            )
        )

        extreme_1000 = int(
            np.sum(
                np.abs(
                    eeg_data
                ) >= 999.9
            )
        )

        # ----------------------------------------------------
        # CHANNELS WITH LARGE STD
        #
        # Relative criterion:
        # > 5 times median channel STD
        # ----------------------------------------------------

        if median_std > 0:

            high_std_channels = int(
                np.sum(
                    channel_stds
                    > 5 * median_std
                )
            )

        else:

            high_std_channels = 0

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"Subject: {subject}"
        )

        print(
            f"Sampling rate: "
            f"{srate:.6f} Hz"
        )

        print(
            f"Channels: {nbchan}"
        )

        print(
            f"Samples: {pnts}"
        )

        print(
            f"Global mean: "
            f"{global_mean:.4f}"
        )

        print(
            f"Global STD: "
            f"{global_std:.4f}"
        )

        print(
            f"Median channel STD: "
            f"{median_std:.4f}"
        )

        print(
            f"Maximum channel STD: "
            f"{max_std:.4f}"
        )

        print(
            f"Median channel PTP: "
            f"{median_ptp:.4f}"
        )

        print(
            f"Global min: "
            f"{global_min:.4f}"
        )

        print(
            f"Global max: "
            f"{global_max:.4f}"
        )

        print(
            f"Values >= |499.9|: "
            f"{extreme_500}"
        )

        print(
            f"Values >= |999.9|: "
            f"{extreme_1000}"
        )

        print(
            f"Channels > 5x median STD: "
            f"{high_std_channels}"
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append({

            "subject":
                subject,

            "file":
                filename,

            "sampling_rate":
                srate,

            "channels":
                nbchan,

            "samples":
                pnts,

            "trials":
                trials,

            "global_mean":
                global_mean,

            "global_std":
                global_std,

            "median_channel_std":
                median_std,

            "mean_channel_std":
                mean_std,

            "min_channel_std":
                min_std,

            "max_channel_std":
                max_std,

            "median_channel_ptp":
                median_ptp,

            "max_channel_ptp":
                max_ptp,

            "global_min":
                global_min,

            "global_max":
                global_max,

            "percentile_1":
                p01,

            "percentile_99":
                p99,

            "values_abs_ge_499_9":
                extreme_500,

            "values_abs_ge_999_9":
                extreme_1000,

            "channels_gt_5x_median_std":
                high_std_channels,

            "fdt_dtype":
                str(dtype)

        })

        # ----------------------------------------------------
        # RELEASE MEMMAP
        # ----------------------------------------------------

        del data

    except Exception as e:

        print()
        print(
            f"ERROR: {repr(e)}"
        )

        results.append({

            "subject":
                subject,

            "file":
                filename,

            "error":
                repr(e)

        })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    results
)


# ============================================================
# SAVE REPORT
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "data_scale_report.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY BY SAMPLING RATE
# ============================================================

print()
print("=" * 80)

print(
    "SCALE SUMMARY BY SAMPLING RATE"
)

print("=" * 80)

print()

if "sampling_rate" in df.columns:

    summary = (
        df.groupby(
            "sampling_rate"
        )[
            [
                "median_channel_std",
                "median_channel_ptp",
                "global_std"
            ]
        ]
        .agg(
            [
                "count",
                "median",
                "min",
                "max"
            ]
        )
    )

    print(
        summary.to_string()
    )


# ============================================================
# FIND FILES WITH EXTREME VALUES
# ============================================================

print()
print("=" * 80)

print(
    "FILES WITH EXTREME VALUES"
)

print("=" * 80)

print()

if "values_abs_ge_499_9" in df.columns:

    extreme_files = df[
        df[
            "values_abs_ge_499_9"
        ] > 0
    ]

    if len(
        extreme_files
    ) == 0:

        print(
            "No files with values near +/-500."
        )

    else:

        print(
            extreme_files[
                [
                    "subject",
                    "file",
                    "sampling_rate",
                    "median_channel_std",
                    "global_min",
                    "global_max",
                    "values_abs_ge_499_9"
                ]
            ].to_string(
                index=False
            )
        )


# ============================================================
# FIND HIGH-SCALE FILES
# ============================================================

print()
print("=" * 80)

print(
    "FILES WITH HIGHEST MEDIAN CHANNEL STD"
)

print("=" * 80)

print()

if "median_channel_std" in df.columns:

    high_scale = (
        df.sort_values(
            "median_channel_std",
            ascending=False
        )
        .head(15)
    )

    print(
        high_scale[
            [
                "subject",
                "file",
                "sampling_rate",
                "median_channel_std",
                "median_channel_ptp"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)

print(
    "DATA SCALE CHECK COMPLETE"
)

print("=" * 80)

print()

print(
    "Report saved to:"
)

print(
    output_file
)

print()
print(
    "DONE."
)