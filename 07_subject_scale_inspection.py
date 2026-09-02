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
    "qc",
    "subject_scale_inspection"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SUBJECTS TO INSPECT
# ============================================================

SUSPICIOUS_SUBJECTS = [
    "sub-002",
    "sub-003",
    "sub-004",
    "sub-008",
    "sub-009",
    "sub-012",
    "sub-014",
    "sub-018",
    "sub-024"
]


# ============================================================
# FIND FILES
# ============================================================

all_files = sorted(
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

set_files = [
    f
    for f in all_files
    if any(
        subject in f
        for subject in SUSPICIOUS_SUBJECTS
    )
]


print("=" * 80)
print("SUBJECT SCALE INSPECTION")
print("=" * 80)

print()
print(
    f"Total suspicious files found: "
    f"{len(set_files)}"
)

print()


# ============================================================
# CHANNEL NAME EXTRACTION
# ============================================================

def decode_hdf5_string(
    file_handle,
    dataset_path
):

    dataset = file_handle[
        dataset_path
    ]

    values = np.array(
        dataset
    ).flatten()

    names = []

    for value in values:

        try:

            value = int(value)

            if value == 0:
                names.append("")
                continue

            # MATLAB HDF5 references
            if value in file_handle:

                obj = file_handle[
                    value
                ]

                if isinstance(
                    obj,
                    h5py.Dataset
                ):

                    chars = np.array(
                        obj
                    ).flatten()

                    text = "".join(
                        chr(int(x))
                        for x in chars
                        if int(x) != 0
                    )

                    names.append(
                        text
                    )

                else:

                    names.append(
                        str(value)
                    )

            else:

                names.append(
                    chr(value)
                )

        except:

            names.append(
                str(value)
            )

    return names


# ============================================================
# READ METADATA
# ============================================================

def read_metadata(
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

        datfile_raw = np.array(
            f["datfile"]
        ).flatten()

        datfile = "".join(
            chr(int(x))
            for x in datfile_raw
            if int(x) != 0
        )

        # ----------------------------------------------------
        # CHANNEL LABELS
        # ----------------------------------------------------

        labels = []

        if (
            "chanlocs" in f
            and "labels" in f["chanlocs"]
        ):

            label_dataset = f[
                "chanlocs/labels"
            ]

            label_values = np.array(
                label_dataset
            ).flatten()

            for value in label_values:

                try:

                    if isinstance(
                        value,
                        np.ndarray
                    ):

                        value = value.item()

                    if isinstance(
                        value,
                        (bytes, np.bytes_)
                    ):

                        label = value.decode(
                            "utf-8",
                            errors="ignore"
                        )

                    else:

                        value = int(value)

                        if value in f:

                            obj = f[value]

                            if isinstance(
                                obj,
                                h5py.Dataset
                            ):

                                chars = np.array(
                                    obj
                                ).flatten()

                                label = "".join(
                                    chr(int(x))
                                    for x in chars
                                    if int(x) != 0
                                )

                            else:

                                label = str(
                                    value
                                )

                        else:

                            label = chr(
                                value
                            )

                    labels.append(
                        label
                    )

                except:

                    labels.append(
                        ""
                    )

    return (
        nbchan,
        pnts,
        trials,
        srate,
        datfile,
        labels
    )


# ============================================================
# RESULTS
# ============================================================

all_results = []


# ============================================================
# PROCESS EACH FILE
# ============================================================

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

    subject = next(
        (
            part
            for part in set_file.split(
                os.sep
            )
            if part.startswith("sub-")
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

        (
            nbchan,
            pnts,
            trials,
            srate,
            datfile,
            labels
        ) = read_metadata(
            set_file
        )

        # ----------------------------------------------------
        # FDT PATH
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
                "ERROR: FDT file not found."
            )

            continue

        # ----------------------------------------------------
        # DETERMINE FDT DTYPE
        # ----------------------------------------------------

        expected_values = (
            nbchan
            * pnts
            * trials
        )

        file_size = os.path.getsize(
            fdt_file
        )

        if (
            file_size
            == expected_values * 4
        ):

            dtype = np.float32

        elif (
            file_size
            == expected_values * 8
        ):

            dtype = np.float64

        else:

            print(
                "ERROR: FDT size mismatch."
            )

            continue

        # ----------------------------------------------------
        # MEMORY MAP
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
        # EEG CHANNELS
        # ----------------------------------------------------

        eeg_count = min(
            69,
            nbchan
        )

        eeg_data = data[
            :eeg_count
        ]

        # ----------------------------------------------------
        # CHANNEL STATISTICS
        # ----------------------------------------------------

        channel_mean = np.mean(
            eeg_data,
            axis=1
        )

        channel_std = np.std(
            eeg_data,
            axis=1
        )

        channel_min = np.min(
            eeg_data,
            axis=1
        )

        channel_max = np.max(
            eeg_data,
            axis=1
        )

        channel_ptp = (
            channel_max
            - channel_min
        )

        # ----------------------------------------------------
        # MEDIAN STD
        # ----------------------------------------------------

        median_std = np.median(
            channel_std
        )

        # ----------------------------------------------------
        # PRINT CHANNEL INFORMATION
        # ----------------------------------------------------

        print()

        print(
            "CHANNEL STATISTICS"
        )

        print(
            "-" * 80
        )

        for i in range(
            eeg_count
        ):

            channel_number = i + 1

            if (
                i < len(labels)
                and labels[i] != ""
            ):

                channel_name = labels[i]

            else:

                channel_name = (
                    f"CH{channel_number}"
                )

            ratio = (
                channel_std[i]
                / median_std
                if median_std > 0
                else np.nan
            )

            print(
                f"{channel_number:3d} | "
                f"{channel_name:<8} | "
                f"STD={channel_std[i]:10.3f} | "
                f"PTP={channel_ptp[i]:10.3f} | "
                f"MIN={channel_min[i]:10.3f} | "
                f"MAX={channel_max[i]:10.3f} | "
                f"RATIO={ratio:7.2f}"
            )

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            all_results.append({

                "subject":
                    subject,

                "file":
                    filename,

                "sampling_rate":
                    srate,

                "channel_index":
                    channel_number,

                "channel_name":
                    channel_name,

                "mean":
                    float(
                        channel_mean[i]
                    ),

                "std":
                    float(
                        channel_std[i]
                    ),

                "min":
                    float(
                        channel_min[i]
                    ),

                "max":
                    float(
                        channel_max[i]
                    ),

                "ptp":
                    float(
                        channel_ptp[i]
                    ),

                "std_ratio_to_median":
                    float(
                        ratio
                    ),

                "near_minus_500":
                    int(
                        np.sum(
                            eeg_data[i]
                            <= -499.9
                        )
                    ),

                "near_plus_500":
                    int(
                        np.sum(
                            eeg_data[i]
                            >= 499.9
                        )
                    ),

                "near_minus_1000":
                    int(
                        np.sum(
                            eeg_data[i]
                            <= -999.9
                        )
                    ),

                "near_plus_1000":
                    int(
                        np.sum(
                            eeg_data[i]
                            >= 999.9
                        )
                    )

            })

        del data

    except Exception as e:

        print()
        print(
            f"ERROR: {repr(e)}"
        )


# ============================================================
# SAVE CHANNEL REPORT
# ============================================================

df = pd.DataFrame(
    all_results
)

channel_report = os.path.join(
    OUTPUT_DIR,
    "subject_scale_channel_report.csv"
)

df.to_csv(
    channel_report,
    index=False
)


# ============================================================
# CREATE FILE-LEVEL SUMMARY
# ============================================================

if len(df) > 0:

    file_summary = (
        df.groupby(
            [
                "subject",
                "file",
                "sampling_rate"
            ]
        )
        .agg(
            median_std=(
                "std",
                "median"
            ),
            max_std=(
                "std",
                "max"
            ),
            median_ptp=(
                "ptp",
                "median"
            ),
            max_ptp=(
                "ptp",
                "max"
            ),
            min_value=(
                "min",
                "min"
            ),
            max_value=(
                "max",
                "max"
            ),
            channels_over_5x=(
                "std_ratio_to_median",
                lambda x:
                int(
                    np.sum(
                        x > 5
                    )
                )
            ),
            channels_over_10x=(
                "std_ratio_to_median",
                lambda x:
                int(
                    np.sum(
                        x > 10
                    )
                )
            ),
            samples_near_1000=(
                "near_minus_1000",
                "sum"
            )
        )
        .reset_index()
    )

else:

    file_summary = pd.DataFrame()


file_summary_path = os.path.join(
    OUTPUT_DIR,
    "subject_scale_file_summary.csv"
)

file_summary.to_csv(
    file_summary_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)

print(
    "SUBJECT SCALE INSPECTION COMPLETE"
)

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
    "File-level report:"
)

print(
    file_summary_path
)

print()

print(
    "DONE."
)