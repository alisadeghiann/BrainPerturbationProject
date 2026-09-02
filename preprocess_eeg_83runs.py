import os
import re
import glob
import json
import traceback

import h5py
import numpy as np
import pandas as pd
import mne


# ============================================================
# CONFIG
# ============================================================

PROJECT = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

DATA_DIR = os.path.join(PROJECT, "data")

QC_DIR = os.path.join(PROJECT, "qc", "final_qc")

CHANNEL_QC = os.path.join(
    QC_DIR,
    "final_channel_qc_83runs.csv"
)

RUN_QC = os.path.join(
    QC_DIR,
    "final_run_qc_83runs.csv"
)

OUT_DIR = os.path.join(
    PROJECT,
    "preprocessed"
)

LOG_DIR = os.path.join(
    OUT_DIR,
    "logs"
)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# PREPROCESSING PARAMETERS
# ============================================================

LOW_CUT = 0.1
HIGH_CUT = 40.0

NOTCH_FREQ = 50.0

# Important:
# Do NOT modify raw data.
# All processing is performed in memory and saved separately.


# ============================================================
# HELPERS
# ============================================================

def matlab_scalar(value):
    """
    Convert an HDF5 MATLAB scalar/object into Python scalar.
    """
    arr = np.asarray(value)

    while arr.size == 1 and arr.dtype == object:
        try:
            arr = np.asarray(arr.flat[0])
        except Exception:
            break

    if arr.size == 1:
        try:
            return float(arr.reshape(-1)[0])
        except Exception:
            pass

    return arr


def decode_numeric_char_array(arr):
    """
    Decode MATLAB uint16/uint8 character arrays.
    """
    arr = np.asarray(arr).reshape(-1)

    chars = []

    for x in arr:
        try:
            v = int(x)
            if v != 0:
                chars.append(chr(v))
        except Exception:
            pass

    return "".join(chars)


def dereference_scalar(h5file, obj):
    """
    Dereference a MATLAB HDF5 object reference.
    """
    if isinstance(obj, h5py.Reference):
        return np.asarray(h5file[obj])

    return np.asarray(obj)


def extract_matlab_string(h5file, dataset):
    """
    Extract MATLAB string/char/object from EEGLAB HDF5.
    """
    try:
        arr = np.asarray(dataset)

        # Direct numeric char array
        if arr.dtype != object:
            if np.issubdtype(arr.dtype, np.integer):
                text = decode_numeric_char_array(arr)
                if text:
                    return text

        # Object reference
        flat = arr.reshape(-1)

        if len(flat) > 0:
            obj = flat[0]

            if isinstance(obj, h5py.Reference):
                data = np.asarray(h5file[obj])

                if data.dtype != object:
                    if np.issubdtype(data.dtype, np.integer):
                        text = decode_numeric_char_array(data)
                        if text:
                            return text

                # recursive attempt
                try:
                    return extract_matlab_string(h5file, h5file[obj])
                except Exception:
                    pass

    except Exception:
        pass

    return ""


def extract_channel_labels(h5file):
    """
    Extract 71 channel labels from EEGLAB chanlocs.
    """

    chanlocs = h5file["chanlocs"]

    labels_ds = chanlocs["labels"]

    labels = []

    arr = np.asarray(labels_ds).reshape(-1)

    for item in arr:

        try:
            if isinstance(item, h5py.Reference):

                obj = h5file[item]

                data = np.asarray(obj)

                if data.dtype != object:
                    label = decode_numeric_char_array(data)
                else:
                    label = extract_matlab_string(h5file, obj)

            else:
                data = np.asarray(item)

                if data.dtype.kind in ["U", "S"]:
                    label = str(data)
                elif np.issubdtype(data.dtype, np.integer):
                    label = decode_numeric_char_array(data)
                else:
                    label = ""

        except Exception:
            label = ""

        label = label.strip()

        labels.append(label)

    # fallback if labels weren't decoded correctly
    if len(labels) != 71 or any(x == "" for x in labels):
        print("WARNING: Some channel labels could not be decoded.")
        print("Decoded labels:", labels)

    return labels


def extract_scalar_from_dataset(h5file, name):
    """
    Extract scalar MATLAB field such as srate, pnts, trials.
    """
    obj = h5file[name]

    try:
        arr = np.asarray(obj)

        if arr.dtype == object:

            ref = arr.reshape(-1)[0]

            if isinstance(ref, h5py.Reference):
                arr = np.asarray(h5file[ref])

        return float(np.asarray(arr).reshape(-1)[0])

    except Exception as e:
        raise RuntimeError(
            f"Could not extract {name}: {e}"
        )


def load_fdt_and_metadata(set_path):
    """
    Load EEG directly from the FDT binary referenced by the SET file.

    Returns:
        data       shape = channels x samples
        labels
        srate
        pnts
        trials
        xmin
        xmax
    """

    with h5py.File(set_path, "r") as f:

        srate = extract_scalar_from_dataset(f, "srate")
        pnts = int(round(extract_scalar_from_dataset(f, "pnts")))
        trials = int(round(extract_scalar_from_dataset(f, "trials")))

        xmin = extract_scalar_from_dataset(f, "xmin")
        xmax = extract_scalar_from_dataset(f, "xmax")

        labels = extract_channel_labels(f)

        datfile_text = extract_matlab_string(
            f,
            f["datfile"]
        )

    if not datfile_text:
        raise RuntimeError(
            f"Could not determine FDT filename from:\n{set_path}"
        )

    # FDT should be next to SET
    set_dir = os.path.dirname(set_path)

    fdt_name = os.path.basename(datfile_text)

    fdt_path = os.path.join(
        set_dir,
        fdt_name
    )

    if not os.path.exists(fdt_path):

        # Sometimes datfile may contain a relative path
        candidate = os.path.join(
            set_dir,
            datfile_text
        )

        if os.path.exists(candidate):
            fdt_path = candidate
        else:
            raise FileNotFoundError(
                f"FDT file not found.\n"
                f"SET: {set_path}\n"
                f"Expected: {fdt_path}"
            )

    print(f"FDT: {fdt_path}")

    expected_channels = len(labels)

    if expected_channels == 0:
        raise RuntimeError(
            "No channel labels were extracted."
        )

    expected_values = expected_channels * pnts

    file_size = os.path.getsize(fdt_path)

    expected_bytes_float32 = expected_values * 4

    if file_size != expected_bytes_float32:

        print("WARNING: FDT size does not match float32 expectation.")

        print("File size:", file_size)
        print("Expected:", expected_bytes_float32)

        # Try float64 if appropriate
        expected_bytes_float64 = expected_values * 8

        if file_size == expected_bytes_float64:

            dtype = np.float64

        else:

            raise RuntimeError(
                "FDT size is inconsistent with metadata.\n"
                f"Expected float32 bytes: {expected_bytes_float32}\n"
                f"Expected float64 bytes: {expected_bytes_float64}\n"
                f"Actual bytes: {file_size}"
            )

    else:

        dtype = np.float32

    # --------------------------------------------------------
    # READ RAW FDT
    # --------------------------------------------------------

    data = np.fromfile(
        fdt_path,
        dtype=dtype
    )

    if data.size != expected_values:

        raise RuntimeError(
            f"FDT sample count mismatch.\n"
            f"Expected: {expected_values}\n"
            f"Actual: {data.size}"
        )

    data = data.reshape(
        expected_channels,
        pnts
    )

    data = np.asarray(
        data,
        dtype=np.float64
    )

    return {
        "data": data,
        "labels": labels,
        "srate": srate,
        "pnts": pnts,
        "trials": trials,
        "xmin": xmin,
        "xmax": xmax,
        "fdt_path": fdt_path,
    }


def get_subject_run_from_filename(filename):
    """
    Example:
    sub-001_ses-01_task-WorkingMemory_run-1_eeg.set
    """

    m_sub = re.search(
        r"(sub-\d+)",
        filename
    )

    m_run = re.search(
        r"_run-(\d+)",
        filename
    )

    if not m_sub or not m_run:
        raise ValueError(
            f"Could not determine subject/run from {filename}"
        )

    return (
        m_sub.group(1),
        int(m_run.group(1))
    )


def get_bad_channels(channel_qc, subject, run):
    """
    Return BAD channel names only.

    REVIEW channels are NOT interpolated automatically.
    """

    rows = channel_qc[
        (channel_qc["subject"].astype(str) == str(subject))
        &
        (channel_qc["run"].astype(int) == int(run))
    ]

    if len(rows) == 0:
        return []

    # final_status is the current column
    status_col = "final_status"

    if status_col not in rows.columns:

        raise ValueError(
            f"Missing {status_col} in channel QC."
        )

    bad_rows = rows[
        rows[status_col].astype(str).str.upper() == "BAD"
    ]

    bad_channels = (
        bad_rows["channel"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    return bad_channels


def get_run_decision(run_qc, subject, run):
    """
    Get final run-level QC decision.
    """

    rows = run_qc[
        (run_qc["subject"].astype(str) == str(subject))
        &
        (run_qc["run"].astype(int) == int(run))
    ]

    if len(rows) == 0:
        return None

    if "final_run_status" not in rows.columns:
        return None

    return str(
        rows.iloc[0]["final_run_status"]
    )


def normalize_channel_name(name):
    return str(name).strip().upper()


# ============================================================
# LOAD QC
# ============================================================

print("=" * 75)
print("EEG PREPROCESSING - 83 RUNS")
print("=" * 75)

print()
print("Loading QC tables...")

channel_qc = pd.read_csv(
    CHANNEL_QC
)

run_qc = pd.read_csv(
    RUN_QC
)

print(
    f"Channel QC records: {len(channel_qc)}"
)

print(
    f"Run QC records: {len(run_qc)}"
)


# ============================================================
# BASIC QC VALIDATION
# ============================================================

required_channel_columns = [
    "subject",
    "run",
    "channel",
    "final_status",
]

for col in required_channel_columns:

    if col not in channel_qc.columns:

        raise ValueError(
            f"Missing channel QC column: {col}"
        )


required_run_columns = [
    "subject",
    "run",
]

for col in required_run_columns:

    if col not in run_qc.columns:

        raise ValueError(
            f"Missing run QC column: {col}"
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
            "*_eeg.set"
        )
    )
)

print()
print(
    f"SET files found: {len(set_files)}"
)

if len(set_files) != 83:

    raise RuntimeError(
        f"Expected 83 SET files, found {len(set_files)}"
    )


# ============================================================
# STANDARD MONTAGE
# ============================================================

print()
print("Loading standard_1005 montage...")

montage = mne.channels.make_standard_montage(
    "standard_1005"
)

print(
    f"Montage channels: {len(montage.ch_names)}"
)


# ============================================================
# PROCESSING
# ============================================================

processing_log = []

processed_count = 0
excluded_count = 0
failed_count = 0

for idx, set_path in enumerate(set_files, start=1):

    filename = os.path.basename(set_path)

    subject, run = get_subject_run_from_filename(
        filename
    )

    print()
    print("=" * 75)
    print(
        f"[{idx}/83] {subject} RUN {run}"
    )
    print("=" * 75)

    log_entry = {
        "subject": subject,
        "run": run,
        "input_set": set_path,
        "status": None,
        "excluded": False,
        "bad_channels": [],
        "interpolated_channels": [],
        "srate": None,
        "duration_seconds": None,
        "n_channels_input": None,
        "n_channels_output": None,
        "filter": f"{LOW_CUT}-{HIGH_CUT} Hz",
        "notch": NOTCH_FREQ,
        "reference": "average",
        "error": "",
    }

    try:

        # ----------------------------------------------------
        # SPECIAL EXCLUSION
        # ----------------------------------------------------

        if subject == "sub-004" and run == 2:

            print()
            print(
                "!!! EXCLUDED BY FINAL QC !!!"
            )

            print(
                "Reason: 69/69 EEG channels BAD"
            )

            log_entry["status"] = "EXCLUDED"
            log_entry["excluded"] = True

            processing_log.append(log_entry)

            excluded_count += 1

            continue


        # ----------------------------------------------------
        # RUN DECISION
        # ----------------------------------------------------

        run_decision = get_run_decision(
            run_qc,
            subject,
            run
        )

        print(
            f"Final run QC status: {run_decision}"
        )


        # ----------------------------------------------------
        # LOAD FDT
        # ----------------------------------------------------

        result = load_fdt_and_metadata(
            set_path
        )

        data = result["data"]
        labels = result["labels"]
        srate = result["srate"]
        pnts = result["pnts"]

        log_entry["srate"] = srate
        log_entry["duration_seconds"] = (
            pnts / srate
        )
        log_entry["n_channels_input"] = len(labels)

        print(
            f"Channels: {len(labels)}"
        )

        print(
            f"Samples: {pnts}"
        )

        print(
            f"Sampling rate: {srate} Hz"
        )

        print(
            f"Duration: {pnts / srate:.3f} sec"
        )


        # ----------------------------------------------------
        # VALIDATE DATA
        # ----------------------------------------------------

        if data.ndim != 2:

            raise RuntimeError(
                f"Unexpected FDT shape: {data.shape}"
            )

        if data.shape[0] != len(labels):

            raise RuntimeError(
                "Channel count mismatch between "
                "FDT and chanlocs."
            )

        if data.shape[1] != pnts:

            raise RuntimeError(
                "Point count mismatch between "
                "FDT and SET metadata."
            )

        if np.isnan(data).any():

            raise RuntimeError(
                "NaN detected in raw FDT."
            )

        if np.isinf(data).any():

            raise RuntimeError(
                "Inf detected in raw FDT."
            )


        # ----------------------------------------------------
        # CREATE MNE RAW
        # ----------------------------------------------------

        ch_types = []

        for name in labels:

            upper = normalize_channel_name(name)

            if upper in ["LEYE", "REYE", "EOG", "HEOG", "VEOG"]:

                ch_types.append("eog")

            else:

                ch_types.append("eeg")


        info = mne.create_info(
            ch_names=labels,
            sfreq=srate,
            ch_types=ch_types
        )

        raw = mne.io.RawArray(
            data,
            info,
            verbose=False
        )


        # ----------------------------------------------------
        # SET MONTAGE
        # ----------------------------------------------------

        # Try standard montage by channel names.
        raw.set_montage(
            montage,
            match_case=False,
            on_missing="ignore",
            verbose=False
        )


        # ----------------------------------------------------
        # GET BAD CHANNELS FROM QC
        # ----------------------------------------------------

        bad_channels = get_bad_channels(
            channel_qc,
            subject,
            run
        )

        # Normalize names
        bad_channels_norm = {
            normalize_channel_name(x)
            for x in bad_channels
        }

        actual_bad_channels = []

        for ch in raw.ch_names:

            if normalize_channel_name(ch) in bad_channels_norm:

                actual_bad_channels.append(ch)


        print()
        print(
            f"QC BAD channels: {len(actual_bad_channels)}"
        )

        if actual_bad_channels:

            print(
                "BAD:",
                ", ".join(actual_bad_channels)
            )


        log_entry["bad_channels"] = (
            actual_bad_channels.copy()
        )


        # ----------------------------------------------------
        # MARK BAD CHANNELS
        # ----------------------------------------------------

        raw.info["bads"] = (
            actual_bad_channels.copy()
        )


        # ----------------------------------------------------
        # INTERPOLATE BAD EEG CHANNELS
        # ----------------------------------------------------

        interpolated_channels = []

        if actual_bad_channels:

            eeg_bad = [
                ch for ch in actual_bad_channels
                if ch in raw.copy().pick_types(
                    eeg=True,
                    exclude=[]
                ).ch_names
            ]

            # EOG channels must never be interpolated here.
            eeg_bad = [
                ch for ch in eeg_bad
                if ch not in [
                    "LEYE",
                    "REYE"
                ]
            ]

            if eeg_bad:

                print()
                print(
                    "Interpolating EEG BAD channels..."
                )

                print(
                    ", ".join(eeg_bad)
                )

                raw.interpolate_bads(
                    reset_bads=False,
                    mode="accurate",
                    origin="auto",
                    verbose=False
                )

                interpolated_channels = (
                    eeg_bad.copy()
                )

                print(
                    "Interpolation complete."
                )

        log_entry["interpolated_channels"] = (
            interpolated_channels
        )


        # ----------------------------------------------------
        # FILTER
        # ----------------------------------------------------

        print()
        print(
            f"Band-pass filtering: "
            f"{LOW_CUT}-{HIGH_CUT} Hz"
        )

        raw.filter(
            l_freq=LOW_CUT,
            h_freq=HIGH_CUT,
            picks="eeg",
            method="fir",
            phase="zero",
            verbose=False
        )


        # ----------------------------------------------------
        # NOTCH
        # ----------------------------------------------------

        print(
            f"Notch filtering: {NOTCH_FREQ} Hz"
        )

        raw.notch_filter(
            freqs=[NOTCH_FREQ],
            picks="eeg",
            method="fir",
            phase="zero",
            verbose=False
        )


        # ----------------------------------------------------
        # AVERAGE REFERENCE
        # ----------------------------------------------------

        print(
            "Applying average EEG reference..."
        )

        raw.set_eeg_reference(
            ref_channels="average",
            projection=False,
            verbose=False
        )


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_name = (
            filename
            .replace(
                "_eeg.set",
                "_preprocessed_raw.fif"
            )
        )

        output_path = os.path.join(
            OUT_DIR,
            output_name
        )

        print()
        print(
            f"Saving:\n{output_path}"
        )

        raw.save(
            output_path,
            overwrite=True,
            verbose=False
        )


        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        print()
        print(
            "FINAL VALIDATION"
        )

        print(
            f"Output channels: {len(raw.ch_names)}"
        )

        print(
            f"Output samples: {raw.n_times}"
        )

        print(
            f"Output sfreq: {raw.info['sfreq']}"
        )

        print(
            f"Output duration: "
            f"{raw.times[-1]:.3f} sec"
        )

        # Check for NaN/Inf in processed data
        # Read in chunks through get_data.
        check_data = raw.get_data(
            reject_by_annotation=None
        )

        if np.isnan(check_data).any():

            raise RuntimeError(
                "NaN detected after preprocessing."
            )

        if np.isinf(check_data).any():

            raise RuntimeError(
                "Inf detected after preprocessing."
            )

        del check_data


        log_entry["n_channels_output"] = (
            len(raw.ch_names)
        )

        log_entry["status"] = "SUCCESS"

        processing_log.append(log_entry)

        processed_count += 1

        print()
        print(
            "STATUS: SUCCESS"
        )


    except Exception as e:

        failed_count += 1

        log_entry["status"] = "FAILED"
        log_entry["error"] = (
            str(e)
            + "\n"
            + traceback.format_exc()
        )

        processing_log.append(log_entry)

        print()
        print(
            "!!! FAILED !!!"
        )

        print(str(e))

        # Continue to next run
        continue


# ============================================================
# SAVE LOG
# ============================================================

log_df = pd.DataFrame(
    processing_log
)

log_csv = os.path.join(
    LOG_DIR,
    "preprocessing_83runs_log.csv"
)

log_df.to_csv(
    log_csv,
    index=False
)


# ============================================================
# JSON LOG
# ============================================================

log_json = os.path.join(
    LOG_DIR,
    "preprocessing_83runs_log.json"
)

with open(
    log_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        processing_log,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SUMMARY
# ============================================================

summary_path = os.path.join(
    LOG_DIR,
    "preprocessing_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 75 + "\n")
    f.write("EEG PREPROCESSING SUMMARY\n")
    f.write("=" * 75 + "\n\n")

    f.write(
        f"Total SET files: {len(set_files)}\n"
    )

    f.write(
        f"Successfully processed: {processed_count}\n"
    )

    f.write(
        f"Excluded: {excluded_count}\n"
    )

    f.write(
        f"Failed: {failed_count}\n"
    )

    f.write("\n")

    f.write(
        f"Band-pass: {LOW_CUT}-{HIGH_CUT} Hz\n"
    )

    f.write(
        f"Notch: {NOTCH_FREQ} Hz\n"
    )

    f.write(
        "Reference: average EEG reference\n"
    )

    f.write(
        "Interpolation: QC BAD EEG channels only\n"
    )

    f.write("\n")
    f.write(
        "RAW DATA WAS NOT MODIFIED.\n"
    )

    f.write(
        "Original SET/FDT files were not modified.\n"
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print()
print("=" * 75)
print("PREPROCESSING COMPLETE")
print("=" * 75)

print()
print(
    f"Total SET files:       {len(set_files)}"
)

print(
    f"Successfully processed: {processed_count}"
)

print(
    f"Excluded:               {excluded_count}"
)

print(
    f"Failed:                 {failed_count}"
)

print()
print(
    f"Output directory:\n{OUT_DIR}"
)

print()
print(
    f"Log:\n{log_csv}"
)

print(
    f"Summary:\n{summary_path}"
)

print()
print(
    "RAW DATA WAS NOT MODIFIED."
)

print(
    "NO ORIGINAL SET/FDT FILE WAS MODIFIED."
)

print("=" * 75)