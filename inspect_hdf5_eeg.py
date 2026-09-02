import os
import glob
import h5py
import numpy as np
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
QC_DIR = os.path.join(PROJECT_DIR, "qc", "hdf5_qc")

os.makedirs(QC_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(QC_DIR, "hdf5_qc_report.csv")

# ============================================================
# HELPERS
# ============================================================

def decode_hdf5_value(f, obj):
    """
    Recursively decode common MATLAB v7.3 / HDF5 values.
    """
    try:
        if isinstance(obj, h5py.Reference):
            if not obj:
                return None

            target = f[obj]

            if isinstance(target, h5py.Dataset):
                data = target[()]

                if isinstance(data, bytes):
                    return data.decode("utf-8", errors="ignore")

                if np.isscalar(data):
                    return data.item() if hasattr(data, "item") else data

                arr = np.asarray(data)

                if arr.dtype.kind in ("S", "U"):
                    return "".join(
                        x.decode("utf-8", errors="ignore")
                        if isinstance(x, bytes) else str(x)
                        for x in arr.flatten()
                    )

                return arr

            return str(target.name)

        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="ignore")

        if np.isscalar(obj):
            return obj.item() if hasattr(obj, "item") else obj

        return obj

    except Exception:
        return None


def find_dataset(f, names):
    """
    Search recursively for a dataset whose final path component
    matches one of the requested names.
    """
    found = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            base = name.split("/")[-1].lower()
            if base in [x.lower() for x in names]:
                found.append(obj)

    f.visititems(visitor)

    return found[0] if found else None


def get_scalar_from_dataset(f, dataset):
    """
    Extract a scalar from an HDF5 dataset, including references.
    """
    if dataset is None:
        return None

    try:
        data = dataset[()]

        if isinstance(data, h5py.Reference):
            return decode_hdf5_value(f, data)

        if isinstance(data, np.ndarray) and data.dtype == object:
            ref = data.flat[0]
            return decode_hdf5_value(f, ref)

        if np.isscalar(data):
            return data.item() if hasattr(data, "item") else data

        arr = np.asarray(data)

        if arr.size == 1:
            value = arr.flat[0]

            if isinstance(value, h5py.Reference):
                return decode_hdf5_value(f, value)

            return value.item() if hasattr(value, "item") else value

    except Exception:
        pass

    return None


# ============================================================
# FIND EEG FILES
# ============================================================

set_files = glob.glob(
    os.path.join(DATA_DIR, "**", "*_eeg.set"),
    recursive=True
)

print("=" * 80)
print("HDF5 EEG QC")
print("=" * 80)

print(f"Data directory:")
print(DATA_DIR)

print(f"\nFound EEG SET files: {len(set_files)}")

if len(set_files) == 0:
    print("\nERROR: No *_eeg.set files found.")
    print("Check DATA_DIR.")
    raise SystemExit

# ============================================================
# PROCESS FILES
# ============================================================

results = []

for idx, set_file in enumerate(sorted(set_files), 1):

    filename = os.path.basename(set_file)

    print("\n" + "=" * 80)
    print(f"PROCESSING {idx}/{len(set_files)}")
    print("=" * 80)
    print(filename)

    row = {
        "file": filename,
        "subject": "",
        "sampling_rate": np.nan,
        "channels": np.nan,
        "samples": np.nan,
        "duration_sec": np.nan,
        "data_path": "",
        "data_shape": "",
        "data_dtype": "",
        "data_min": np.nan,
        "data_max": np.nan,
        "data_std": np.nan,
        "status": "ERROR"
    }

    # --------------------------------------------------------
    # SUBJECT
    # --------------------------------------------------------

    parts = filename.split("_")

    if len(parts) > 0:
        row["subject"] = parts[0]

    # --------------------------------------------------------
    # OPEN HDF5
    # --------------------------------------------------------

    try:

        with h5py.File(set_file, "r") as f:

            print("HDF5: OK")

            # ------------------------------------------------
            # TOP LEVEL
            # ------------------------------------------------

            print("Top-level keys:")
            print(list(f.keys())[:20])

            # ------------------------------------------------
            # SAMPLING RATE
            # ------------------------------------------------

            srate_ds = find_dataset(
                f,
                ["srate", "samplingrate", "sampling_rate"]
            )

            srate = get_scalar_from_dataset(f, srate_ds)

            if srate is not None:
                try:
                    row["sampling_rate"] = float(np.asarray(srate).squeeze())
                except Exception:
                    pass

            # ------------------------------------------------
            # DATASET SEARCH
            # ------------------------------------------------

            data_candidates = []

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    base = name.split("/")[-1].lower()

                    if base in ["data", "eegdata", "eeg"]:
                        data_candidates.append(obj)

            f.visititems(visitor)

            data_ds = data_candidates[0] if data_candidates else None

            if data_ds is None:

                # fallback: search largest datasets
                datasets = []

                def collect(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        try:
                            size = np.prod(obj.shape)
                            datasets.append((size, name, obj))
                        except Exception:
                            pass

                f.visititems(collect)

                datasets.sort(reverse=True)

                if datasets:
                    data_ds = datasets[0][2]

            # ------------------------------------------------
            # DATA INFORMATION
            # ------------------------------------------------

            if data_ds is not None:

                row["data_path"] = data_ds.name
                row["data_shape"] = str(data_ds.shape)
                row["data_dtype"] = str(data_ds.dtype)

                print(f"Data dataset: {data_ds.name}")
                print(f"Data shape: {data_ds.shape}")
                print(f"Data dtype: {data_ds.dtype}")

                # ------------------------------------------------
                # READ DATA ONLY IF NUMERIC
                # ------------------------------------------------

                if np.issubdtype(data_ds.dtype, np.number):

                    data = data_ds[()]

                    data = np.asarray(data)

                    print(f"Loaded data shape: {data.shape}")

                    row["data_min"] = float(np.nanmin(data))
                    row["data_max"] = float(np.nanmax(data))
                    row["data_std"] = float(np.nanstd(data))

                    # Determine channels / samples
                    if data.ndim >= 2:

                        shape = data.shape

                        # EEG normally = channels x samples
                        if shape[0] <= 128 and shape[1] > shape[0]:
                            n_channels = shape[0]
                            n_samples = shape[1]

                        elif shape[1] <= 128 and shape[0] > shape[1]:
                            n_channels = shape[1]
                            n_samples = shape[0]

                        else:
                            n_channels = shape[0]
                            n_samples = shape[1]

                        row["channels"] = int(n_channels)
                        row["samples"] = int(n_samples)

                        if np.isfinite(row["sampling_rate"]):
                            row["duration_sec"] = (
                                n_samples / row["sampling_rate"]
                            )

                    print(f"Channels: {row['channels']}")
                    print(f"Samples: {row['samples']}")
                    print(f"Sampling rate: {row['sampling_rate']}")
                    print(f"Min: {row['data_min']}")
                    print(f"Max: {row['data_max']}")
                    print(f"STD: {row['data_std']}")

                else:
                    print("WARNING: Data dataset is not numeric.")

            else:
                print("WARNING: EEG data dataset not found.")

            # ------------------------------------------------
            # EVENT INFORMATION
            # ------------------------------------------------

            event_fields = [
                "type",
                "latency",
                "sample",
                "trial",
                "letter",
                "memory_cond",
                "task_role",
                "value",
                "urevent"
            ]

            found_events = []

            for field in event_fields:

                ds = find_dataset(f, [field])

                if ds is not None:
                    found_events.append(field)

            print("\nEvent fields found:")
            print(found_events)

            row["event_fields"] = ",".join(found_events)
            row["event_field_count"] = len(found_events)

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if (
                row["channels"] == 71
                and np.isfinite(row["sampling_rate"])
                and row["samples"] > 0
            ):
                row["status"] = "OK"

            elif row["data_path"]:
                row["status"] = "PARTIAL"

            else:
                row["status"] = "ERROR"

    except Exception as e:

        print(f"ERROR: {e}")
        row["status"] = "ERROR"
        row["error"] = str(e)

    results.append(row)


# ============================================================
# SAVE REPORT
# ============================================================

df = pd.DataFrame(results)

if "error" not in df.columns:
    df["error"] = ""

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL HDF5 QC SUMMARY")
print("=" * 80)

print(f"Total files: {len(df)}")

print("\nSTATUS:")
print(df["status"].value_counts(dropna=False))

print("\nSAMPLING RATES:")
print(df["sampling_rate"].value_counts(dropna=False).sort_index())

print("\nCHANNEL COUNTS:")
print(df["channels"].value_counts(dropna=False).sort_index())

print("\nSUBJECTS:")
print(df["subject"].nunique())

print("\nFILES WITH ERRORS:")
errors = df[df["status"] == "ERROR"]

if len(errors):
    print(errors[["subject", "file", "error"]].to_string(index=False))
else:
    print("None")

print("\n" + "=" * 80)
print("REPORT SAVED")
print("=" * 80)

print(OUTPUT_FILE)

print("\nDONE.")
print("=" * 80)