from pathlib import Path
import h5py
import numpy as np
import pandas as pd


# ============================================================
# BRAIN PERTURBATION PROJECT
# DATASET-LEVEL EEG QUALITY CONTROL
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# Subjects actually present in the dataset.
# sub-001 is excluded because the download is incomplete.
# sub-013 is not present in the current dataset.
SUBJECTS = [
    f"sub-{i:03d}"
    for i in range(2, 25)
    if i != 13
]


print("=" * 80)
print("BRAIN PERTURBATION PROJECT")
print("DATASET-LEVEL EEG QUALITY CONTROL")
print("=" * 80)

print()
print("Target subjects:")
print(", ".join(SUBJECTS))
print(f"\nTotal target subjects: {len(SUBJECTS)}")
print("Excluded: sub-001 (incomplete download)")
print("Missing:  sub-013 (not present)")
print()


def decode_hdf5_string(obj):
    """
    Convert MATLAB/HDF5 string-like objects to Python strings.
    """
    try:
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="ignore")

        if isinstance(obj, np.ndarray):
            if obj.dtype.kind in {"S", "U"}:
                values = obj.flatten().tolist()
                return [
                    x.decode("utf-8", errors="ignore")
                    if isinstance(x, bytes)
                    else str(x)
                    for x in values
                ]

            if obj.dtype == object:
                return [
                    decode_hdf5_string(x)
                    for x in obj.flatten()
                ]

        return str(obj)

    except Exception:
        return str(obj)


def inspect_set_file(set_path):
    """
    Inspect a MATLAB v7.3 EEGLAB .set file using h5py.

    This function does NOT modify the dataset.
    """

    result = {
        "set_file": set_path.name,
        "set_exists": set_path.exists(),
        "set_readable": False,
        "file_format": "unknown",
        "nbchan": np.nan,
        "pnts": np.nan,
        "trials": np.nan,
        "srate": np.nan,
        "xmin": np.nan,
        "xmax": np.nan,
        "duration_sec": np.nan,
        "datfile": "",
        "events_raw": 0,
        "remember_events": 0,
        "ignore_events": 0,
        "event_types": "",
        "error": "",
    }

    if not set_path.exists():
        result["error"] = "SET file does not exist"
        return result

    try:

        # ----------------------------------------------------
        # Open MATLAB v7.3 / HDF5 file
        # ----------------------------------------------------

        with h5py.File(set_path, "r") as f:

            result["set_readable"] = True
            result["file_format"] = "MATLAB v7.3 / HDF5"

            # ------------------------------------------------
            # Basic EEG metadata
            # ------------------------------------------------

            def scalar_value(name):

                if name not in f:
                    return np.nan

                value = np.array(f[name])

                if value.size == 0:
                    return np.nan

                return float(value.flatten()[0])

            result["nbchan"] = scalar_value("nbchan")
            result["pnts"] = scalar_value("pnts")
            result["trials"] = scalar_value("trials")
            result["srate"] = scalar_value("srate")
            result["xmin"] = scalar_value("xmin")
            result["xmax"] = scalar_value("xmax")

            if (
                not np.isnan(result["pnts"])
                and not np.isnan(result["srate"])
            ):
                result["duration_sec"] = (
                    result["pnts"] / result["srate"]
                )

            # ------------------------------------------------
            # FDT reference
            # ------------------------------------------------

            if "datfile" in f:

                try:
                    raw_datfile = np.array(
                        f["datfile"]
                    )

                    if raw_datfile.size > 0:

                        chars = raw_datfile.flatten()

                        text = ""

                        for c in chars:

                            if isinstance(c, bytes):
                                text += c.decode(
                                    "utf-8",
                                    errors="ignore"
                                )

                            else:
                                try:
                                    text += chr(int(c))
                                except Exception:
                                    text += str(c)

                        result["datfile"] = text

                except Exception:
                    pass

            # ------------------------------------------------
            # Event structure
            # ------------------------------------------------

            event_types = []

            if "event" in f:

                event_group = f["event"]

                # MATLAB structures can be represented
                # differently depending on EEGLAB version.
                #
                # We inspect available event-related objects
                # without assuming a fixed structure.

                def recursive_event_search(name, obj):

                    if len(event_types) > 10000:
                        return

                    try:

                        if isinstance(obj, h5py.Dataset):

                            data = obj[()]

                            if data.size == 0:
                                return

                            # Search text-like datasets.
                            if data.dtype.kind in {"S", "U"}:

                                for item in data.flatten():

                                    if isinstance(item, bytes):
                                        text = item.decode(
                                            "utf-8",
                                            errors="ignore"
                                        )
                                    else:
                                        text = str(item)

                                    if text.strip():
                                        event_types.append(
                                            text.strip()
                                        )

                    except Exception:
                        pass

                event_group.visititems(
                    recursive_event_search
                )

            # ------------------------------------------------
            # Eventdescription may contain useful labels
            # ------------------------------------------------

            if "eventdescription" in f:

                try:

                    event_desc = f["eventdescription"]

                    def search_descriptions(name, obj):

                        if isinstance(obj, h5py.Dataset):

                            try:
                                data = obj[()]

                                if data.size == 0:
                                    return

                                if data.dtype.kind in {"S", "U"}:

                                    for item in data.flatten():

                                        if isinstance(item, bytes):
                                            text = item.decode(
                                                "utf-8",
                                                errors="ignore"
                                            )
                                        else:
                                            text = str(item)

                                        if text.strip():
                                            event_types.append(
                                                text.strip()
                                            )

                            except Exception:
                                pass

                    event_desc.visititems(
                        search_descriptions
                    )

                except Exception:
                    pass

            # ------------------------------------------------
            # Clean event labels
            # ------------------------------------------------

            cleaned_events = []

            for event in event_types:

                event = str(event).strip()

                if event:

                    cleaned_events.append(event)

            # Remove duplicates while preserving order
            unique_events = list(
                dict.fromkeys(cleaned_events)
            )

            result["events_raw"] = len(unique_events)

            result["event_types"] = "; ".join(
                unique_events
            )

            # Count relevant conditions
            for event in unique_events:

                lower_event = event.lower()

                if (
                    "remember" in lower_event
                    or "to_remember" in lower_event
                ):
                    result["remember_events"] += 1

                if (
                    "ignore" in lower_event
                    or "to_ignore" in lower_event
                ):
                    result["ignore_events"] += 1

    except Exception as e:

        result["error"] = (
            type(e).__name__ + ": " + str(e)
        )

    return result


# ============================================================
# MAIN SCAN
# ============================================================

all_results = []


for subject in SUBJECTS:

    print()
    print("=" * 80)
    print(subject)
    print("=" * 80)

    subject_dir = DATA_DIR / subject

    if not subject_dir.exists():

        print("[MISSING] Subject directory not found")

        all_results.append({
            "subject": subject,
            "run": np.nan,
            "set_file": "",
            "set_exists": False,
            "set_readable": False,
            "error": "Subject directory missing"
        })

        continue

    eeg_dirs = list(subject_dir.rglob("eeg"))

    if not eeg_dirs:

        print("[WARNING] No EEG directory found")

        continue

    set_files = []

    for eeg_dir in eeg_dirs:

        set_files.extend(
            sorted(eeg_dir.glob(
                f"{subject}_*_task-WorkingMemory_run-*_eeg.set"
            ))
        )

    set_files = sorted(
        set_files,
        key=lambda p: p.name
    )

    if not set_files:

        print("[WARNING] No SET files found")
        continue

    print(f"Found {len(set_files)} run(s)")

    for set_path in set_files:

        print()
        print("Inspecting:")
        print(set_path.name)

        # Extract run number
        run_number = "unknown"

        parts = set_path.stem.split("_")

        for part in parts:

            if part.startswith("run-"):
                run_number = part.replace(
                    "run-", ""
                )

        info = inspect_set_file(
            set_path
        )

        info["subject"] = subject
        info["run"] = run_number

        # ----------------------------------------------------
        # Check FDT
        # ----------------------------------------------------

        fdt_path = set_path.with_suffix(".fdt")

        info["fdt_exists"] = fdt_path.exists()

        if fdt_path.exists():

            info["fdt_size_MB"] = (
                fdt_path.stat().st_size
                / (1024 ** 2)
            )

        else:

            info["fdt_size_MB"] = np.nan

        # ----------------------------------------------------
        # Print QC information
        # ----------------------------------------------------

        print(
            f"  SET readable : "
            f"{info['set_readable']}"
        )

        print(
            f"  FDT exists   : "
            f"{info['fdt_exists']}"
        )

        print(
            f"  Channels     : "
            f"{info['nbchan']}"
        )

        print(
            f"  Points       : "
            f"{info['pnts']}"
        )

        print(
            f"  Trials       : "
            f"{info['trials']}"
        )

        print(
            f"  Sampling Hz  : "
            f"{info['srate']}"
        )

        print(
            f"  Time range   : "
            f"{info['xmin']} → {info['xmax']} sec"
        )

        print(
            f"  Event labels : "
            f"{info['event_types']}"
        )

        print(
            f"  Error        : "
            f"{info['error'] or 'None'}"
        )

        all_results.append(info)


# ============================================================
# SAVE QC TABLE
# ============================================================

df = pd.DataFrame(all_results)

# Put important columns first
preferred_columns = [
    "subject",
    "run",
    "set_file",
    "set_exists",
    "set_readable",
    "fdt_exists",
    "fdt_size_MB",
    "nbchan",
    "pnts",
    "trials",
    "srate",
    "xmin",
    "xmax",
    "duration_sec",
    "datfile",
    "events_raw",
    "remember_events",
    "ignore_events",
    "event_types",
    "file_format",
    "error",
]

existing_columns = [
    c for c in preferred_columns
    if c in df.columns
]

remaining_columns = [
    c for c in df.columns
    if c not in existing_columns
]

df = df[
    existing_columns + remaining_columns
]


output_file = (
    RESULTS_DIR /
    "dataset_QC_report.csv"
)

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("DATASET QC SUMMARY")
print("=" * 80)

if len(df) > 0:

    valid_set = df["set_readable"].fillna(False)

    print(
        f"Subjects detected: "
        f"{df['subject'].nunique()}"
    )

    print(
        f"SET files detected: "
        f"{len(df)}"
    )

    print(
        f"Readable SET files: "
        f"{valid_set.sum()}"
    )

    if "fdt_exists" in df.columns:

        print(
            f"SET files with FDT: "
            f"{df['fdt_exists'].sum()}"
        )

    print()
    print("Runs per subject:")

    runs_per_subject = (
        df.groupby("subject")
        .size()
    )

    for subject, count in runs_per_subject.items():

        print(
            f"  {subject}: "
            f"{count} run(s)"
        )

print()
print("=" * 80)
print("QC REPORT SAVED")
print("=" * 80)

print(output_file)

print()
print("[DONE]")