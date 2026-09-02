import os
import glob
import csv
import h5py
import numpy as np
import mne

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "preprocessed_v2")
LOG_DIR = os.path.join(OUT_DIR, "logs")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOWCUT = 0.1
HIGHCUT = 40.0
NOTCH = 50.0

# ------------------------------------------------------------
# HDF5 MATLAB helpers
# ------------------------------------------------------------

def decode_hdf5_string(h5, obj):
    """
    Decode MATLAB v7.3 string/char stored as uint16/uint8/object refs.
    """
    try:
        if isinstance(obj, h5py.Reference):
            obj = h5[obj]

        if isinstance(obj, h5py.Dataset):
            value = obj[()]

            if isinstance(value, np.ndarray):
                value = np.array(value).squeeze()

            if isinstance(value, np.ndarray):
                if value.dtype == object:
                    vals = []
                    for x in value.flat:
                        try:
                            vals.append(decode_hdf5_string(h5, x))
                        except Exception:
                            pass
                    return "".join(vals)

                if value.dtype.kind in ("u", "i"):
                    chars = []
                    for x in value.flat:
                        ix = int(x)
                        if ix != 0:
                            chars.append(chr(ix))
                    return "".join(chars)

                if value.dtype.kind in ("S", "U"):
                    return "".join(str(x) for x in value.flat)

            if isinstance(value, bytes):
                return value.decode("utf-8", errors="ignore")

            return str(value)

        return str(obj)

    except Exception:
        return str(obj)


def read_scalar_hdf5(h5, obj):
    """
    Recursively resolve a MATLAB v7.3 scalar/object reference.
    """
    if isinstance(obj, h5py.Reference):
        obj = h5[obj]

    if isinstance(obj, h5py.Dataset):
        value = obj[()]

        if isinstance(value, np.ndarray):
            value = np.array(value).squeeze()

        if isinstance(value, h5py.Reference):
            return read_scalar_hdf5(h5, value)

        if isinstance(value, np.ndarray):

            if value.dtype == object:
                if value.size == 0:
                    return None
                return read_scalar_hdf5(h5, value.flat[0])

            if value.size == 1:
                return value.item()

            return value

        return value

    return obj


def extract_event_field(h5, field_dataset, n_events):
    """
    Extract a MATLAB v7.3 event field.
    """
    result = []

    raw = field_dataset[()]

    raw = np.array(raw).reshape(-1)

    for i in range(min(n_events, len(raw))):
        try:
            value = read_scalar_hdf5(h5, raw[i])

            if isinstance(value, np.ndarray):
                value = np.array(value).squeeze()

                if value.size == 1:
                    value = value.item()

            result.append(value)

        except Exception:
            result.append(None)

    while len(result) < n_events:
        result.append(None)

    return result


def extract_events_from_set(set_file):
    """
    Extract event type + latency from MATLAB v7.3 EEGLAB .set file.
    """
    events = []

    with h5py.File(set_file, "r") as h5:

        if "event" not in h5:
            return events

        event_group = h5["event"]

        if "latency" not in event_group:
            return events

        latency_ds = event_group["latency"]

        raw_latency = latency_ds[()]
        raw_latency = np.array(raw_latency).reshape(-1)

        n_events = len(raw_latency)

        type_values = None

        if "type" in event_group:
            type_values = extract_event_field(
                h5,
                event_group["type"],
                n_events
            )

        for i in range(n_events):

            try:
                latency = read_scalar_hdf5(
                    h5,
                    raw_latency[i]
                )

                if isinstance(latency, np.ndarray):
                    latency = np.array(latency).squeeze()

                    if latency.size == 1:
                        latency = latency.item()

                latency = float(latency)

            except Exception:
                continue

            if type_values is not None and i < len(type_values):
                event_type = type_values[i]
            else:
                event_type = "event"

            if isinstance(event_type, np.ndarray):
                event_type = np.array(event_type).squeeze()

                if event_type.size == 1:
                    event_type = event_type.item()

            event_type = str(event_type)

            events.append({
                "latency": latency,
                "type": event_type
            })

    return events


# ------------------------------------------------------------
# Find matching FDT
# ------------------------------------------------------------

def find_fdt(set_file):
    base = os.path.splitext(set_file)[0]
    candidate = base + ".fdt"

    if os.path.exists(candidate):
        return candidate

    return None


# ------------------------------------------------------------
# Main processing
# ------------------------------------------------------------

set_files = sorted(
    glob.glob(
        os.path.join(DATA_DIR, "**", "*.set"),
        recursive=True
    )
)

print("=" * 80)
print("PREPROCESSING WITH EVENT PRESERVATION")
print("=" * 80)

print("SET files found:", len(set_files))
print()

log_rows = []

success = 0
failed = 0

for idx, set_file in enumerate(set_files, 1):

    filename = os.path.basename(set_file)

    print()
    print("=" * 80)
    print(f"[{idx}/{len(set_files)}] {filename}")
    print("=" * 80)

    try:

        fdt_file = find_fdt(set_file)

        if fdt_file is None:
            raise FileNotFoundError(
                "Matching FDT file not found."
            )

        print("SET:", set_file)
        print("FDT:", fdt_file)

        # ----------------------------------------------------
        # Read metadata from SET
        # ----------------------------------------------------

        with h5py.File(set_file, "r") as h5:

            nbchan = float(
                np.array(h5["nbchan"][()]).squeeze()
            )

            srate = float(
                np.array(h5["srate"][()]).squeeze()
            )

            pnts = float(
                np.array(h5["pnts"][()]).squeeze()
            )

            trials = float(
                np.array(h5["trials"][()]).squeeze()
            )

            duration = pnts / srate

        print("Channels:", int(nbchan))
        print("Sampling rate:", srate)
        print("Points:", int(pnts))
        print("Trials:", int(trials))
        print("Duration:", duration)

        # ----------------------------------------------------
        # Read channel names from SET
        # ----------------------------------------------------

        ch_names = []

        with h5py.File(set_file, "r") as h5:

            chanlocs = h5["chanlocs"]

            if "labels" not in chanlocs:
                raise RuntimeError(
                    "chanlocs/labels not found."
                )

            labels = chanlocs["labels"][()]
            labels = np.array(labels).reshape(-1)

            for ref in labels:

                label = decode_hdf5_string(
                    h5,
                    ref
                )

                label = label.strip()

                if not label:
                    label = f"CH{len(ch_names)+1:02d}"

                ch_names.append(label)

        if len(ch_names) != int(nbchan):
            print(
                "WARNING: channel label count differs from nbchan:",
                len(ch_names),
                int(nbchan)
            )

        # ----------------------------------------------------
        # Read FDT directly
        # ----------------------------------------------------

        expected_values = int(nbchan * pnts)

        print()
        print("Reading FDT...")
        print("Expected float32 values:", expected_values)

        file_size = os.path.getsize(fdt_file)

        expected_bytes = expected_values * 4

        print("FDT bytes:", file_size)
        print("Expected bytes:", expected_bytes)

        if file_size < expected_bytes:
            raise RuntimeError(
                f"FDT too small. "
                f"Expected {expected_bytes}, got {file_size}"
            )

        data = np.fromfile(
            fdt_file,
            dtype=np.float32,
            count=expected_values
        )

        if data.size != expected_values:
            raise RuntimeError(
                f"Unexpected FDT size: "
                f"{data.size} vs {expected_values}"
            )

        data = data.reshape(
            (int(nbchan), int(pnts)),
            order="F"
        )

        print("Data shape:", data.shape)
        print("Data dtype:", data.dtype)
        print("Min:", float(np.nanmin(data)))
        print("Max:", float(np.nanmax(data)))
        print("Mean:", float(np.nanmean(data)))
        print("STD:", float(np.nanstd(data)))

        # ----------------------------------------------------
        # Create MNE RawArray
        # ----------------------------------------------------

        ch_types = []

        for name in ch_names:

            upper = name.upper()

            if upper in ["LEYE", "REYE", "HEOG", "VEOG", "EOG"]:
                ch_types.append("eog")
            else:
                ch_types.append("eeg")

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=srate,
            ch_types=ch_types
        )

        raw = mne.io.RawArray(
            data,
            info,
            verbose=False
        )

        # ----------------------------------------------------
        # Extract events BEFORE filtering
        # ----------------------------------------------------

        print()
        print("Extracting events from SET...")

        event_records = extract_events_from_set(
            set_file
        )

        print("Events extracted:", len(event_records))

        # ----------------------------------------------------
        # Convert event latency to annotations
        #
        # EEGLAB latency is 1-based sample position.
        # MNE onset is seconds from beginning.
        # ----------------------------------------------------

        annotations = []

        for ev in event_records:

            latency = float(ev["latency"])

            onset = (latency - 1.0) / srate

            if onset < 0:
                continue

            if onset >= raw.times[-1]:
                continue

            description = str(ev["type"])

            annotations.append(
                mne.Annotations(
                    onset=[onset],
                    duration=[0.0],
                    description=[description]
                )
            )

        if len(annotations) > 0:

            combined = annotations[0]

            for ann in annotations[1:]:
                combined += ann

            raw.set_annotations(combined)

        print(
            "Annotations in Raw:",
            len(raw.annotations)
        )

        # ----------------------------------------------------
        # Filtering
        # ----------------------------------------------------

        print()
        print("Band-pass filtering:",
              LOWCUT, "-", HIGHCUT, "Hz")

        raw.filter(
            l_freq=LOWCUT,
            h_freq=HIGHCUT,
            picks="eeg",
            method="fir",
            phase="zero",
            verbose=False
        )

        print(
            "Notch filtering:",
            NOTCH,
            "Hz"
        )

        raw.notch_filter(
            freqs=[NOTCH],
            picks="eeg",
            method="fir",
            phase="zero",
            verbose=False
        )

        # ----------------------------------------------------
        # Average EEG reference
        # ----------------------------------------------------

        print("Applying average EEG reference...")

        raw.set_eeg_reference(
            "average",
            projection=False,
            verbose=False
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        final_events = len(raw.annotations)

        if final_events != len(event_records):
            print(
                "WARNING: event count changed:",
                len(event_records),
                "->",
                final_events
            )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        stem = os.path.splitext(filename)[0]

        out_file = os.path.join(
            OUT_DIR,
            stem.replace(
                "_eeg",
                "_preprocessed_raw"
            ) + ".fif"
        )

        print()
        print("Saving:")
        print(out_file)

        raw.save(
            out_file,
            overwrite=True,
            verbose=False
        )

        # ----------------------------------------------------
        # Final validation after save
        # ----------------------------------------------------

        check = mne.io.read_raw_fif(
            out_file,
            preload=False,
            verbose=False
        )

        saved_annotations = len(
            check.annotations
        )

        saved_duration = check.times[-1]

        print()
        print("FINAL VALIDATION")
        print("Output channels:", len(check.ch_names))
        print("Output samples:", check.n_times)
        print("Output sfreq:", check.info["sfreq"])
        print("Output duration:", saved_duration)
        print("Output annotations:", saved_annotations)

        if saved_annotations != len(event_records):
            raise RuntimeError(
                "EVENT COUNT MISMATCH AFTER SAVING"
            )

        print()
        print("STATUS: SUCCESS")

        success += 1

        log_rows.append({
            "subject": stem.split("_")[0],
            "file": filename,
            "status": "SUCCESS",
            "events_extracted": len(event_records),
            "events_saved": saved_annotations,
            "srate": srate,
            "duration_seconds": saved_duration,
            "output": out_file
        })

    except Exception as e:

        print()
        print("STATUS: FAILED")
        print("ERROR:", repr(e))

        failed += 1

        log_rows.append({
            "subject": os.path.basename(set_file).split("_")[0],
            "file": filename,
            "status": "FAILED",
            "events_extracted": "",
            "events_saved": "",
            "srate": "",
            "duration_seconds": "",
            "output": "",
            "error": repr(e)
        })


# ------------------------------------------------------------
# Save log
# ------------------------------------------------------------

log_file = os.path.join(
    LOG_DIR,
    "preprocessing_with_events_82runs_log.csv"
)

fieldnames = [
    "subject",
    "file",
    "status",
    "events_extracted",
    "events_saved",
    "srate",
    "duration_seconds",
    "output",
    "error"
]

with open(
    log_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in log_rows:
        writer.writerow(row)


summary_file = os.path.join(
    LOG_DIR,
    "preprocessing_with_events_82runs_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 80 + "\n")
    f.write("PREPROCESSING WITH EVENTS SUMMARY\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"SET files: {len(set_files)}\n")
    f.write(f"Successful: {success}\n")
    f.write(f"Failed: {failed}\n\n")

    f.write(
        "RAW DATA WAS NOT MODIFIED.\n"
    )

    f.write(
        "ORIGINAL SET/FDT FILES WERE NOT MODIFIED.\n"
    )

print()
print("=" * 80)
print("PREPROCESSING WITH EVENTS COMPLETE")
print("=" * 80)

print("Total SET files:", len(set_files))
print("Successfully processed:", success)
print("Failed:", failed)

print()
print("Output directory:")
print(OUT_DIR)

print()
print("Log:")
print(log_file)

print()
print("Summary:")
print(summary_file)

print()
print("RAW DATA WAS NOT MODIFIED.")
print("ORIGINAL SET/FDT FILES WERE NOT MODIFIED.")
print("=" * 80)