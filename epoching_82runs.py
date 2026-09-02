import os
import glob
import traceback
import numpy as np
import pandas as pd
import mne
import h5py


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

PREPROCESSED_DIR = os.path.join(
    PROJECT,
    "preprocessed"
)

RAW_DATA_DIR = os.path.join(
    PROJECT,
    "data"
)

OUTPUT_DIR = os.path.join(
    PROJECT,
    "epochs"
)

LOG_DIR = os.path.join(
    OUTPUT_DIR,
    "logs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# EPOCH PARAMETERS
# ============================================================

TMIN = -0.2
TMAX = 0.8

BASELINE = (-0.2, 0)

# Conservative EEG rejection threshold
# 300 microvolts
REJECT = {
    "eeg": 300e-6
}


# ============================================================
# HDF5 DECODING FUNCTIONS
# ============================================================

def decode_hdf5_value(h5file, value):

    if isinstance(value, h5py.h5r.Reference):

        if not value:
            return None

        obj = h5file[value]

        if isinstance(obj, h5py.Dataset):

            data = obj[()]

            return decode_hdf5_value(
                h5file,
                data
            )

        elif isinstance(obj, h5py.Group):

            result = {}

            for key in obj.keys():

                child = obj[key]

                if isinstance(child, h5py.Dataset):

                    result[key] = decode_hdf5_value(
                        h5file,
                        child[()]
                    )

            return result

    if isinstance(value, np.ndarray):

        if value.dtype == object:

            result = []

            for item in value.flatten():

                result.append(
                    decode_hdf5_value(
                        h5file,
                        item
                    )
                )

            return result

        if value.size == 1:

            return value.flatten()[0].item()

        return value

    if isinstance(value, bytes):

        try:
            return value.decode("utf-8")
        except Exception:
            return str(value)

    if isinstance(value, np.generic):

        return value.item()

    return value


def clean_string(value):

    if value is None:
        return ""

    if isinstance(value, bytes):

        try:
            return value.decode("utf-8")
        except Exception:
            return str(value)

    if isinstance(value, np.ndarray):

        if value.size == 0:
            return ""

        if value.dtype == object:

            if value.size == 1:

                return clean_string(
                    value.flatten()[0]
                )

            return " ".join(
                clean_string(x)
                for x in value.flatten()
            )

        if value.dtype.kind in ["U", "S"]:

            return "".join(
                value.astype(str).flatten()
            )

        if value.size == 1:

            return clean_string(
                value.flatten()[0]
            )

    return str(value)


# ============================================================
# FIND ORIGINAL SET
# ============================================================

def find_set_file(subject, run):

    path = os.path.join(
        RAW_DATA_DIR,
        subject,
        "ses-01",
        "eeg",
        f"{subject}_ses-01_task-WorkingMemory_run-{run}_eeg.set"
    )

    if os.path.exists(path):

        return path

    return None


# ============================================================
# READ EVENTS FROM EEGLAB HDF5 SET
# ============================================================

def read_events_from_set(set_file):

    events = []

    with h5py.File(
        set_file,
        "r"
    ) as f:

        if "event" not in f:

            raise ValueError(
                "EVENT group not found."
            )

        event_group = f["event"]

        if "latency" not in event_group:

            raise ValueError(
                "EVENT latency field not found."
            )

        if "type" not in event_group:

            raise ValueError(
                "EVENT type field not found."
            )

        latency_ds = event_group["latency"]
        type_ds = event_group["type"]

        n_events = latency_ds.shape[0]

        optional_fields = [
            "duration",
            "letter",
            "memory_cond",
            "sample",
            "task_role",
            "trial",
            "urevent",
            "value"
        ]

        for i in range(n_events):

            # ------------------------------------------------
            # LATENCY
            # ------------------------------------------------

            latency_raw = latency_ds[i, 0]

            latency = decode_hdf5_value(
                f,
                latency_raw
            )

            try:

                latency = float(latency)

            except Exception:

                latency = np.nan


            # ------------------------------------------------
            # EVENT TYPE
            # ------------------------------------------------

            type_raw = type_ds[i, 0]

            event_type = decode_hdf5_value(
                f,
                type_raw
            )

            event_type = clean_string(
                event_type
            )


            event = {
                "event_index": i,
                "latency": latency,
                "type": event_type
            }


            # ------------------------------------------------
            # OPTIONAL EVENT FIELDS
            # ------------------------------------------------

            for field in optional_fields:

                if field not in event_group:

                    event[field] = ""

                    continue

                try:

                    raw_value = event_group[
                        field
                    ][i, 0]

                    value = decode_hdf5_value(
                        f,
                        raw_value
                    )

                    if isinstance(
                        value,
                        (np.ndarray, list)
                    ):

                        value = clean_string(
                            value
                        )

                    event[field] = value

                except Exception:

                    event[field] = ""


            events.append(event)


    return events


# ============================================================
# EXTRACT SUBJECT AND RUN
# ============================================================

def get_subject_and_run(filename):

    parts = filename.split("_")

    subject = parts[0]

    run = None

    for part in parts:

        if part.startswith("run-"):

            try:

                run = int(
                    part.replace(
                        "run-",
                        ""
                    )
                )

            except Exception:

                run = None


    return subject, run


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 80)
print("EPOCHING PIPELINE")
print("=" * 80)

print()
print("Project:")
print(PROJECT)

print()
print("Preprocessed directory:")
print(PREPROCESSED_DIR)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("=" * 80)


# ============================================================
# FIND PREPROCESSED FIF FILES
# ============================================================

fif_files = sorted(
    glob.glob(
        os.path.join(
            PREPROCESSED_DIR,
            "*_preprocessed_raw.fif"
        )
    )
)

print()
print(
    f"Preprocessed FIF files found: {len(fif_files)}"
)

if len(fif_files) != 82:

    print()
    print("WARNING")
    print(
        f"Expected 82 files but found {len(fif_files)}"
    )

print()


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS FILES
# ============================================================

for counter, fif_file in enumerate(
    fif_files,
    start=1
):

    filename = os.path.basename(
        fif_file
    )

    print()
    print("=" * 80)
    print(
        f"PROCESSING {counter}/{len(fif_files)}"
    )
    print("=" * 80)

    print()
    print("FIF:")
    print(filename)


    # --------------------------------------------------------
    # SUBJECT / RUN
    # --------------------------------------------------------

    subject, run = get_subject_and_run(
        filename
    )

    print()
    print("Subject:", subject)
    print("Run:", run)


    if run is None:

        print()
        print("STATUS: FAILED")
        print("Could not determine run number.")

        results.append({
            "subject": subject,
            "run": "",
            "file": filename,
            "events_found": 0,
            "valid_events": 0,
            "edge_removed": 0,
            "epochs_before_rejection": 0,
            "epochs_retained": 0,
            "epochs_rejected": 0,
            "status": "FAILED",
            "reason": "Could not determine run number"
        })

        continue


    # --------------------------------------------------------
    # FIND ORIGINAL SET
    # --------------------------------------------------------

    set_file = find_set_file(
        subject,
        run
    )

    if set_file is None:

        print()
        print("STATUS: FAILED")
        print("Original SET file not found.")

        results.append({
            "subject": subject,
            "run": run,
            "file": filename,
            "events_found": 0,
            "valid_events": 0,
            "edge_removed": 0,
            "epochs_before_rejection": 0,
            "epochs_retained": 0,
            "epochs_rejected": 0,
            "status": "FAILED",
            "reason": "Original SET file not found"
        })

        continue


    print()
    print("SET:")
    print(set_file)


    try:

        # ====================================================
        # LOAD PREPROCESSED FIF
        # ====================================================

        print()
        print("Reading preprocessed FIF...")

        raw = mne.io.read_raw_fif(
            fif_file,
            preload=True,
            verbose=False
        )

        print(
            "Channels:",
            len(raw.ch_names)
        )

        print(
            "Sampling rate:",
            raw.info["sfreq"]
        )

        print(
            "Samples:",
            raw.n_times
        )

        print(
            "Duration:",
            round(
                raw.times[-1],
                3
            ),
            "sec"
        )


        # ====================================================
        # READ EVENTS
        # ====================================================

        print()
        print("Reading events from SET...")

        events_data = read_events_from_set(
            set_file
        )

        print(
            "Events found:",
            len(events_data)
        )


        if len(events_data) == 0:

            raise ValueError(
                "No events found."
            )


        # ====================================================
        # VALIDATE LATENCIES
        # ====================================================

        sfreq = raw.info["sfreq"]

        valid_events = []

        invalid_events = []

        for event in events_data:

            latency = event["latency"]

            if not np.isfinite(latency):

                invalid_events.append(event)

                continue


            # EEGLAB latency is 1-based.
            sample = int(
                round(latency)
            ) - 1


            if sample < 0:

                invalid_events.append(event)

                continue


            if sample >= raw.n_times:

                invalid_events.append(event)

                continue


            event["sample"] = sample

            valid_events.append(event)


        print(
            "Valid events:",
            len(valid_events)
        )

        print(
            "Invalid events:",
            len(invalid_events)
        )


        # ====================================================
        # EVENT TYPE IDs
        # ====================================================

        event_id = {}

        next_id = 1

        mne_events = []

        metadata_rows = []


        for event in valid_events:

            event_type = clean_string(
                event["type"]
            )

            if event_type == "":

                continue


            if event_type not in event_id:

                event_id[
                    event_type
                ] = next_id

                next_id += 1


            current_id = event_id[
                event_type
            ]


            sample = int(
                event["sample"]
            )


            mne_events.append([
                sample,
                0,
                current_id
            ])


            metadata_rows.append({
                "event_index":
                    event["event_index"],

                "sample":
                    sample,

                "latency":
                    event["latency"],

                "event_type":
                    event_type,

                "duration":
                    event.get(
                        "duration",
                        ""
                    ),

                "letter":
                    event.get(
                        "letter",
                        ""
                    ),

                "memory_cond":
                    event.get(
                        "memory_cond",
                        ""
                    ),

                "sample_field":
                    event.get(
                        "sample",
                        ""
                    ),

                "task_role":
                    event.get(
                        "task_role",
                        ""
                    ),

                "trial":
                    event.get(
                        "trial",
                        ""
                    ),

                "urevent":
                    event.get(
                        "urevent",
                        ""
                    ),

                "value":
                    event.get(
                        "value",
                        ""
                    )
            })


        if len(mne_events) == 0:

            raise ValueError(
                "No usable event types found."
            )


        mne_events = np.asarray(
            mne_events,
            dtype=int
        )


        metadata_df = pd.DataFrame(
            metadata_rows
        )


        # ====================================================
        # EVENT TYPE SUMMARY
        # ====================================================

        print()
        print("EVENT TYPES")
        print("-" * 70)

        for name, eid in event_id.items():

            count = int(
                np.sum(
                    mne_events[:, 2] == eid
                )
            )

            print(
                f"{eid:3d} | {name:30s} | {count}"
            )


        # ====================================================
        # REMOVE EDGE EVENTS
        # ====================================================

        start_offset = int(
            round(
                TMIN * sfreq
            )
        )

        end_offset = int(
            round(
                TMAX * sfreq
            )
        )


        epoch_events = []

        epoch_metadata = []

        edge_removed = 0


        for idx, event in enumerate(
            mne_events
        ):

            sample = int(
                event[0]
            )


            if (
                sample + start_offset < 0
                or
                sample + end_offset >= raw.n_times
            ):

                edge_removed += 1

                continue


            epoch_events.append(
                event
            )

            epoch_metadata.append(
                metadata_df.iloc[
                    idx
                ].to_dict()
            )


        epoch_events = np.asarray(
            epoch_events,
            dtype=int
        )


        epoch_metadata = pd.DataFrame(
            epoch_metadata
        )


        print()
        print(
            "Events removed at edges:",
            edge_removed
        )

        print(
            "Events entering epoching:",
            len(epoch_events)
        )


        if len(epoch_events) == 0:

            raise ValueError(
                "No events remain after edge validation."
            )


        # ====================================================
        # CREATE EPOCHS
        # ====================================================

        print()
        print("Creating epochs...")

        epochs = mne.Epochs(
            raw,
            epoch_events,
            event_id=event_id,
            tmin=TMIN,
            tmax=TMAX,
            baseline=BASELINE,
            reject=REJECT,
            reject_by_annotation=True,
            preload=True,
            detrend=None,
            verbose=False
        )


        epochs_before = len(
            epoch_events
        )

        epochs_after = len(
            epochs
        )

        epochs_rejected = (
            epochs_before
            - epochs_after
        )


        print()
        print(
            "Epochs before rejection:",
            epochs_before
        )

        print(
            "Epochs retained:",
            epochs_after
        )

        print(
            "Epochs rejected:",
            epochs_rejected
        )


        # ====================================================
        # RETAINED METADATA
        # ====================================================

        if len(epochs.selection) > 0:

            retained_metadata = (
                epoch_metadata.iloc[
                    epochs.selection
                ]
                .reset_index(
                    drop=True
                )
            )

        else:

            retained_metadata = pd.DataFrame()


        # ====================================================
        # OUTPUT NAMES
        # ====================================================

        output_filename = filename.replace(
            "_preprocessed_raw.fif",
            "_epo.fif"
        )

        output_fif = os.path.join(
            OUTPUT_DIR,
            output_filename
        )


        metadata_filename = (
            output_filename.replace(
                "_epo.fif",
                "_events.csv"
            )
        )


        metadata_output = os.path.join(
            OUTPUT_DIR,
            metadata_filename
        )


        # ====================================================
        # SAVE EPOCHS
        # ====================================================

        print()
        print("Saving epochs:")

        print(output_fif)

        epochs.save(
            output_fif,
            overwrite=True,
            verbose=False
        )


        # ====================================================
        # SAVE EVENT METADATA
        # ====================================================

        retained_metadata.to_csv(
            metadata_output,
            index=False
        )


        # ====================================================
        # RESULT
        # ====================================================

        results.append({

            "subject":
                subject,

            "run":
                run,

            "file":
                filename,

            "events_found":
                len(events_data),

            "valid_events":
                len(valid_events),

            "edge_removed":
                edge_removed,

            "epochs_before_rejection":
                epochs_before,

            "epochs_retained":
                epochs_after,

            "epochs_rejected":
                epochs_rejected,

            "channels":
                len(raw.ch_names),

            "sfreq":
                raw.info["sfreq"],

            "duration_seconds":
                raw.times[-1],

            "status":
                "SUCCESS",

            "reason":
                ""
        })


        print()
        print("STATUS: SUCCESS")


    except Exception as e:

        print()
        print("STATUS: FAILED")

        print()
        print(
            "ERROR:",
            str(e)
        )

        traceback.print_exc()


        results.append({

            "subject":
                subject,

            "run":
                run,

            "file":
                filename,

            "events_found":
                "",

            "valid_events":
                "",

            "edge_removed":
                "",

            "epochs_before_rejection":
                "",

            "epochs_retained":
                "",

            "epochs_rejected":
                "",

            "channels":
                "",

            "sfreq":
                "",

            "duration_seconds":
                "",

            "status":
                "FAILED",

            "reason":
                str(e)
        })


# ============================================================
# SAVE MASTER LOG
# ============================================================

results_df = pd.DataFrame(
    results
)

master_log = os.path.join(
    LOG_DIR,
    "epoching_82runs_log.csv"
)

results_df.to_csv(
    master_log,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

success_count = int(
    np.sum(
        results_df["status"]
        == "SUCCESS"
    )
)

failed_count = int(
    np.sum(
        results_df["status"]
        == "FAILED"
    )
)


summary_file = os.path.join(
    LOG_DIR,
    "epoching_summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EPOCHING SUMMARY\n"
    )

    f.write(
        "=" * 70
        + "\n"
    )

    f.write(
        f"Preprocessed files found: "
        f"{len(fif_files)}\n"
    )

    f.write(
        f"Successfully processed: "
        f"{success_count}\n"
    )

    f.write(
        f"Failed: "
        f"{failed_count}\n"
    )

    f.write(
        f"TMIN: {TMIN}\n"
    )

    f.write(
        f"TMAX: {TMAX}\n"
    )

    f.write(
        f"Baseline: {BASELINE}\n"
    )

    f.write(
        "EEG rejection threshold: "
        f"{REJECT['eeg']} V\n"
    )

    f.write(
        "\n"
    )

    f.write(
        "RAW DATA WAS NOT MODIFIED.\n"
    )

    f.write(
        "ORIGINAL SET/FDT FILES WERE NOT MODIFIED.\n"
    )

    f.write(
        "PREPROCESSED FIF FILES WERE NOT MODIFIED.\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print()
print("=" * 80)
print("EPOCHING COMPLETE")
print("=" * 80)

print()
print(
    "Total preprocessed runs:",
    len(fif_files)
)

print(
    "Successfully processed:",
    success_count
)

print(
    "Failed:",
    failed_count
)

print()
print("Epoch output directory:")
print(OUTPUT_DIR)

print()
print("Master log:")
print(master_log)

print()
print("Summary:")
print(summary_file)

print()
print("RAW DATA WAS NOT MODIFIED.")
print("NO SET FILES WERE MODIFIED.")
print("NO FDT FILES WERE MODIFIED.")
print("NO PREPROCESSED FIF FILES WERE MODIFIED.")

print()
print("=" * 80)