import os
import re
import csv
import h5py
import numpy as np


# ============================================================
# PATHS
# ============================================================

DATA = r"C:\Users\Ali\Desktop\BrainPerturbationProject\data"

TEMPORAL_QC = (
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
    r"\qc\sub004_temporal\sub004_temporal_qc.csv"
)

OUT_DIR = (
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
    r"\qc\sub004_temporal"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# HDF5 MATLAB V7.3 HELPERS
# ============================================================

def dereference_value(f, ref):
    """
    Resolve one MATLAB v7.3 HDF5 object reference.
    """

    try:
        obj = f[ref]

        if isinstance(obj, h5py.Dataset):

            arr = np.array(obj)

            # Numeric value
            if np.issubdtype(arr.dtype, np.number):
                if arr.size == 1:
                    return float(arr.flatten()[0])

                return arr

            # Character codes
            if arr.dtype.kind in ("u", "i"):
                vals = arr.flatten().tolist()

                try:
                    return "".join(
                        chr(int(x))
                        for x in vals
                        if int(x) != 0
                    )
                except Exception:
                    return str(arr)

            # String/object
            return str(arr)

        return ""

    except Exception:
        return ""


def read_hdf5_field(f, group, field):
    """
    Read an EEGLAB HDF5 field that may contain
    object references.
    """

    if field not in group:
        return []

    ds = group[field]

    arr = np.array(ds)

    values = []

    for item in arr.flatten():

        # Object reference
        if isinstance(
            item,
            h5py.h5r.Reference
        ):
            values.append(
                dereference_value(f, item)
            )

        else:
            try:
                values.append(float(item))
            except Exception:
                values.append(str(item))

    return values


def scalar(f, key, default=np.nan):

    if key not in f:
        return default

    try:
        arr = np.array(f[key])

        return float(arr.flatten()[0])

    except Exception:
        return default


# ============================================================
# FIND SUB-004 SET FILES
# ============================================================

EEG_DIR = os.path.join(
    DATA,
    "sub-004",
    "ses-01",
    "eeg"
)

set_files = sorted(
    [
        os.path.join(EEG_DIR, f)
        for f in os.listdir(EEG_DIR)
        if f.endswith(".set")
        and "WorkingMemory" in f
    ]
)

print("=" * 80)
print("SUB-004 ARTIFACT ↔ EVENT OVERLAP ANALYSIS V2")
print("=" * 80)

print()
print("SET files found:", len(set_files))

if len(set_files) != 4:

    print(
        "WARNING: Expected 4 runs "
        f"but found {len(set_files)}"
    )


# ============================================================
# LOAD TEMPORAL QC
# ============================================================

if not os.path.exists(TEMPORAL_QC):

    raise FileNotFoundError(
        "\nTemporal QC file not found:\n"
        + TEMPORAL_QC
    )


temporal_rows = []

with open(
    TEMPORAL_QC,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        temporal_rows.append(row)


print(
    "Temporal QC windows loaded:",
    len(temporal_rows)
)


# ============================================================
# OUTPUT STORAGE
# ============================================================

event_rows = []
run_summary = []


# ============================================================
# PROCESS EACH RUN
# ============================================================

for set_path in set_files:

    filename = os.path.basename(set_path)

    match = re.search(
        r"run-(\d+)",
        filename
    )

    if not match:
        continue

    run = int(match.group(1))

    print()
    print("=" * 80)
    print(f"RUN {run}")
    print("=" * 80)

    # --------------------------------------------------------
    # TEMPORAL WINDOWS
    # --------------------------------------------------------

    windows = [
        r
        for r in temporal_rows
        if int(float(r["run"])) == run
    ]

    print(
        "Temporal windows:",
        len(windows)
    )

    # --------------------------------------------------------
    # RECORDING INFORMATION
    # --------------------------------------------------------

    with h5py.File(
        set_path,
        "r"
    ) as f:

        srate = scalar(
            f,
            "srate"
        )

        pnts = scalar(
            f,
            "pnts"
        )

        trials = scalar(
            f,
            "trials"
        )

        xmin = scalar(
            f,
            "xmin",
            0
        )

        xmax = scalar(
            f,
            "xmax"
        )

        duration = xmax - xmin

        print(
            f"Sampling rate: {srate}"
        )

        print(
            f"Points:        {pnts}"
        )

        print(
            f"Trials:        {trials}"
        )

        print(
            f"Duration:      {duration:.3f} sec"
        )

        # ----------------------------------------------------
        # EVENTS
        # ----------------------------------------------------

        events = []

        if "event" in f:

            event_group = f["event"]

            latencies = read_hdf5_field(
                f,
                event_group,
                "latency"
            )

            types = read_hdf5_field(
                f,
                event_group,
                "type"
            )

            memory_cond = read_hdf5_field(
                f,
                event_group,
                "memory_cond"
            )

            task_role = read_hdf5_field(
                f,
                event_group,
                "task_role"
            )

            letters = read_hdf5_field(
                f,
                event_group,
                "letter"
            )

            print(
                "Raw latency entries:",
                len(latencies)
            )

            # ------------------------------------------------
            # CONVERT LATENCIES
            # ------------------------------------------------

            for i, latency in enumerate(
                latencies
            ):

                try:

                    latency_samples = float(
                        latency
                    )

                except Exception:

                    continue

                # EEGLAB latency is 1-based
                time_sec = (
                    latency_samples - 1
                ) / srate

                event = {

                    "event_index":
                        i + 1,

                    "latency_samples":
                        latency_samples,

                    "time_sec":
                        time_sec,

                    "type":
                        types[i]
                        if i < len(types)
                        else "",

                    "memory_cond":
                        memory_cond[i]
                        if i < len(memory_cond)
                        else "",

                    "task_role":
                        task_role[i]
                        if i < len(task_role)
                        else "",

                    "letter":
                        letters[i]
                        if i < len(letters)
                        else ""
                }

                events.append(event)

        else:

            print(
                "WARNING: event group missing."
            )

    print(
        "Successfully decoded events:",
        len(events)
    )

    # --------------------------------------------------------
    # SHOW FIRST EVENTS
    # --------------------------------------------------------

    print()
    print("FIRST 10 EVENTS")
    print("-" * 80)

    for e in events[:10]:

        print(
            f"{e['event_index']:4d} | "
            f"{e['time_sec']:9.3f} sec | "
            f"type={e['type']} | "
            f"memory={e['memory_cond']} | "
            f"role={e['task_role']}"
        )

    # --------------------------------------------------------
    # WINDOW STATUS COUNTS
    # --------------------------------------------------------

    normal_windows = 0
    review_windows = 0
    artifact_windows = 0

    normal_seconds = 0
    review_seconds = 0
    artifact_seconds = 0

    for w in windows:

        start = float(
            w["start_sec"]
        )

        end = float(
            w["end_sec"]
        )

        dur = end - start

        status = w[
            "status"
        ].strip()

        if status == "NORMAL":

            normal_windows += 1
            normal_seconds += dur

        elif status == "REVIEW":

            review_windows += 1
            review_seconds += dur

        elif status == "GLOBAL_ARTIFACT":

            artifact_windows += 1
            artifact_seconds += dur

    # --------------------------------------------------------
    # EVENT ↔ QC OVERLAP
    # --------------------------------------------------------

    events_normal = 0
    events_review = 0
    events_artifact = 0
    events_outside = 0

    for event in events:

        t = event["time_sec"]

        status = "OUTSIDE_QC"

        matched_start = ""
        matched_end = ""

        for w in windows:

            start = float(
                w["start_sec"]
            )

            end = float(
                w["end_sec"]
            )

            if start <= t < end:

                status = w[
                    "status"
                ].strip()

                matched_start = start
                matched_end = end

                break

        if status == "NORMAL":

            events_normal += 1

        elif status == "REVIEW":

            events_review += 1

        elif status == "GLOBAL_ARTIFACT":

            events_artifact += 1

        else:

            events_outside += 1

        event_rows.append({

            "subject":
                "sub-004",

            "run":
                run,

            "event_index":
                event["event_index"],

            "latency_samples":
                event["latency_samples"],

            "event_time_sec":
                round(
                    event["time_sec"],
                    6
                ),

            "event_type":
                event["type"],

            "memory_cond":
                event["memory_cond"],

            "task_role":
                event["task_role"],

            "letter":
                event["letter"],

            "qc_status":
                status,

            "qc_start_sec":
                matched_start,

            "qc_end_sec":
                matched_end
        })

    # --------------------------------------------------------
    # PERCENTAGES
    # --------------------------------------------------------

    total_window_seconds = (
        normal_seconds
        + review_seconds
        + artifact_seconds
    )

    artifact_percent = (
        100 * artifact_seconds
        / total_window_seconds
        if total_window_seconds > 0
        else 0
    )

    review_percent = (
        100 * review_seconds
        / total_window_seconds
        if total_window_seconds > 0
        else 0
    )

    if len(events) > 0:

        event_artifact_percent = (
            100
            * events_artifact
            / len(events)
        )

        event_review_percent = (
            100
            * events_review
            / len(events)
        )

    else:

        event_artifact_percent = 0
        event_review_percent = 0

    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print()
    print("EVENT OVERLAP")
    print("-" * 80)

    print(
        f"Total events:                 "
        f"{len(events)}"
    )

    print(
        f"Events in NORMAL:             "
        f"{events_normal}"
    )

    print(
        f"Events in REVIEW:             "
        f"{events_review}"
    )

    print(
        f"Events in GLOBAL_ARTIFACT:    "
        f"{events_artifact}"
    )

    print(
        f"Events outside QC:            "
        f"{events_outside}"
    )

    print()
    print("EVENT PERCENTAGES")
    print("-" * 80)

    print(
        f"Artifact events: "
        f"{event_artifact_percent:.2f}%"
    )

    print(
        f"Review events:   "
        f"{event_review_percent:.2f}%"
    )

    print()
    print("TIME COVERAGE")
    print("-" * 80)

    print(
        f"GLOBAL_ARTIFACT: "
        f"{artifact_seconds:.2f} sec "
        f"({artifact_percent:.2f}%)"
    )

    print(
        f"REVIEW:          "
        f"{review_seconds:.2f} sec "
        f"({review_percent:.2f}%)"
    )

    print(
        f"NORMAL:          "
        f"{normal_seconds:.2f} sec"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    run_summary.append({

        "subject":
            "sub-004",

        "run":
            run,

        "total_windows":
            len(windows),

        "normal_windows":
            normal_windows,

        "review_windows":
            review_windows,

        "global_artifact_windows":
            artifact_windows,

        "total_events":
            len(events),

        "events_normal":
            events_normal,

        "events_review":
            events_review,

        "events_global_artifact":
            events_artifact,

        "events_outside_qc":
            events_outside,

        "artifact_seconds":
            round(
                artifact_seconds,
                3
            ),

        "artifact_percent":
            round(
                artifact_percent,
                3
            ),

        "review_seconds":
            round(
                review_seconds,
                3
            ),

        "review_percent":
            round(
                review_percent,
                3
            ),

        "artifact_event_percent":
            round(
                event_artifact_percent,
                3
            ),

        "review_event_percent":
            round(
                event_review_percent,
                3
            )
    })


# ============================================================
# SAVE EVENT CSV
# ============================================================

event_csv = os.path.join(
    OUT_DIR,
    "sub004_artifact_event_overlap_v2.csv"
)

event_fields = [

    "subject",
    "run",
    "event_index",
    "latency_samples",
    "event_time_sec",
    "event_type",
    "memory_cond",
    "task_role",
    "letter",
    "qc_status",
    "qc_start_sec",
    "qc_end_sec"
]

with open(
    event_csv,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=event_fields
    )

    writer.writeheader()

    writer.writerows(
        event_rows
    )


# ============================================================
# SAVE RUN SUMMARY
# ============================================================

summary_csv = os.path.join(
    OUT_DIR,
    "sub004_artifact_event_run_summary_v2.csv"
)

summary_fields = list(
    run_summary[0].keys()
)

with open(
    summary_csv,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_fields
    )

    writer.writeheader()

    writer.writerows(
        run_summary
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("FINAL RUN SUMMARY")
print("=" * 80)

print()

for r in run_summary:

    print(
        f"Run {r['run']} | "
        f"windows={r['total_windows']} | "
        f"artifact={r['artifact_percent']:.2f}% | "
        f"review={r['review_percent']:.2f}% | "
        f"events={r['total_events']} | "
        f"artifact_events="
        f"{r['events_global_artifact']} "
        f"({r['artifact_event_percent']:.2f}%)"
    )

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)

print()
print("Saved:")
print(event_csv)
print(summary_csv)

print()
print("RAW DATA WAS NOT MODIFIED.")
print("NO CHANNELS WERE REMOVED.")
print("NO SAMPLES WERE DELETED.")
print("NO INTERPOLATION WAS PERFORMED.")