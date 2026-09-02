import os
import re
import csv
import h5py
import numpy as np

# ============================================================
# PATHS
# ============================================================

DATA = r"C:\Users\Ali\Desktop\BrainPerturbationProject\data"
QC_TEMPORAL = r"C:\Users\Ali\Desktop\BrainPerturbationProject\qc\sub004_temporal"

OUT_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject\qc\sub004_temporal"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def decode_h5_string(obj):
    """
    Decode EEGLAB MATLAB v7.3 string-like HDF5 objects.
    """
    try:
        arr = np.array(obj)

        if arr.dtype.kind in ("u", "i"):
            vals = arr.flatten().tolist()

            # ASCII-like character codes
            if all(0 <= int(x) <= 65535 for x in vals):
                return "".join(chr(int(x)) for x in vals).rstrip("\x00")

        if arr.dtype.kind in ("S", "U", "O"):
            return str(arr.flatten()[0])

    except Exception:
        pass

    return ""


def get_scalar(f, key, default=np.nan):
    try:
        x = np.array(f[key])
        return float(x.flatten()[0])
    except Exception:
        return default


def get_event_field(event_group, field_name):
    """
    Try to extract an EEGLAB event field.
    """
    if field_name not in event_group:
        return []

    obj = event_group[field_name]

    try:
        # Direct dataset
        if isinstance(obj, h5py.Dataset):
            arr = np.array(obj)

            result = []

            for x in arr.flatten():
                try:
                    if np.issubdtype(arr.dtype, np.number):
                        result.append(float(x))
                    else:
                        result.append(str(x))
                except Exception:
                    result.append(str(x))

            return result

        # HDF5 object references / groups
        result = []

        arr = np.array(obj)

        for ref in arr.flatten():

            try:
                target = f[ref]

                if isinstance(target, h5py.Dataset):
                    value = decode_h5_string(target)
                else:
                    value = ""

                result.append(value)

            except Exception:
                result.append("")

        return result

    except Exception:
        return []


# ============================================================
# FIND SUB-004 FILES
# ============================================================

sub_dir = os.path.join(
    DATA,
    "sub-004",
    "ses-01",
    "eeg"
)

set_files = sorted([
    os.path.join(sub_dir, x)
    for x in os.listdir(sub_dir)
    if x.endswith(".set")
    and "WorkingMemory" in x
])

if len(set_files) == 0:
    raise RuntimeError("No sub-004 WorkingMemory .set files found.")

print("=" * 80)
print("SUB-004 ARTIFACT ↔ EVENT OVERLAP ANALYSIS")
print("=" * 80)

print(f"\nFound {len(set_files)} SET files.")

# ============================================================
# LOAD TEMPORAL QC
# ============================================================

temporal_csv = os.path.join(
    QC_TEMPORAL,
    "sub004_temporal_qc.csv"
)

if not os.path.exists(temporal_csv):
    raise FileNotFoundError(
        f"Temporal QC file not found:\n{temporal_csv}"
    )

rows = []

with open(
    temporal_csv,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        rows.append(row)

print(f"Loaded {len(rows)} temporal QC windows.")

# ============================================================
# OUTPUTS
# ============================================================

overlap_rows = []

summary_rows = []

# ============================================================
# PROCESS EACH RUN
# ============================================================

for set_path in set_files:

    filename = os.path.basename(set_path)

    match = re.search(r"run-(\d+)", filename)

    if not match:
        print(f"WARNING: Could not determine run: {filename}")
        continue

    run = int(match.group(1))

    print("\n" + "=" * 80)
    print(f"RUN {run}")
    print("=" * 80)

    # --------------------------------------------------------
    # TEMPORAL QC WINDOWS FOR THIS RUN
    # --------------------------------------------------------

    run_windows = [
        r for r in rows
        if int(float(r["run"])) == run
    ]

    if len(run_windows) == 0:
        print("No temporal QC windows found.")
        continue

    # --------------------------------------------------------
    # BASIC WINDOW SUMMARY
    # --------------------------------------------------------

    normal = 0
    review = 0
    global_artifact = 0

    for r in run_windows:

        status = r["status"].strip()

        if status == "NORMAL":
            normal += 1

        elif status == "REVIEW":
            review += 1

        elif status == "GLOBAL_ARTIFACT":
            global_artifact += 1

    total_windows = len(run_windows)

    print(f"Total windows:       {total_windows}")
    print(f"NORMAL:              {normal}")
    print(f"REVIEW:              {review}")
    print(f"GLOBAL_ARTIFACT:     {global_artifact}")

    # --------------------------------------------------------
    # RECORDING INFO + EVENTS
    # --------------------------------------------------------

    with h5py.File(set_path, "r") as f:

        srate = get_scalar(f, "srate")
        pnts = get_scalar(f, "pnts")
        trials = get_scalar(f, "trials")
        xmin = get_scalar(f, "xmin")
        xmax = get_scalar(f, "xmax")

        print(f"Sampling rate:       {srate}")
        print(f"Points:              {pnts}")
        print(f"Trials:              {trials}")
        print(f"Duration:            {xmax - xmin:.3f} sec")

        # ----------------------------------------------------
        # EVENTS
        # ----------------------------------------------------

        events = []

        if "event" in f:

            event_group = f["event"]

            latencies = get_event_field(
                event_group,
                "latency"
            )

            event_types = get_event_field(
                event_group,
                "type"
            )

            memory_cond = get_event_field(
                event_group,
                "memory_cond"
            )

            task_role = get_event_field(
                event_group,
                "task_role"
            )

            letters = get_event_field(
                event_group,
                "letter"
            )

            n_events = len(latencies)

            print(f"Events:              {n_events}")

            for i in range(n_events):

                try:
                    latency_samples = float(
                        latencies[i]
                    )
                except Exception:
                    continue

                # EEGLAB latency is sample-based, normally 1-indexed
                time_sec = (
                    latency_samples - 1
                ) / srate

                event = {
                    "index": i + 1,
                    "latency_samples": latency_samples,
                    "time_sec": time_sec,
                    "type": (
                        event_types[i]
                        if i < len(event_types)
                        else ""
                    ),
                    "memory_cond": (
                        memory_cond[i]
                        if i < len(memory_cond)
                        else ""
                    ),
                    "task_role": (
                        task_role[i]
                        if i < len(task_role)
                        else ""
                    ),
                    "letter": (
                        letters[i]
                        if i < len(letters)
                        else ""
                    )
                }

                events.append(event)

        else:
            print("WARNING: event group not found.")

    # --------------------------------------------------------
    # EVENT ↔ WINDOW OVERLAP
    # --------------------------------------------------------

    run_event_count = len(events)

    events_in_artifact = 0
    events_in_review = 0
    events_in_normal = 0

    for event in events:

        t = event["time_sec"]

        matched_status = "OUTSIDE_QC"

        matched_window = None

        for r in run_windows:

            start = float(r["start_sec"])
            end = float(r["end_sec"])

            if start <= t < end:

                matched_status = r["status"].strip()
                matched_window = r

                break

        if matched_status == "GLOBAL_ARTIFACT":
            events_in_artifact += 1

        elif matched_status == "REVIEW":
            events_in_review += 1

        elif matched_status == "NORMAL":
            events_in_normal += 1

        overlap_rows.append({
            "subject": "sub-004",
            "run": run,
            "event_index": event["index"],
            "event_time_sec": round(t, 4),
            "event_type": event["type"],
            "memory_cond": event["memory_cond"],
            "task_role": event["task_role"],
            "letter": event["letter"],
            "qc_status": matched_status,
            "qc_start_sec": (
                matched_window["start_sec"]
                if matched_window
                else ""
            ),
            "qc_end_sec": (
                matched_window["end_sec"]
                if matched_window
                else ""
            )
        })

    # --------------------------------------------------------
    # ARTIFACT TIME
    # --------------------------------------------------------

    artifact_seconds = 0.0
    review_seconds = 0.0
    normal_seconds = 0.0

    for r in run_windows:

        start = float(r["start_sec"])
        end = float(r["end_sec"])

        duration = end - start

        status = r["status"].strip()

        if status == "GLOBAL_ARTIFACT":
            artifact_seconds += duration

        elif status == "REVIEW":
            review_seconds += duration

        elif status == "NORMAL":
            normal_seconds += duration

    total_seconds = (
        artifact_seconds
        + review_seconds
        + normal_seconds
    )

    artifact_pct = (
        100 * artifact_seconds / total_seconds
        if total_seconds > 0
        else 0
    )

    review_pct = (
        100 * review_seconds / total_seconds
        if total_seconds > 0
        else 0
    )

    # --------------------------------------------------------
    # PRINT RUN SUMMARY
    # --------------------------------------------------------

    print("\nEVENT OVERLAP")
    print("-" * 80)

    print(
        f"Total events:                    "
        f"{run_event_count}"
    )

    print(
        f"Events in NORMAL windows:       "
        f"{events_in_normal}"
    )

    print(
        f"Events in REVIEW windows:       "
        f"{events_in_review}"
    )

    print(
        f"Events in GLOBAL_ARTIFACT:      "
        f"{events_in_artifact}"
    )

    print("\nTIME COVERAGE")
    print("-" * 80)

    print(
        f"GLOBAL_ARTIFACT: "
        f"{artifact_seconds:.2f} sec "
        f"({artifact_pct:.2f}%)"
    )

    print(
        f"REVIEW:          "
        f"{review_seconds:.2f} sec "
        f"({review_pct:.2f}%)"
    )

    print(
        f"NORMAL:          "
        f"{normal_seconds:.2f} sec"
    )

    # --------------------------------------------------------
    # WORST ARTIFACT WINDOWS
    # --------------------------------------------------------

    artifact_windows = [
        r for r in run_windows
        if r["status"].strip() == "GLOBAL_ARTIFACT"
    ]

    artifact_windows.sort(
        key=lambda x: float(x["global_bad_channels"]),
        reverse=True
    )

    print("\nGLOBAL ARTIFACT WINDOWS")
    print("-" * 80)

    if len(artifact_windows) == 0:

        print("None.")

    else:

        for r in artifact_windows:

            print(
                f"{float(r['start_sec']):8.1f} - "
                f"{float(r['end_sec']):8.1f} sec | "
                f"bad_channels="
                f"{r['global_bad_channels']} | "
                f"status={r['status']}"
            )

    # --------------------------------------------------------
    # SUMMARY RECORD
    # --------------------------------------------------------

    summary_rows.append({
        "subject": "sub-004",
        "run": run,
        "total_windows": total_windows,
        "normal_windows": normal,
        "review_windows": review,
        "global_artifact_windows": global_artifact,
        "total_events": run_event_count,
        "events_normal": events_in_normal,
        "events_review": events_in_review,
        "events_global_artifact": events_in_artifact,
        "artifact_seconds": round(
            artifact_seconds, 3
        ),
        "artifact_percent": round(
            artifact_pct, 3
        ),
        "review_seconds": round(
            review_seconds, 3
        ),
        "review_percent": round(
            review_pct, 3
        )
    })


# ============================================================
# SAVE EVENT OVERLAP CSV
# ============================================================

event_csv = os.path.join(
    OUT_DIR,
    "sub004_artifact_event_overlap.csv"
)

fieldnames = [
    "subject",
    "run",
    "event_index",
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
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(overlap_rows)


# ============================================================
# SAVE RUN SUMMARY
# ============================================================

summary_csv = os.path.join(
    OUT_DIR,
    "sub004_artifact_event_run_summary.csv"
)

summary_fields = list(summary_rows[0].keys())

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

    writer.writerows(summary_rows)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("ARTIFACT ↔ EVENT ANALYSIS COMPLETE")
print("=" * 80)

print("\nSaved:")
print(event_csv)
print(summary_csv)

print("\nRUN SUMMARY")
print("-" * 80)

for r in summary_rows:

    print(
        f"Run {r['run']}: "
        f"artifact={r['artifact_percent']:.2f}% | "
        f"review={r['review_percent']:.2f}% | "
        f"events={r['total_events']} | "
        f"artifact_events={r['events_global_artifact']}"
    )

print("\n")
print("=" * 80)
print("IMPORTANT")
print("=" * 80)

print("RAW DATA WAS NOT MODIFIED.")
print("No channels were removed.")
print("No samples were deleted.")
print("No interpolation was performed.")
print("This script only analyzes artifact/event overlap.")
print("=" * 80)