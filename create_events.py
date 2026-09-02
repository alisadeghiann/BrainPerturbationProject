import os
import h5py
import numpy as np
import mne


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

SET_FILE = os.path.join(
    BASE_DIR,
    "data",
    "sub-009",
    "ses-01",
    "eeg",
    "sub-009_ses-01_task-WorkingMemory_run-1_eeg.set"
)

FILTERED_FIF = os.path.join(
    BASE_DIR,
    "processed",
    "sub-009_ses-01_task-WorkingMemory_run-1_filtered_raw.fif"
)

OUTPUT_EVE = os.path.join(
    BASE_DIR,
    "processed",
    "sub-009_ses-01_task-WorkingMemory_run-1_events-eve.fif"
)


# ============================================================
# HDF5 STRING DECODER
# ============================================================

def decode_value(f, value):

    # HDF5 object reference
    if isinstance(value, h5py.Reference):

        if not value:
            return ""

        obj = f[value]

        if isinstance(obj, h5py.Dataset):

            data = obj[()]

            arr = np.asarray(data).squeeze()

            # Single value
            if arr.size == 1:

                item = arr.item()

                if isinstance(item, h5py.Reference):
                    return decode_value(f, item)

                if isinstance(item, bytes):
                    return item.decode(
                        "utf-8",
                        errors="ignore"
                    )

                return str(item)

            # ASCII / Unicode character codes
            if arr.dtype.kind in ("u", "i"):

                try:
                    return "".join(
                        chr(int(x))
                        for x in arr.flatten()
                    )
                except Exception:
                    return str(arr)

            # Bytes
            if arr.dtype.kind == "S":

                try:
                    return b"".join(
                        arr.flatten()
                    ).decode(
                        "utf-8",
                        errors="ignore"
                    )
                except Exception:
                    return str(arr)

            # Unicode
            if arr.dtype.kind == "U":
                return "".join(
                    arr.flatten().tolist()
                )

            return str(arr)

        return str(obj)

    # Bytes
    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="ignore"
        )

    # NumPy scalar
    if isinstance(value, np.generic):
        return decode_value(
            f,
            value.item()
        )

    # NumPy array
    if isinstance(value, np.ndarray):

        arr = np.asarray(value).squeeze()

        if arr.size == 1:
            return decode_value(
                f,
                arr.item()
            )

        if arr.dtype.kind in ("u", "i"):

            try:
                return "".join(
                    chr(int(x))
                    for x in arr.flatten()
                )
            except Exception:
                return str(arr)

        return str(arr)

    return str(value)


# ============================================================
# READ STRING FIELD
# ============================================================

def read_string_field(f, group, field):

    print(f"Reading {field}...")

    dataset = group[field]

    raw = dataset[()]

    arr = np.asarray(raw).reshape(-1)

    values = []

    for value in arr:

        values.append(
            decode_value(f, value)
        )

    return values


# ============================================================
# READ NUMERIC FIELD
# ============================================================

def read_numeric_field(f, group, field):

    print(f"Reading {field}...")

    dataset = group[field]

    raw = dataset[()]

    arr = np.asarray(raw).reshape(-1)

    values = []

    for value in arr:

        # HDF5 reference
        if isinstance(value, h5py.Reference):

            if not value:
                values.append(np.nan)
                continue

            obj = f[value]

            data = np.asarray(
                obj[()]
            ).squeeze()

            if data.size == 0:
                values.append(np.nan)

            else:
                try:
                    values.append(
                        float(data.flat[0])
                    )
                except Exception:
                    values.append(np.nan)

        else:

            try:
                values.append(
                    float(value)
                )
            except Exception:
                values.append(np.nan)

    return np.asarray(
        values,
        dtype=float
    )


# ============================================================
# LOAD ORIGINAL SET
# ============================================================

print("=" * 70)
print("LOADING ORIGINAL EEGLAB SET")
print("=" * 70)

print("SET file:")
print(SET_FILE)

if not os.path.exists(SET_FILE):

    raise FileNotFoundError(
        f"\nSET file not found:\n{SET_FILE}"
    )


with h5py.File(
    SET_FILE,
    "r"
) as f:

    # --------------------------------------------------------
    # SHOW TOP LEVEL
    # --------------------------------------------------------

    print("\nTop-level HDF5 objects:")

    for key in f.keys():
        print(" ", key)

    # --------------------------------------------------------
    # EVENT IS DIRECTLY IN ROOT
    # --------------------------------------------------------

    if "event" not in f:

        raise RuntimeError(
            "The SET file does not contain an 'event' object."
        )

    event_group = f["event"]

    print("\nEvent object found:")
    print(event_group)

    # --------------------------------------------------------
    # EVENT FIELDS
    # --------------------------------------------------------

    print("\nEvent fields:")

    fields = list(
        event_group.keys()
    )

    print(fields)

    # --------------------------------------------------------
    # NUMBER OF EVENTS
    # --------------------------------------------------------

    if "type" in event_group:

        n_events = event_group[
            "type"
        ].shape[0]

    else:

        raise RuntimeError(
            "Event object does not contain 'type'."
        )

    print("\nNumber of events:")
    print(n_events)

    # --------------------------------------------------------
    # READ EVENT FIELDS
    # --------------------------------------------------------

    event_types = read_string_field(
        f,
        event_group,
        "type"
    )

    task_roles = read_string_field(
        f,
        event_group,
        "task_role"
    )

    memory_conditions = read_string_field(
        f,
        event_group,
        "memory_cond"
    )

    letters = read_string_field(
        f,
        event_group,
        "letter"
    )

    trials = read_numeric_field(
        f,
        event_group,
        "trial"
    )

    samples = read_numeric_field(
        f,
        event_group,
        "sample"
    )

    latencies = read_numeric_field(
        f,
        event_group,
        "latency"
    )


# ============================================================
# VERIFY EVENT DATA
# ============================================================

print("\n" + "=" * 70)
print("EVENT DATA CHECK")
print("=" * 70)

print(
    "Types:",
    len(event_types)
)

print(
    "Roles:",
    len(task_roles)
)

print(
    "Memory conditions:",
    len(memory_conditions)
)

print(
    "Letters:",
    len(letters)
)

print(
    "Trials:",
    len(trials)
)

print(
    "Samples:",
    len(samples)
)

print(
    "Latencies:",
    len(latencies)
)


# ============================================================
# FIRST 30 EVENTS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 30 ORIGINAL EVENTS")
print("=" * 70)

print(
    f"{'INDEX':<7}"
    f"{'TYPE':<20}"
    f"{'ROLE':<24}"
    f"{'MEM':<8}"
    f"{'SAMPLE':<10}"
    f"{'LETTER':<8}"
    f"{'TRIAL':<8}"
)

print("-" * 95)

for i in range(
    min(30, n_events)
):

    print(
        f"{i:<7}"
        f"{event_types[i]:<20}"
        f"{task_roles[i]:<24}"
        f"{memory_conditions[i]:<8}"
        f"{samples[i]:<10.0f}"
        f"{letters[i]:<8}"
        f"{trials[i]:<8.0f}"
    )


# ============================================================
# LOAD FILTERED EEG
# ============================================================

print("\n" + "=" * 70)
print("LOADING FILTERED EEG")
print("=" * 70)

print(FILTERED_FIF)

if not os.path.exists(
    FILTERED_FIF
):

    raise FileNotFoundError(
        f"\nFiltered FIF not found:\n{FILTERED_FIF}"
    )


raw = mne.io.read_raw_fif(
    FILTERED_FIF,
    preload=False
)

print(raw)


# ============================================================
# EEG INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("EEG INFORMATION")
print("=" * 70)

print(
    "Sampling rate:",
    raw.info["sfreq"]
)

print(
    "Number of samples:",
    raw.n_times
)

print(
    "Duration:",
    raw.times[-1]
)


# ============================================================
# CHECK EVENT SAMPLES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING EVENT SAMPLES")
print("=" * 70)

valid = (
    np.isfinite(samples)
    &
    (samples >= 0)
    &
    (samples < raw.n_times)
)

print(
    "Total original events:",
    len(samples)
)

print(
    "Valid events:",
    np.sum(valid)
)

print(
    "Invalid events:",
    np.sum(~valid)
)

print(
    "First sample:",
    samples[0]
)

print(
    "Last sample:",
    samples[-1]
)


# ============================================================
# CREATE EVENT ID
# ============================================================

print("\n" + "=" * 70)
print("CREATING EVENT ID")
print("=" * 70)

# We use task_role because it directly describes
# the experimental meaning of each event.

unique_roles = []

for role in task_roles:

    if role not in unique_roles:
        unique_roles.append(role)


event_id = {}

for code, role in enumerate(
    unique_roles,
    start=1
):

    event_id[role] = code


print("\nEvent ID mapping:")

for role, code in event_id.items():

    print(
        f"{code:3d} -> {role}"
    )


# ============================================================
# CREATE MNE EVENTS
# ============================================================

events_list = []

for i in range(
    n_events
):

    if not valid[i]:
        continue

    sample = int(
        round(samples[i])
    )

    role = task_roles[i]

    code = event_id[role]

    events_list.append(
        [
            sample,
            0,
            code
        ]
    )


events = np.asarray(
    events_list,
    dtype=int
)


# ============================================================
# SORT EVENTS
# ============================================================

if len(events) > 0:

    events = events[
        np.argsort(
            events[:, 0]
        )
    ]


# ============================================================
# EVENT COUNTS
# ============================================================

print("\n" + "=" * 70)
print("EVENT COUNTS")
print("=" * 70)

for role, code in event_id.items():

    count = np.sum(
        events[:, 2] == code
    )

    print(
        f"{role:<25} {count}"
    )

print(
    "\nTotal MNE events:",
    len(events)
)


# ============================================================
# FIRST 30 MNE EVENTS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 30 MNE EVENTS")
print("=" * 70)

print(
    f"{'INDEX':<8}"
    f"{'SAMPLE':<12}"
    f"{'TIME(s)':<12}"
    f"{'CODE':<8}"
    f"{'ROLE':<25}"
)

print("-" * 75)

reverse_event_id = {
    code: role
    for role, code in event_id.items()
}

for i, event in enumerate(
    events[:30]
):

    sample = int(
        event[0]
    )

    code = int(
        event[2]
    )

    time_sec = (
        sample /
        raw.info["sfreq"]
    )

    role = reverse_event_id[
        code
    ]

    print(
        f"{i:<8}"
        f"{sample:<12}"
        f"{time_sec:<12.3f}"
        f"{code:<8}"
        f"{role:<25}"
    )


# ============================================================
# SAVE EVENTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING EVENTS")
print("=" * 70)

mne.write_events(
    OUTPUT_EVE,
    events,
    overwrite=True
)

print("\nEvents saved successfully:")

print(
    OUTPUT_EVE
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "Original events:",
    n_events
)

print(
    "Valid events:",
    len(events)
)

print(
    "\nEvents file:"
)

print(
    OUTPUT_EVE
)

print(
    "\nThe next step is EEG epoching."
)