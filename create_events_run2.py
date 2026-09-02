import h5py
import numpy as np
import mne
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

SET_FILE = PROJECT_DIR / (
    r"data\sub-009\ses-01\eeg"
    r"\sub-009_ses-01_task-WorkingMemory_run-2_eeg.set"
)

FILTERED_FILE = PROJECT_DIR / (
    r"processed\sub-009_ses-01_task-WorkingMemory_run-2_filtered_raw.fif"
)

OUTPUT_FILE = PROJECT_DIR / (
    r"processed\sub-009_ses-01_task-WorkingMemory_run-2_events-eve.fif"
)


def decode_string(f, ref):
    obj = f[ref]
    values = np.array(obj[()]).flatten()
    return "".join(chr(int(x)) for x in values)


def decode_number(f, ref):
    obj = f[ref]
    values = np.array(obj[()]).flatten()
    return float(values[0])


print("=" * 70)
print("LOADING RUN 2 EVENTS")
print("=" * 70)

with h5py.File(SET_FILE, "r") as f:

    event = f["event"]

    n_events = event["sample"].shape[0]

    print("Number of events:", n_events)

    event_types = []
    event_roles = []
    event_samples = []

    for i in range(n_events):

        type_ref = event["type"][i, 0]
        role_ref = event["task_role"][i, 0]
        sample_ref = event["sample"][i, 0]

        event_types.append(
            decode_string(f, type_ref)
        )

        event_roles.append(
            decode_string(f, role_ref)
        )

        event_samples.append(
            decode_number(f, sample_ref)
        )


print()
print("=" * 70)
print("FIRST 30 EVENTS")
print("=" * 70)

for i in range(min(30, n_events)):

    print(
        f"{i:<5}"
        f"{event_types[i]:<25}"
        f"{event_roles[i]:<30}"
        f"sample={event_samples[i]:.0f}"
    )


print()
print("=" * 70)
print("LOADING FILTERED EEG")
print("=" * 70)

raw = mne.io.read_raw_fif(
    FILTERED_FILE,
    preload=False
)

print(raw)


print()
print("=" * 70)
print("CREATING EVENT ID")
print("=" * 70)

unique_roles = []

for role in event_roles:
    if role not in unique_roles:
        unique_roles.append(role)

event_id = {
    role: i + 1
    for i, role in enumerate(unique_roles)
}

for role, code in event_id.items():
    print(f"{code:2d} -> {role}")


print()
print("=" * 70)
print("CREATING MNE EVENTS")
print("=" * 70)

events = []

for sample, role in zip(event_samples, event_roles):

    sample = int(round(sample))

    if 0 <= sample < raw.n_times:

        events.append([
            sample,
            0,
            event_id[role]
        ])

events = np.array(events, dtype=int)

print("Total MNE events:", len(events))


print()
print("=" * 70)
print("EVENT COUNTS")
print("=" * 70)

for role, code in event_id.items():

    count = np.sum(events[:, 2] == code)

    print(
        f"{role:<30}: {count}"
    )


print()
print("=" * 70)
print("SAVING EVENTS")
print("=" * 70)

mne.write_events(
    OUTPUT_FILE,
    events,
    overwrite=True
)

print()
print("[DONE]")
print("Events saved successfully:")
print(OUTPUT_FILE)