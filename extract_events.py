import mne
import os

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = r"C:\Users\Ali\Desktop\BrainPerturbationProject\processed\sub-009_ses-01_task-WorkingMemory_run-1_filtered_raw.fif"


# ============================================================
# LOAD EEG
# ============================================================

print("=" * 70)
print("LOADING FILTERED EEG")
print("=" * 70)

raw = mne.io.read_raw_fif(
    INPUT_FILE,
    preload=False
)

print(raw)


# ============================================================
# LOAD EVENTS FROM FIF
# ============================================================

print("\n" + "=" * 70)
print("CHECKING EVENTS")
print("=" * 70)

events, event_id = mne.events_from_annotations(
    raw
)

print("\nNumber of events:", len(events))

print("\nEvent ID:")
print(event_id)


# ============================================================
# PRINT ALL EVENTS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 50 EVENTS")
print("=" * 70)

for i, event in enumerate(events[:50]):

    sample = event[0]
    event_code = event[2]

    event_name = None

    for name, code in event_id.items():

        if code == event_code:
            event_name = name
            break

    time_sec = sample / raw.info["sfreq"]

    print(
        f"{i:3d} | "
        f"{event_name:30s} | "
        f"sample={sample:7d} | "
        f"time={time_sec:8.3f} sec"
    )


# ============================================================
# EVENT COUNTS
# ============================================================

print("\n" + "=" * 70)
print("EVENT COUNTS")
print("=" * 70)

from collections import Counter

event_names = []

for event in events:

    event_code = event[2]

    for name, code in event_id.items():

        if code == event_code:
            event_names.append(name)
            break

counts = Counter(event_names)

for name, count in counts.items():

    print(
        f"{name:30s} : {count}"
    )


# ============================================================
# SAVE EVENTS
# ============================================================

output_file = r"C:\Users\Ali\Desktop\BrainPerturbationProject\processed\sub-009_ses-01_task-WorkingMemory_run-1_events-eve.fif"

mne.write_events(
    output_file,
    events,
    overwrite=True
)

print("\n" + "=" * 70)
print("EVENT FILE SAVED")
print("=" * 70)

print(output_file)

print("\n[DONE]")