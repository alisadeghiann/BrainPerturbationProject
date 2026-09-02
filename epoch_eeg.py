import os
import mne

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

SUBJECT = "sub-009"
RUN = "run-1"

RAW_FILE = os.path.join(
    PROCESSED_DIR,
    f"{SUBJECT}_ses-01_task-WorkingMemory_{RUN}_filtered_raw.fif"
)

EVENT_FILE = os.path.join(
    PROCESSED_DIR,
    f"{SUBJECT}_ses-01_task-WorkingMemory_{RUN}_events-eve.fif"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    f"{SUBJECT}_ses-01_task-WorkingMemory_{RUN}_epochs.fif"
)

# ============================================================
# EPOCH SETTINGS
# ============================================================

# Epoch from 500 ms before event until 1500 ms after event
TMIN = -0.5
TMAX = 1.5

# Baseline correction
BASELINE = (None, 0)

# ============================================================
# LOAD FILTERED EEG
# ============================================================

print("=" * 70)
print("LOADING FILTERED EEG")
print("=" * 70)

raw = mne.io.read_raw_fif(
    RAW_FILE,
    preload=True
)

print(raw)

print()
print("=" * 70)
print("EEG INFORMATION")
print("=" * 70)

print("Channels:", len(raw.ch_names))
print("Sampling rate:", raw.info["sfreq"])
print("Duration:", raw.times[-1])
print("Samples:", raw.n_times)

# ============================================================
# LOAD EVENTS
# ============================================================

print()
print("=" * 70)
print("LOADING EVENTS")
print("=" * 70)

events = mne.read_events(EVENT_FILE)

print("Number of events:", len(events))

# ============================================================
# EVENT ID
# ============================================================

event_id = {
    "fixate": 1,
    "to_remember": 2,
    "to_ignore": 3,
    "work_memory": 4,
    "probe_target": 5,
    "remembered_correct": 6,
    "feedback_correct": 7,
    "indicate_ready": 8,
    "ignored_incorrect": 9,
    "feedback_incorrect": 10,
    "probe_not_shown": 11,
    "ignored_correct": 12,
}

print()
print("=" * 70)
print("EVENT ID")
print("=" * 70)

for name, code in event_id.items():
    print(f"{code:2d} -> {name}")

# ============================================================
# SELECT MAIN WORKING-MEMORY CONDITIONS
# ============================================================

selected_event_id = {
    "to_remember": 2,
    "to_ignore": 3,
    "work_memory": 4,
    "probe_target": 5,
}

print()
print("=" * 70)
print("SELECTED EVENTS FOR EPOCHING")
print("=" * 70)

for name, code in selected_event_id.items():
    count = (events[:, 2] == code).sum()
    print(f"{name:20s}: {count}")

# ============================================================
# CREATE EPOCHS
# ============================================================

print()
print("=" * 70)
print("CREATING EPOCHS")
print("=" * 70)

epochs = mne.Epochs(
    raw,
    events,
    event_id=selected_event_id,
    tmin=TMIN,
    tmax=TMAX,
    baseline=BASELINE,
    preload=True,
    reject_by_annotation=False
)

print()
print("=" * 70)
print("EPOCH INFORMATION")
print("=" * 70)

print("Number of epochs:", len(epochs))
print("Epoch time range:", epochs.tmin, "to", epochs.tmax)
print("Number of channels:", len(epochs.ch_names))
print("Epoch data shape:", epochs.get_data().shape)

# ============================================================
# EPOCH COUNTS
# ============================================================

print()
print("=" * 70)
print("EPOCH COUNTS")
print("=" * 70)

for condition in selected_event_id:
    try:
        print(f"{condition:20s}: {len(epochs[condition])}")
    except:
        print(f"{condition:20s}: 0")

# ============================================================
# SAVE EPOCHS
# ============================================================

print()
print("=" * 70)
print("SAVING EPOCHS")
print("=" * 70)

epochs.save(
    OUTPUT_FILE,
    overwrite=True
)

print()
print("[DONE]")
print("Epochs saved successfully:")
print(OUTPUT_FILE)

# ============================================================
# BASIC QUALITY CHECK
# ============================================================

print()
print("=" * 70)
print("BASIC EPOCH QUALITY CHECK")
print("=" * 70)

data = epochs.get_data()

print("Final epoch shape:", data.shape)
print("NaN count:", data != data)

print()
print("Epoching completed successfully.")