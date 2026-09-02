import mne
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

RAW_FILE = PROJECT_DIR / (
    r"processed\sub-009_ses-01_task-WorkingMemory_run-2_filtered_raw.fif"
)

EVENT_FILE = PROJECT_DIR / (
    r"processed\sub-009_ses-01_task-WorkingMemory_run-2_events-eve.fif"
)

OUTPUT_FILE = PROJECT_DIR / (
    r"processed\sub-009_ses-01_task-WorkingMemory_run-2_epochs-epo.fif"
)

print("=" * 70)
print("LOADING RUN 2 FILTERED EEG")
print("=" * 70)

raw = mne.io.read_raw_fif(
    RAW_FILE,
    preload=True
)

print(raw)

print()
print("=" * 70)
print("LOADING EVENTS")
print("=" * 70)

events = mne.read_events(EVENT_FILE)

print("Number of events:", len(events))


event_id = {
    "to_remember": 2,
    "to_ignore": 3,
    "work_memory": 4,
    "probe_target": 5
}

print()
print("Selected events:")
for name, code in event_id.items():
    print(f"{name:<20}: code {code}")


print()
print("=" * 70)
print("CREATING EPOCHS")
print("=" * 70)

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=-0.5,
    tmax=1.5,
    baseline=(-0.5, 0),
    preload=True,
    reject_by_annotation=True
)

print()
print("=" * 70)
print("EPOCH INFORMATION")
print("=" * 70)

print("Number of epochs:", len(epochs))
print("Number of channels:", len(epochs.ch_names))
print("Sampling rate:", epochs.info["sfreq"])
print("Time range:", epochs.tmin, "to", epochs.tmax)
print("Data shape:", epochs.get_data().shape)


print()
print("=" * 70)
print("EPOCH COUNTS")
print("=" * 70)

for condition in event_id:

    count = len(epochs[condition])

    print(
        f"{condition:<20}: {count}"
    )


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