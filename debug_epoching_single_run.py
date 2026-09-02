import os
import numpy as np
import mne

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

FIF_FILE = os.path.join(
    BASE,
    "preprocessed",
    "sub-001_ses-01_task-WorkingMemory_run-1_preprocessed_raw.fif"
)

print("=" * 80)
print("SINGLE-RUN EPOCH DEBUG")
print("=" * 80)
print()

if not os.path.exists(FIF_FILE):
    raise FileNotFoundError(FIF_FILE)

print("Loading:")
print(FIF_FILE)
print()

raw = mne.io.read_raw_fif(
    FIF_FILE,
    preload=True,
    verbose=False
)

print("=" * 80)
print("RAW INFORMATION")
print("=" * 80)

print("Channels:", len(raw.ch_names))
print("Sampling rate:", raw.info["sfreq"])
print("Samples:", raw.n_times)
print("Duration:", raw.times[-1])
print()

print("=" * 80)
print("ANNOTATIONS")
print("=" * 80)

print("Number of annotations:", len(raw.annotations))
print()

for i, ann in enumerate(raw.annotations[:30]):
    print(
        f"{i+1:4d} | "
        f"onset={ann['onset']:.4f} | "
        f"duration={ann['duration']:.4f} | "
        f"description={ann['description']}"
    )

if len(raw.annotations) > 30:
    print("...")
    print("Only first 30 annotations shown.")

print()

print("=" * 80)
print("EVENT EXTRACTION")
print("=" * 80)

events, event_id = mne.events_from_annotations(
    raw,
    verbose=False
)

print("Events shape:", events.shape)
print("Number of events:", len(events))
print()

print("Event ID:")
for key, value in event_id.items():
    print(f"  {key} -> {value}")

print()

if len(events) == 0:
    print("ERROR: ZERO EVENTS FOUND")
    raise SystemExit(1)

print("=" * 80)
print("FIRST 30 EVENTS")
print("=" * 80)

for i, event in enumerate(events[:30]):
    print(
        f"{i+1:4d} | "
        f"sample={event[0]} | "
        f"time={event[0] / raw.info['sfreq']:.4f} | "
        f"previous={event[1]} | "
        f"code={event[2]}"
    )

print()

print("=" * 80)
print("EVENT CODE COUNTS")
print("=" * 80)

unique_codes, counts = np.unique(events[:, 2], return_counts=True)

for code, count in zip(unique_codes, counts):
    names = [
        name for name, value in event_id.items()
        if value == code
    ]

    name = names[0] if names else "UNKNOWN"

    print(
        f"code={code:4d} | "
        f"count={count:4d} | "
        f"name={name}"
    )

print()

# ------------------------------------------------------------------
# Choose usable event codes
# ------------------------------------------------------------------

usable_event_id = {}

for name, code in event_id.items():

    name_lower = name.lower()

    # We are interested in actual experimental events,
    # not continuous/background markers.
    if any(x in name_lower for x in [
        "show_cross",
        "sound_buzz",
        "sound_beep"
    ]):
        continue

    usable_event_id[name] = code

print("=" * 80)
print("USABLE EVENT TYPES")
print("=" * 80)

if len(usable_event_id) == 0:
    print("No filtered event types were selected.")
    print("Using ALL event types instead.")
    usable_event_id = event_id.copy()

for name, code in usable_event_id.items():
    print(f"{name} -> {code}")

print()

# ------------------------------------------------------------------
# Determine event timing
# ------------------------------------------------------------------

print("=" * 80)
print("EVENT TIMING")
print("=" * 80)

selected_codes = set(usable_event_id.values())

selected_events = events[
    np.isin(events[:, 2], list(selected_codes))
]

print("Selected events:", len(selected_events))

if len(selected_events) > 0:

    first_time = selected_events[0, 0] / raw.info["sfreq"]
    last_time = selected_events[-1, 0] / raw.info["sfreq"]

    print("First event:", first_time, "sec")
    print("Last event :", last_time, "sec")

    if len(selected_events) > 1:
        diffs = np.diff(selected_events[:, 0]) / raw.info["sfreq"]

        print("Median event interval:", np.median(diffs), "sec")
        print("Minimum event interval:", np.min(diffs), "sec")
        print("Maximum event interval:", np.max(diffs), "sec")

print()

# ------------------------------------------------------------------
# TEST 1: Epoching WITHOUT rejection
# ------------------------------------------------------------------

print("=" * 80)
print("TEST 1: EPOCHING WITHOUT REJECTION")
print("=" * 80)

# Broad window deliberately chosen for debugging.
# This test does NOT reject epochs based on amplitude.

tmin = -0.2
tmax = 0.8

print("tmin:", tmin)
print("tmax:", tmax)
print()

epochs_no_reject = mne.Epochs(
    raw,
    selected_events,
    event_id=usable_event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True,
    reject=None,
    flat=None,
    reject_by_annotation=False,
    detrend=None,
    verbose=False
)

print("Epochs created:", len(epochs_no_reject))
print("Epoch shape:", epochs_no_reject.get_data().shape)

print()

# ------------------------------------------------------------------
# Drop log
# ------------------------------------------------------------------

print("=" * 80)
print("DROP LOG - NO REJECTION")
print("=" * 80)

drop_counts = {}

for log in epochs_no_reject.drop_log:

    if len(log) == 0:
        key = "KEPT"
    else:
        key = ";".join(log)

    drop_counts[key] = drop_counts.get(key, 0) + 1

for key, count in sorted(
    drop_counts.items(),
    key=lambda x: (-x[1], x[0])
):
    print(f"{key:40s} : {count}")

print()

# ------------------------------------------------------------------
# TEST 2: Epoching with annotation rejection disabled
# ------------------------------------------------------------------

print("=" * 80)
print("TEST 2: CHECK EDGE DROPS")
print("=" * 80)

if len(selected_events) > 0:

    pre_samples = int(abs(tmin) * raw.info["sfreq"])
    post_samples = int(tmax * raw.info["sfreq"])

    print("Samples before event:", pre_samples)
    print("Samples after event :", post_samples)

    valid_count = 0
    edge_count = 0

    for event in selected_events:

        sample = int(event[0])

        if (
            sample - pre_samples >= 0
            and
            sample + post_samples < raw.n_times
        ):
            valid_count += 1
        else:
            edge_count += 1

    print("Events with enough data:", valid_count)
    print("Events too close to edge:", edge_count)

print()

# ------------------------------------------------------------------
# Save ONLY diagnostic text
# ------------------------------------------------------------------

OUT_DIR = os.path.join(
    BASE,
    "epochs",
    "logs"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_FILE = os.path.join(
    OUT_DIR,
    "debug_sub001_run1.txt"
)

with open(OUT_FILE, "w", encoding="utf-8") as f:

    f.write("SINGLE RUN EPOCH DEBUG\n")
    f.write("=" * 80 + "\n")
    f.write(f"File: {FIF_FILE}\n")
    f.write(f"Channels: {len(raw.ch_names)}\n")
    f.write(f"Sampling rate: {raw.info['sfreq']}\n")
    f.write(f"Samples: {raw.n_times}\n")
    f.write(f"Duration: {raw.times[-1]}\n")
    f.write(f"Annotations: {len(raw.annotations)}\n")
    f.write(f"Events: {len(events)}\n")
    f.write(f"Selected events: {len(selected_events)}\n")
    f.write(f"Epochs without rejection: {len(epochs_no_reject)}\n")
    f.write(f"Epoch shape: {epochs_no_reject.get_data().shape}\n")
    f.write("\nEVENT IDS\n")
    for name, code in event_id.items():
        f.write(f"{name} -> {code}\n")

    f.write("\nDROP LOG\n")
    for key, count in drop_counts.items():
        f.write(f"{key}: {count}\n")

print("=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)

print("Epochs without rejection:", len(epochs_no_reject))
print()

if len(epochs_no_reject) > 0:
    print("GOOD:")
    print("Epoching itself works.")
    print("The previous ZERO_EPOCH problem is most likely caused by")
    print("the rejection criteria / annotation rejection / event selection.")
else:
    print("PROBLEM:")
    print("Even without amplitude rejection, zero epochs remain.")
    print("The next issue is event timing, event selection, or annotations.")

print()
print("Diagnostic file saved:")
print(OUT_FILE)

print()
print("RAW DATA WAS NOT MODIFIED.")
print("PREPROCESSED FIF WAS NOT MODIFIED.")
print("NO EPOCH FILE WAS MODIFIED.")