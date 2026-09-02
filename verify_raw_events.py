import mne
from pathlib import Path


# ============================================================
# FILE
# ============================================================

raw_file = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
) / "sub-002_working_memory_combined_raw.fif"


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("LOADING COMBINED MNE RAW")
print("=" * 80)

raw = mne.io.read_raw_fif(
    raw_file,
    preload=False,
    verbose=True
)

print("\nRaw loaded successfully.")

print("\nChannels:", len(raw.ch_names))
print("Samples:", raw.n_times)
print("Sampling rate:", raw.info["sfreq"])
print("Duration:", raw.times[-1], "seconds")


# ============================================================
# CHANNEL TYPES
# ============================================================

print("\n" + "=" * 80)
print("CHANNEL TYPES")
print("=" * 80)

print(raw.get_channel_types())


# ============================================================
# ANNOTATIONS
# ============================================================

print("\n" + "=" * 80)
print("ANNOTATIONS")
print("=" * 80)

print("Total annotations:", len(raw.annotations))


# ============================================================
# COUNT IMPORTANT EVENT TYPES
# ============================================================

descriptions = raw.annotations.description

important_events = [
    "show_cross",
    "show_letter",
    "show_dash",
    "probe_target",
    "probe_not_shown",
    "right_click",
    "left_click",
    "sound_beep"
]

print("\nEvent counts:")

for event_name in important_events:

    count = sum(
        event_name in description
        for description in descriptions
    )

    print(
        f"{event_name:20s}: {count}"
    )


# ============================================================
# SHOW FIRST EVENTS
# ============================================================

print("\n" + "=" * 80)
print("FIRST 20 ANNOTATIONS")
print("=" * 80)

for onset, duration, description in zip(
    raw.annotations.onset[:20],
    raw.annotations.duration[:20],
    raw.annotations.description[:20]
):

    print(
        f"{onset:10.3f}s | "
        f"{duration:8.3f}s | "
        f"{description}"
    )


# ============================================================
# SHOW LAST EVENTS
# ============================================================

print("\n" + "=" * 80)
print("LAST 20 ANNOTATIONS")
print("=" * 80)

for onset, duration, description in zip(
    raw.annotations.onset[-20:],
    raw.annotations.duration[-20:],
    raw.annotations.description[-20:]
):

    print(
        f"{onset:10.3f}s | "
        f"{duration:8.3f}s | "
        f"{description}"
    )


# ============================================================
# CHECK MEMORY CONDITIONS
# ============================================================

print("\n" + "=" * 80)
print("MEMORY CONDITION COUNTS")
print("=" * 80)

for condition in ["cond-3", "cond-5", "cond-7"]:

    count = sum(
        condition in description
        for description in descriptions
    )

    print(
        f"{condition}: {count}"
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("EVENT VERIFICATION COMPLETED")
print("=" * 80)