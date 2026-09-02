import pandas as pd
import json
from pathlib import Path


# ============================================================
# PATH
# ============================================================

eeg_dir = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-002\ses-01\eeg"
)

run = "sub-002_ses-01_task-WorkingMemory_run-1"


channels_file = eeg_dir / f"{run}_channels.tsv"
electrodes_file = eeg_dir / f"{run}_electrodes.tsv"
events_file = eeg_dir / f"{run}_events.tsv"
eeg_json_file = eeg_dir / f"{run}_eeg.json"


# ============================================================
# 1. CHANNELS
# ============================================================

print("=" * 70)
print("CHANNEL INFORMATION")
print("=" * 70)

channels = pd.read_csv(
    channels_file,
    sep="\t"
)

print("\nColumns:")
print(channels.columns.tolist())

print("\nNumber of channels:", len(channels))

print("\nFirst rows:")
print(channels.head(15).to_string(index=False))


# ============================================================
# 2. ELECTRODES
# ============================================================

print("\n" + "=" * 70)
print("ELECTRODE INFORMATION")
print("=" * 70)

electrodes = pd.read_csv(
    electrodes_file,
    sep="\t"
)

print("\nColumns:")
print(electrodes.columns.tolist())

print("\nNumber of electrodes:", len(electrodes))

print("\nFirst rows:")
print(electrodes.head(15).to_string(index=False))


# ============================================================
# 3. EVENTS
# ============================================================

print("\n" + "=" * 70)
print("EVENT INFORMATION")
print("=" * 70)

events = pd.read_csv(
    events_file,
    sep="\t"
)

print("\nColumns:")
print(events.columns.tolist())

print("\nNumber of events:", len(events))

print("\nFirst 30 events:")
print(events.head(30).to_string(index=False))


# ============================================================
# 4. EVENT TYPES
# ============================================================

print("\n" + "=" * 70)
print("EVENT / TRIAL TYPES")
print("=" * 70)

for column in events.columns:

    print(f"\nColumn: {column}")

    try:
        print(events[column].value_counts().head(30))
    except:
        pass


# ============================================================
# 5. EEG JSON
# ============================================================

print("\n" + "=" * 70)
print("EEG JSON METADATA")
print("=" * 70)

with open(
    eeg_json_file,
    "r",
    encoding="utf-8"
) as f:

    eeg_info = json.load(f)

for key, value in eeg_info.items():
    print(f"{key}: {value}")


# ============================================================
# 6. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("Channels:", len(channels))
print("Electrodes:", len(electrodes))
print("Events:", len(events))

print("\nMetadata inspection completed.")