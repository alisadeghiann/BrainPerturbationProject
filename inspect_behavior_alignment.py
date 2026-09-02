from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

FEATURES = BASE / "features" / "basic" / "eeg_feature_matrix.csv"
EVENTS = BASE / "final_dataset" / "perturbation" / "logs" / "final_dataset_build_summary.csv"

OUT = BASE / "features" / "behavior_aligned"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURES)

print("=" * 80)
print("BEHAVIOR ALIGNMENT CHECK")
print("=" * 80)

print("Feature matrix loaded:")
print(f"Rows: {len(df):,}")

print("\nColumns:")
print(df.columns[:10].tolist())

print("\nChecking available metadata...")

required = ["file", "subject", "run", "epoch"]

missing = [c for c in required if c not in df.columns]

if missing:
    print("MISSING REQUIRED COLUMNS:", missing)
    raise SystemExit(1)

print("All structural metadata present.")

# IMPORTANT:
# We do NOT guess behavioral labels.
# First inspect metadata stored inside the final epochs.

import mne

sample_file = BASE / "final_dataset" / "perturbation" / "epochs" / df.iloc[0]["file"]

epochs = mne.read_epochs(sample_file, preload=False, verbose=False)

print("\nSample epoch metadata:")

if epochs.metadata is None:
    print("NO MNE METADATA FOUND")
else:
    print(epochs.metadata.head())
    print("\nMetadata columns:")
    print(epochs.metadata.columns.tolist())

print("\nEvent IDs:")
print(epochs.event_id)

print("=" * 80)
print("BEHAVIOR ALIGNMENT INSPECTION COMPLETE")
print("=" * 80)
