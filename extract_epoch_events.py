from pathlib import Path
import mne
import pandas as pd

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = BASE / "final_dataset" / "perturbation" / "epochs"
OUTPUT = BASE / "features" / "behavior_aligned"
OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT.glob("*_final_epo.fif"))

rows = []

print("=" * 80)
print("BEHAVIOR / EVENT STRUCTURE EXTRACTION")
print("=" * 80)

for i, f in enumerate(files, 1):

    print(f"[{i}/{len(files)}] {f.name}")

    epochs = mne.read_epochs(f, preload=False, verbose=False)

    events = epochs.events
    event_id = {v: k for k, v in epochs.event_id.items()}

    for epoch_idx, event in enumerate(events):

        code = int(event[2])
        event_name = event_id.get(code, "UNKNOWN")

        rows.append({
            "file": f.name,
            "subject": f.name.split("_")[0],
            "run": f.name.split("_")[1],
            "epoch": epoch_idx,
            "event_code": code,
            "event_name": event_name
        })

df = pd.DataFrame(rows)

output = OUTPUT / "epoch_event_map.csv"
df.to_csv(output, index=False)

print()
print("=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)

print(f"Rows: {len(df):,}")
print(f"Subjects: {df['subject'].nunique()}")
print(f"Runs: {df['file'].nunique()}")

print()
print("EVENT DISTRIBUTION:")
print(df["event_name"].value_counts())

print()
print("Saved:")
print(output)

print("=" * 80)
