from pathlib import Path
import mne
import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")
INPUT = BASE / "final_dataset" / "perturbation" / "epochs"

files = sorted(INPUT.glob("*_final_epo.fif"))

print("=" * 90)
print("FINAL DATASET VALIDATION")
print("=" * 90)

print(f"Final files: {len(files)}")

results = []

for i, f in enumerate(files, 1):
    print(f"\n[{i}/{len(files)}] {f.name}")

    try:
        epochs = mne.read_epochs(f, preload=True, verbose=False)

        data = epochs.get_data()

        row = {
            "file": f.name,
            "epochs": len(epochs),
            "channels": len(epochs.ch_names),
            "sfreq": epochs.info["sfreq"],
            "tmin": epochs.tmin,
            "tmax": epochs.tmax,
            "nan": int(np.isnan(data).sum()),
            "inf": int(np.isinf(data).sum()),
            "event_types": len(epochs.event_id),
        }

        print(f"Epochs:     {row['epochs']}")
        print(f"Channels:   {row['channels']}")
        print(f"SFREQ:      {row['sfreq']}")
        print(f"Time:       {row['tmin']} ? {row['tmax']}")
        print(f"NaN:        {row['nan']}")
        print(f"Inf:        {row['inf']}")
        print(f"Event IDs:  {row['event_types']}")

        results.append(row)

    except Exception as e:
        print(f"ERROR: {e}")

        results.append({
            "file": f.name,
            "error": str(e)
        })

df = pd.DataFrame(results)

OUTPUT = BASE / "final_dataset" / "perturbation" / "logs"
OUTPUT.mkdir(parents=True, exist_ok=True)

csv_path = OUTPUT / "final_dataset_validation.csv"
df.to_csv(csv_path, index=False)

print("\n" + "=" * 90)
print("VALIDATION COMPLETE")
print("=" * 90)

print(f"Files found:       {len(files)}")
print(f"Successfully read: {len(df[df.get('error').isna()]) if 'error' in df.columns else len(df)}")

if "nan" in df.columns:
    print(f"Total NaN:         {df['nan'].sum()}")

if "inf" in df.columns:
    print(f"Total Inf:         {df['inf'].sum()}")

print(f"\nSaved:")
print(csv_path)

print("=" * 90)
