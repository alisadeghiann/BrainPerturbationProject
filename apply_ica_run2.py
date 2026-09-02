import mne
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")
PROCESSED = PROJECT_DIR / "processed"

EPOCHS_FILE = PROCESSED / "sub-009_ses-01_task-WorkingMemory_run-2_epochs-epo.fif"
ICA_FILE = PROCESSED / "sub-009_ses-01_task-WorkingMemory_run-2_ica.fif"
OUTPUT_FILE = PROCESSED / "sub-009_ses-01_task-WorkingMemory_run-2_clean-epo.fif"

print("=" * 60)
print("LOADING RUN 2 EPOCHS")
print("=" * 60)

epochs = mne.read_epochs(EPOCHS_FILE, preload=True)

print("Epochs:", len(epochs))
print("Channels:", len(epochs.ch_names))

print("\n" + "=" * 60)
print("LOADING ICA")
print("=" * 60)

ica = mne.preprocessing.read_ica(ICA_FILE)

print("ICA components:", ica.n_components_)

print("\n" + "=" * 60)
print("REMOVING ICA COMPONENT")
print("=" * 60)

ica.exclude = [0]

print("Excluded components:", ica.exclude)

print("\nApplying ICA...")

ica.apply(epochs)

print("ICA cleaning completed.")

print("\n" + "=" * 60)
print("RE-APPLYING BASELINE")
print("=" * 60)

epochs.apply_baseline((-0.5, 0.0))

print("Baseline correction completed.")

print("\n" + "=" * 60)
print("QUALITY CHECK")
print("=" * 60)

data = epochs.get_data()

print("Shape:", data.shape)
print("NaN:", np.isnan(data).sum())
print("Inf:", np.isinf(data).sum())
print("Minimum:", data.min())
print("Maximum:", data.max())
print("Mean:", data.mean())
print("STD:", data.std())

print("\n" + "=" * 60)
print("EPOCH COUNTS")
print("=" * 60)

for event_name, event_code in epochs.event_id.items():
    count = sum(epochs.events[:, 2] == event_code)
    print(f"{event_name:25s}: {count}")

print("\n" + "=" * 60)
print("SAVING CLEAN EPOCHS")
print("=" * 60)

epochs.save(OUTPUT_FILE, overwrite=True)

print("\n[DONE]")
print("Clean Run 2 saved:")
print(OUTPUT_FILE)