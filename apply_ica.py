import os
import mne

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

EPOCHS_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "sub-009_ses-01_task-WorkingMemory_run-1_epochs.fif"
)

ICA_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "sub-009_ses-01_task-WorkingMemory_run-1_ica.fif"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "sub-009_ses-01_task-WorkingMemory_run-1_clean-epo.fif"
)

# ============================================================
# LOAD EPOCHS
# ============================================================

print("=" * 70)
print("LOADING EPOCHS")
print("=" * 70)

epochs = mne.read_epochs(
    EPOCHS_FILE,
    preload=True,
    verbose=True
)

print(epochs)

# ============================================================
# LOAD ICA
# ============================================================

print()
print("=" * 70)
print("LOADING ICA")
print("=" * 70)

ica = mne.preprocessing.read_ica(
    ICA_FILE,
    verbose=True
)

print(f"Number of ICA components: {ica.n_components_}")

# ============================================================
# ICA COMPONENTS SELECTED FOR REMOVAL
# ============================================================

print()
print("=" * 70)
print("ICA COMPONENTS TO REMOVE")
print("=" * 70)

components_to_remove = [1, 3]

print("Components:", components_to_remove)

# ============================================================
# APPLY ICA
# ============================================================

print()
print("=" * 70)
print("APPLYING ICA")
print("=" * 70)

ica.exclude = components_to_remove

print("ICA exclude list:")
print(ica.exclude)

epochs_clean = epochs.copy()

ica.apply(
    epochs_clean,
    verbose=True
)

print()
print("ICA cleaning completed.")

# ============================================================
# BASIC QUALITY CHECK
# ============================================================

print()
print("=" * 70)
print("QUALITY CHECK AFTER ICA")
print("=" * 70)

data = epochs_clean.get_data()

print("Clean epoch shape:", data.shape)
print("NaN count:", int(data.__class__.__name__ == "ndarray") and int(__import__("numpy").isnan(data).sum()))
print("Inf count:", int(__import__("numpy").isinf(data).sum()))

print("Minimum:", data.min())
print("Maximum:", data.max())
print("Mean:", data.mean())
print("STD:", data.std())

# ============================================================
# SAVE CLEAN EPOCHS
# ============================================================

print()
print("=" * 70)
print("SAVING CLEAN EPOCHS")
print("=" * 70)

epochs_clean.save(
    OUTPUT_FILE,
    overwrite=True,
    verbose=True
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print("Clean epochs saved successfully:")
print(OUTPUT_FILE)