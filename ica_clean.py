import os
import numpy as np
import mne
from mne.preprocessing import ICA

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

SUBJECT = "sub-009"
RUN = "run-1"

EPOCH_FILE = os.path.join(
    PROCESSED_DIR,
    f"{SUBJECT}_ses-01_task-WorkingMemory_{RUN}_epochs.fif"
)

ICA_FILE = os.path.join(
    PROCESSED_DIR,
    f"{SUBJECT}_ses-01_task-WorkingMemory_{RUN}_ica.fif"
)

# ============================================================
# LOAD EPOCHS
# ============================================================

print("=" * 70)
print("LOADING EPOCHS")
print("=" * 70)

epochs = mne.read_epochs(
    EPOCH_FILE,
    preload=True
)

print("Epochs:", epochs)
print("Number of epochs:", len(epochs))
print("Number of channels:", len(epochs.ch_names))
print("Sampling rate:", epochs.info["sfreq"])

# ============================================================
# LOAD SAVED ICA
# ============================================================

print()
print("=" * 70)
print("LOADING ICA")
print("=" * 70)

ica = mne.preprocessing.read_ica(ICA_FILE)

print("ICA components:", ica.n_components_)

# ============================================================
# EOG INFORMATION
# ============================================================

print()
print("=" * 70)
print("EOG INFORMATION")
print("=" * 70)

eog_names = ["LEYE", "REYE"]

for name in eog_names:
    if name in epochs.ch_names:
        print("Found EOG:", name)
    else:
        print("WARNING: EOG channel not found:", name)

# ============================================================
# FIND EOG COMPONENTS AGAIN
# ============================================================

print()
print("=" * 70)
print("EOG COMPONENT ANALYSIS")
print("=" * 70)

all_eog_components = set()

for eog_name in eog_names:

    if eog_name not in epochs.ch_names:
        continue

    print()
    print("Analyzing:", eog_name)

    try:

        eog_inds, scores = ica.find_bads_eog(
            epochs,
            ch_name=eog_name
        )

        print("Detected components:", eog_inds)

        if len(eog_inds) > 0:

            for component in eog_inds:

                score = scores[component]

                print(
                    f"Component {component}: "
                    f"EOG score = {score:.4f}"
                )

                all_eog_components.add(int(component))

    except Exception as e:

        print("Error analyzing", eog_name)
        print(e)

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL EOG COMPONENT SUMMARY")
print("=" * 70)

if len(all_eog_components) == 0:

    print("No EOG-related components detected.")

else:

    print(
        "Potential EOG components:",
        sorted(all_eog_components)
    )

# ============================================================
# SHOW SELECTED COMPONENTS
# ============================================================

print()
print("=" * 70)
print("OPENING COMPONENT DETAILS")
print("=" * 70)

for component in sorted(all_eog_components):

    print()
    print("Opening component:", component)

    try:

        ica.plot_properties(
            epochs,
            picks=[component],
            psd_args={"fmax": 40},
            show=True
        )

    except Exception as e:

        print(
            "Could not plot properties for component",
            component
        )
        print(e)

print()
print("=" * 70)
print("ANALYSIS FINISHED")
print("=" * 70)

print()
print("IMPORTANT:")
print("No ICA components were removed.")
print("No EEG data was modified.")
print("This step was diagnostic only.")