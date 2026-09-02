import mne
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

EPOCH_FILE = PROJECT_DIR / (
    r"processed\sub-009_ses-01_task-WorkingMemory_run-2_epochs-epo.fif"
)

ICA_FILE = PROJECT_DIR / (
    r"processed\sub-009_ses-01_task-WorkingMemory_run-2_ica.fif"
)

print("=" * 70)
print("LOADING RUN 2 EPOCHS")
print("=" * 70)

epochs = mne.read_epochs(
    EPOCH_FILE,
    preload=True
)

print(epochs)
print("Epochs:", len(epochs))
print("Channels:", len(epochs.ch_names))
print("Sampling rate:", epochs.info["sfreq"])

print()
print("=" * 70)
print("CHANNEL TYPES")
print("=" * 70)

eeg_channels = mne.pick_types(
    epochs.info,
    eeg=True,
    exclude=[]
)

eog_channels = mne.pick_types(
    epochs.info,
    eog=True,
    exclude=[]
)

print("EEG channels:", len(eeg_channels))
print("EOG channels:", len(eog_channels))

print("EOG names:")

for idx in eog_channels:
    print(" ", epochs.ch_names[idx])


print()
print("=" * 70)
print("CREATING ICA")
print("=" * 70)

ica = mne.preprocessing.ICA(
    n_components=0.99,
    method="fastica",
    random_state=42,
    max_iter="auto"
)

print("Fitting ICA...")

ica.fit(
    epochs,
    picks=eeg_channels
)

print()
print("ICA fitting completed.")
print("Number of ICA components:", ica.n_components_)


print()
print("=" * 70)
print("SEARCHING FOR EOG COMPONENTS")
print("=" * 70)

eog_components = []

for eog_name in ["LEYE", "REYE"]:

    if eog_name in epochs.ch_names:

        print()
        print("Searching using:", eog_name)

        eog_inds, scores = ica.find_bads_eog(
            epochs,
            ch_name=eog_name
        )

        print("Detected components:", eog_inds)

        for component, score in zip(eog_inds, scores[eog_inds]):
            print(
                f"Component {component}: "
                f"EOG score = {score:.4f}"
            )

        eog_components.extend(eog_inds)


eog_components = sorted(
    set(eog_components)
)

print()
print("=" * 70)
print("FINAL EOG COMPONENT SUMMARY")
print("=" * 70)

print(
    "Potential EOG components:",
    eog_components
)


print()
print("=" * 70)
print("SAVING ICA")
print("=" * 70)

ica.save(
    ICA_FILE,
    overwrite=True
)

print()
print("[DONE]")
print("ICA saved successfully:")
print(ICA_FILE)


print()
print("=" * 70)
print("OPENING ICA COMPONENT VIEWER")
print("=" * 70)

print()
print("A graphical window should open.")
print("Inspect the ICA components visually.")
print("DO NOT remove components yet.")

ica.plot_components(
    show=True
)

for component in eog_components:

    print()
    print("Opening component:", component)

    ica.plot_properties(
        epochs,
        picks=component,
        show=True
    )

print()
print("=" * 70)
print("ICA ANALYSIS COMPLETED")
print("=" * 70)

print("No components were removed.")
print("No EEG data was modified.")