from pathlib import Path
import mne
import numpy as np

BASE_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

DATA_DIR = (
    BASE_DIR
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v4"
    / "ELIGIBLE"
)

files = sorted(
    DATA_DIR.glob(
        "sub-024_run-*_standardized_epo.fif"
    )
)

print("=" * 80)
print("SUB-024 EVENT MAPPING INSPECTION")
print("=" * 80)

for filepath in files:

    print()
    print("=" * 80)
    print(filepath.name)
    print("=" * 80)

    epochs = mne.read_epochs(
        filepath,
        preload=False,
        verbose=False
    )

    print()
    print("EVENT_ID:")
    print(epochs.event_id)

    print()
    print("EVENTS:")

    ids, counts = np.unique(
        epochs.events[:, 2],
        return_counts=True
    )

    inverse = {
        int(v): k
        for k, v in epochs.event_id.items()
    }

    for eid, count in zip(ids, counts):

        print(
            f"ID {eid:>2} | "
            f"{inverse.get(int(eid), 'UNKNOWN'):<25} | "
            f"{count}"
        )

    print()
    print("EPOCH METADATA:")

    if epochs.metadata is not None:

        print(
            epochs.metadata.head()
        )

        print(
            "Metadata columns:",
            list(epochs.metadata.columns)
        )

    else:

        print("NO METADATA")

print()
print("=" * 80)
print("DONE")
print("=" * 80)