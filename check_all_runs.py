import os

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"
PROCESSED = os.path.join(BASE, "processed")

print("=" * 70)
print("CHECKING ALL EEG RUNS")
print("=" * 70)

for run in range(1, 5):

    prefix = f"sub-009_ses-01_task-WorkingMemory_run-{run}"

    files = {
        "Raw FIF": f"{prefix}_raw.fif",
        "Filtered": f"{prefix}_filtered_raw.fif",
        "Events": f"{prefix}_events-eve.fif",
        "Epochs": f"{prefix}_epochs.fif",
        "Clean Epochs": f"{prefix}_clean-epo.fif",
        "ICA": f"{prefix}_ica.fif",
        "PSD": f"{prefix}_psd.npz",
        "Spatial": f"{prefix}_spatial.npz",
    }

    print("\n" + "=" * 70)
    print(f"RUN {run}")
    print("=" * 70)

    for name, filename in files.items():

        path = os.path.join(PROCESSED, filename)

        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[YES] {name:15s} | {size_mb:.2f} MB")
        else:
            print(f"[NO ] {name:15s}")

print("\n" + "=" * 70)
print("CHECK COMPLETED")
print("=" * 70)