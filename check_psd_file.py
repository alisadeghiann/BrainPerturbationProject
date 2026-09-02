import numpy as np
from pathlib import Path

FILE = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\processed\sub-009_ses-01_task-WorkingMemory_run-1_psd.npz"
)

print("=" * 60)
print("CHECKING PSD FILE")
print("=" * 60)

data = np.load(FILE)

print("\nKeys inside PSD file:")

for key in data.files:
    print(
        f"{key:25s}",
        "shape =", data[key].shape,
        "dtype =", data[key].dtype
    )

print("\nDONE")