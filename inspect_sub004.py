import os
import h5py
import numpy as np

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-004\ses-01\eeg"

files = sorted([
    f for f in os.listdir(BASE)
    if f.endswith(".set")
])

print("=" * 70)
print("SUB-004 RAW DATA INSPECTION")
print("=" * 70)

for fname in files:

    path = os.path.join(BASE, fname)

    print("\n" + "=" * 70)
    print(fname)
    print("=" * 70)

    with h5py.File(path, "r") as f:

        data = f["data"][:]

        print("Shape:", data.shape)
        print("Dtype:", data.dtype)
        print("Min:", np.min(data))
        print("Max:", np.max(data))
        print("Mean:", np.mean(data))
        print("STD:", np.std(data))

        print(
            "|x| > 100:",
            np.mean(np.abs(data) > 100) * 100,
            "%"
        )

        print(
            "|x| > 200:",
            np.mean(np.abs(data) > 200) * 100,
            "%"
        )

        print(
            "|x| > 300:",
            np.mean(np.abs(data) > 300) * 100,
            "%"
        )

        print(
            "<= -499:",
            np.sum(data <= -499)
        )

        print(
            ">= 499:",
            np.sum(data >= 499)
        )

        print(
            "Channels:",
            float(f["nbchan"][0][0])
        )

        print(
            "Sampling rate:",
            float(f["srate"][0][0])
        )

        print(
            "Duration:",
            float(f["xmax"][0][0] - f["xmin"][0][0]),
            "sec"
        )