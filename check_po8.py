from pathlib import Path
import numpy as np
import h5py

SET_FILE = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-018\ses-01\eeg\sub-018_ses-01_task-WorkingMemory_run-1_eeg.set"
)

FDT_FILE = SET_FILE.with_suffix(".fdt")


with h5py.File(SET_FILE, "r") as h5:

    nbchan = int(np.asarray(h5["nbchan"][()]).squeeze())
    pnts = int(np.asarray(h5["pnts"][()]).squeeze())
    srate = float(np.asarray(h5["srate"][()]).squeeze())

    labels = h5["chanlocs"]["labels"]

    channel_names = []

    for i in range(labels.shape[0]):

        arr = np.asarray(
            h5[labels[i, 0]][()]
        ).squeeze()

        if arr.dtype.kind in "iu":

            name = "".join(
                chr(int(x))
                for x in arr.flatten()
                if int(x) != 0
            )

        else:

            name = str(arr)

        channel_names.append(name)


data = np.fromfile(
    FDT_FILE,
    dtype="<f4"
).reshape(
    (nbchan, pnts),
    order="F"
)


def get_channel(name):

    for i, ch in enumerate(channel_names):

        if ch.strip().upper() == name:

            return i

    raise ValueError(f"Channel not found: {name}")


channels = [
    "PO3",
    "POZ",
    "PO4",
    "PO8",
    "O1",
    "OZ",
    "O2"
]


print("=" * 70)
print("PO8 / OCCIPITAL CHANNEL ARTIFACT CHECK")
print("=" * 70)

print(f"\nSampling rate: {srate}")
print(f"Duration: {pnts / srate:.2f} sec")


for name in channels:

    signal = data[get_channel(name)].astype(np.float64)

    print("\n" + "-" * 70)
    print(name)

    print(
        f"STD: {np.std(signal):.3f}"
    )

    print(
        f"Mean: {np.mean(signal):.3f}"
    )

    print(
        f"Min: {np.min(signal):.3f}"
    )

    print(
        f"Max: {np.max(signal):.3f}"
    )

    print(
        f"|x| > 100: "
        f"{np.sum(np.abs(signal) > 100)} "
        f"({np.mean(np.abs(signal) > 100)*100:.3f}%)"
    )

    print(
        f"|x| > 200: "
        f"{np.sum(np.abs(signal) > 200)} "
        f"({np.mean(np.abs(signal) > 200)*100:.3f}%)"
    )

    print(
        f"|x| > 300: "
        f"{np.sum(np.abs(signal) > 300)} "
        f"({np.mean(np.abs(signal) > 300)*100:.3f}%)"
    )

    print(
        f"<= -499: "
        f"{np.sum(signal <= -499)}"
    )

    print(
        f">= 499: "
        f"{np.sum(signal >= 499)}"
    )


print("\n" + "=" * 70)
print("10-SECOND PO8 ANALYSIS")
print("=" * 70)

po8 = data[get_channel("PO8")].astype(np.float64)

window_samples = int(10 * srate)

n_windows = pnts // window_samples

results = []

for w in range(n_windows):

    start = w * window_samples
    end = start + window_samples

    segment = po8[start:end]

    results.append(
        (
            w,
            start / srate,
            end / srate,
            np.std(segment),
            np.min(segment),
            np.max(segment),
            np.mean(np.abs(segment) > 200) * 100
        )
    )


results.sort(
    key=lambda x: x[3],
    reverse=True
)


print(
    "\nWindow | Start | End | STD | Min | Max | >200%"
)

print("-" * 70)

for row in results[:20]:

    print(
        f"{row[0]:6d} | "
        f"{row[1]:7.1f} | "
        f"{row[2]:7.1f} | "
        f"{row[3]:8.2f} | "
        f"{row[4]:8.2f} | "
        f"{row[5]:8.2f} | "
        f"{row[6]:6.2f}"
    )


print("\n" + "=" * 70)
print("FINAL")
print("=" * 70)

print("""
DO NOT REMOVE ANY CHANNEL YET.
Send the complete terminal output.
""")