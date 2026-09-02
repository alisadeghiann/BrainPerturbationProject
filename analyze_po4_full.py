from pathlib import Path
import numpy as np
import h5py


# ============================================================
# FILES
# ============================================================

SET_FILE = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-018\ses-01\eeg\sub-018_ses-01_task-WorkingMemory_run-1_eeg.set"
)

FDT_FILE = SET_FILE.with_suffix(".fdt")


# ============================================================
# READ BASIC INFO
# ============================================================

with h5py.File(SET_FILE, "r") as h5:

    nbchan = int(
        np.asarray(h5["nbchan"][()]).squeeze()
    )

    pnts = int(
        np.asarray(h5["pnts"][()]).squeeze()
    )

    srate = float(
        np.asarray(h5["srate"][()]).squeeze()
    )

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


# ============================================================
# LOAD DATA
# ============================================================

data = np.fromfile(
    FDT_FILE,
    dtype="<f4"
).reshape(
    (nbchan, pnts),
    order="F"
)


# ============================================================
# FIND PO4 AND NEIGHBORS
# ============================================================

targets = [
    "PO3",
    "POZ",
    "PO4",
    "PO8",
    "O1",
    "OZ",
    "O2"
]

indices = {}

for target in targets:

    for i, name in enumerate(channel_names):

        if name.strip().upper() == target:

            indices[target] = i
            break


po4 = data[
    indices["PO4"]
].astype(np.float64)


neighbors = np.vstack([
    data[indices["PO3"]],
    data[indices["POZ"]],
    data[indices["PO8"]]
]).astype(np.float64)


# ============================================================
# FULL RECORDING
# ============================================================

duration = pnts / srate

print("=" * 70)
print("FULL PO4 ARTIFACT ANALYSIS")
print("=" * 70)

print(f"\nDuration: {duration:.2f} seconds")
print(f"Sampling rate: {srate} Hz")

print("\nPO4 FULL RECORDING")
print("-" * 70)

print(
    f"Mean: {np.mean(po4):.3f}"
)

print(
    f"STD: {np.std(po4):.3f}"
)

print(
    f"Min: {np.min(po4):.3f}"
)

print(
    f"Max: {np.max(po4):.3f}"
)

print(
    f"Peak-to-peak: {np.ptp(po4):.3f}"
)


# ============================================================
# WINDOW ANALYSIS
# ============================================================

window_seconds = 10

window_samples = int(
    window_seconds * srate
)

n_windows = pnts // window_samples

window_results = []


print("\n" + "=" * 70)
print("10-SECOND WINDOW ANALYSIS")
print("=" * 70)

for w in range(n_windows):

    start = w * window_samples
    end = start + window_samples

    segment = po4[start:end]

    neighbor_segment = neighbors[
        :,
        start:end
    ]

    po4_std = np.std(segment)

    neighbor_std = np.median(
        np.std(
            neighbor_segment,
            axis=1
        )
    )

    ratio = (
        po4_std /
        neighbor_std
        if neighbor_std > 0
        else np.inf
    )

    window_results.append(
        (
            w,
            start / srate,
            end / srate,
            po4_std,
            neighbor_std,
            ratio
        )
    )


# ============================================================
# SHOW WORST WINDOWS
# ============================================================

window_results.sort(
    key=lambda x: x[5],
    reverse=True
)

print(
    "\nTOP 20 WORST WINDOWS:"
)

print(
    "\nWindow | Start | End | PO4 STD | Neighbor STD | Ratio"
)

print("-" * 70)

for row in window_results[:20]:

    w, start, end, ps, ns, ratio = row

    print(
        f"{w:6d} | "
        f"{start:7.1f} | "
        f"{end:7.1f} | "
        f"{ps:9.2f} | "
        f"{ns:12.2f} | "
        f"{ratio:7.2f}"
    )


# ============================================================
# EXTREME AMPLITUDE
# ============================================================

print("\n" + "=" * 70)
print("EXTREME AMPLITUDE CHECK")
print("=" * 70)

thresholds = [
    100,
    200,
    300,
    500,
    750,
    1000
]

for threshold in thresholds:

    count = np.sum(
        np.abs(po4) > threshold
    )

    percentage = (
        count /
        len(po4)
        *
        100
    )

    print(
        f"|PO4| > {threshold:4d}: "
        f"{count:8d} samples "
        f"({percentage:.4f}%)"
    )


# ============================================================
# NEIGHBOR COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("NEIGHBOR COMPARISON")
print("=" * 70)

for name in [
    "PO3",
    "POZ",
    "PO4",
    "PO8",
    "O1",
    "OZ",
    "O2"
]:

    signal = data[
        indices[name]
    ]

    print(
        f"{name:5s} | "
        f"STD={np.std(signal):9.3f} | "
        f"MIN={np.min(signal):9.3f} | "
        f"MAX={np.max(signal):9.3f}"
    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

ratios = np.array([
    x[5]
    for x in window_results
])

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"\nMedian 10-sec PO4/neighbor ratio: "
    f"{np.median(ratios):.2f}"
)

print(
    f"Maximum 10-sec ratio: "
    f"{np.max(ratios):.2f}"
)

print(
    f"Windows with ratio > 5: "
    f"{np.sum(ratios > 5)} / {len(ratios)}"
)

print(
    f"Windows with ratio > 10: "
    f"{np.sum(ratios > 10)} / {len(ratios)}"
)

print(
    f"Windows with ratio > 20: "
    f"{np.sum(ratios > 20)} / {len(ratios)}"
)

print("\nIMPORTANT:")
print(
    "Do NOT remove PO4 yet."
)

print(
    "Send me the complete terminal output."
)

print("=" * 70)