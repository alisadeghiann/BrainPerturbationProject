from pathlib import Path
import numpy as np
import h5py


SET_FILE = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-018\ses-01\eeg\sub-018_ses-01_task-WorkingMemory_run-1_eeg.set"
)

FDT_FILE = SET_FILE.with_suffix(".fdt")


# ============================================================
# READ SET
# ============================================================

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


# ============================================================
# LOAD FDT
# ============================================================

data = np.fromfile(
    FDT_FILE,
    dtype="<f4"
).reshape(
    (nbchan, pnts),
    order="F"
)


def idx(name):

    for i, ch in enumerate(channel_names):

        if ch.strip().upper() == name:

            return i

    raise ValueError(
        f"Channel not found: {name}"
    )


po4 = data[idx("PO4")].astype(np.float64)
po8 = data[idx("PO8")].astype(np.float64)

po3 = data[idx("PO3")].astype(np.float64)
poz = data[idx("POZ")].astype(np.float64)
o1 = data[idx("O1")].astype(np.float64)
oz = data[idx("OZ")].astype(np.float64)
o2 = data[idx("O2")].astype(np.float64)


# ============================================================
# CORRELATION
# ============================================================

print("=" * 70)
print("PO4 / PO8 RELATIONSHIP ANALYSIS")
print("=" * 70)

print(f"\nDuration: {pnts / srate:.2f} sec")


corr_po4_po8 = np.corrcoef(
    po4,
    po8
)[0, 1]

print(
    f"\nFull-recording PO4 ↔ PO8 correlation: "
    f"{corr_po4_po8:.4f}"
)


# ============================================================
# CLIPPING CO-OCCURRENCE
# ============================================================

po4_clip = np.abs(po4) >= 499
po8_clip = np.abs(po8) >= 499

print("\n" + "=" * 70)
print("CLIPPING CO-OCCURRENCE")
print("=" * 70)

print(
    f"\nPO4 clipped samples: "
    f"{po4_clip.sum()}"
)

print(
    f"PO8 clipped samples: "
    f"{po8_clip.sum()}"
)

both = po4_clip & po8_clip

print(
    f"Both clipped simultaneously: "
    f"{both.sum()}"
)

print(
    f"Percentage of PO4 clipping also involving PO8: "
    f"{both.sum() / max(po4_clip.sum(), 1) * 100:.3f}%"
)


# ============================================================
# PO4 HIGH / PO8 RESPONSE
# ============================================================

po4_high = np.abs(po4) > 200

print("\n" + "=" * 70)
print("PO8 DURING PO4 HIGH-AMPLITUDE SAMPLES")
print("=" * 70)

print(
    f"\nPO8 STD during normal PO4: "
    f"{np.std(po8[~po4_high]):.3f}"
)

print(
    f"PO8 STD during |PO4| > 200: "
    f"{np.std(po8[po4_high]):.3f}"
)

print(
    f"PO8 |x|>200 during normal PO4: "
    f"{np.mean(np.abs(po8[~po4_high]) > 200) * 100:.3f}%"
)

print(
    f"PO8 |x|>200 during |PO4|>200: "
    f"{np.mean(np.abs(po8[po4_high]) > 200) * 100:.3f}%"
)


# ============================================================
# 10 SECOND WINDOWS
# ============================================================

print("\n" + "=" * 70)
print("10-SECOND WINDOW CORRELATION")
print("=" * 70)

window = int(10 * srate)

results = []

for start in range(0, pnts - window, window):

    end = start + window

    x = po4[start:end]
    y = po8[start:end]

    r = np.corrcoef(x, y)[0, 1]

    results.append(
        (
            start / srate,
            end / srate,
            np.std(x),
            np.std(y),
            r,
            np.mean(np.abs(x) > 200) * 100,
            np.mean(np.abs(y) > 200) * 100
        )
    )


results.sort(
    key=lambda x: x[2],
    reverse=True
)


print(
    "\nStart | End | PO4 STD | PO8 STD | Corr | PO4>200 | PO8>200"
)

print("-" * 70)

for row in results[:20]:

    print(
        f"{row[0]:6.1f} | "
        f"{row[1]:6.1f} | "
        f"{row[2]:8.2f} | "
        f"{row[3]:8.2f} | "
        f"{row[4]:6.3f} | "
        f"{row[5]:8.2f}% | "
        f"{row[6]:8.2f}%"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FINAL")
print("=" * 70)

print("""
PO4 is currently considered a strong bad-channel candidate.

PO8 remains under REVIEW.

Do NOT remove either channel yet.
""")

print("=" * 70)