import os
import csv
import numpy as np
import h5py

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-004\ses-01\eeg"

OUTPUT_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject\qc\sub004_temporal"

RUNS = [1, 2, 3, 4]

EEG_CHANNELS = 69
DTYPE = np.float32

WINDOW_SEC = 10

# Thresholds
HIGH_AMPLITUDE_THRESHOLD = 200
CLIP_LOW = -499
CLIP_HIGH = 499

# Global artifact thresholds
GLOBAL_HIGH_AMPLITUDE_PERCENT = 2.0
GLOBAL_CLIP_PERCENT = 0.5

# ============================================================
# SET METADATA
# ============================================================

def load_set_metadata(set_file):

    with h5py.File(set_file, "r") as f:

        nbchan = int(np.array(f["nbchan"]).squeeze())
        pnts = int(np.array(f["pnts"]).squeeze())
        srate = float(np.array(f["srate"]).squeeze())
        trials = int(np.array(f["trials"]).squeeze())

    return nbchan, pnts, srate, trials


# ============================================================
# MAIN
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_results = []

print("=" * 80)
print("SUB-004 TEMPORAL / GLOBAL EEG QC")
print("=" * 80)

for run in RUNS:

    print()
    print("=" * 80)
    print(f"RUN {run}")
    print("=" * 80)

    base = f"sub-004_ses-01_task-WorkingMemory_run-{run}_eeg"

    set_file = os.path.join(DATA_DIR, base + ".set")
    fdt_file = os.path.join(DATA_DIR, base + ".fdt")

    if not os.path.exists(set_file):
        print("SET NOT FOUND:", set_file)
        continue

    if not os.path.exists(fdt_file):
        print("FDT NOT FOUND:", fdt_file)
        continue

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    nbchan, pnts, srate, trials = load_set_metadata(set_file)

    duration = (pnts * trials) / srate

    print("Channels:", nbchan)
    print("EEG channels:", EEG_CHANNELS)
    print("Sampling rate:", srate)
    print("Points:", pnts)
    print("Duration:", round(duration, 3), "sec")

    # --------------------------------------------------------
    # Memory map REAL FDT
    # --------------------------------------------------------

    raw = np.memmap(
        fdt_file,
        dtype=DTYPE,
        mode="r",
        shape=(nbchan, pnts * trials),
        order="F"
    )

    # Only EEG channels
    eeg = raw[:EEG_CHANNELS]

    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    samples_per_window = int(WINDOW_SEC * srate)

    n_windows = int(
        np.ceil(eeg.shape[1] / samples_per_window)
    )

    print("Window:", WINDOW_SEC, "sec")
    print("Number of windows:", n_windows)

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print(
        f"{'Window':>7} "
        f"{'Start':>9} "
        f"{'End':>9} "
        f"{'MedianSTD':>12} "
        f"{'MeanSTD':>12} "
        f"{'>200%':>9} "
        f"{'Clip%':>9} "
        f"{'BadCh':>7} "
        f"{'Status':<15}"
    )

    print("-" * 105)

    # --------------------------------------------------------
    # Temporal QC
    # --------------------------------------------------------

    for w in range(n_windows):

        start_sample = w * samples_per_window
        end_sample = min(
            (w + 1) * samples_per_window,
            eeg.shape[1]
        )

        start_sec = start_sample / srate
        end_sec = end_sample / srate

        window = np.asarray(
            eeg[:, start_sample:end_sample],
            dtype=np.float64
        )

        # ----------------------------------------------------
        # Channel STD
        # ----------------------------------------------------

        channel_std = np.std(
            window,
            axis=1
        )

        median_std = np.median(channel_std)
        mean_std = np.mean(channel_std)

        # ----------------------------------------------------
        # High amplitude
        # ----------------------------------------------------

        high_amp_mask = (
            np.abs(window) >
            HIGH_AMPLITUDE_THRESHOLD
        )

        high_amp_percent = (
            np.mean(high_amp_mask) * 100
        )

        # ----------------------------------------------------
        # Clipping
        # ----------------------------------------------------

        clip_mask = (
            (window <= CLIP_LOW) |
            (window >= CLIP_HIGH)
        )

        clip_percent = (
            np.mean(clip_mask) * 100
        )

        # ----------------------------------------------------
        # Number of channels affected
        # ----------------------------------------------------

        channel_high_amp = (
            np.mean(high_amp_mask, axis=1) * 100
        )

        channel_clip = (
            np.mean(clip_mask, axis=1) * 100
        )

        bad_channels = np.sum(
            (channel_high_amp > 2) |
            (channel_clip > 0.5)
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if (
            high_amp_percent >= GLOBAL_HIGH_AMPLITUDE_PERCENT
            or
            clip_percent >= GLOBAL_CLIP_PERCENT
        ):

            status = "GLOBAL_ARTIFACT"

        elif (
            high_amp_percent >= 0.5
            or
            clip_percent >= 0.1
        ):

            status = "REVIEW"

        else:

            status = "NORMAL"

        print(
            f"{w+1:7d} "
            f"{start_sec:9.1f} "
            f"{end_sec:9.1f} "
            f"{median_std:12.3f} "
            f"{mean_std:12.3f} "
            f"{high_amp_percent:9.3f} "
            f"{clip_percent:9.3f} "
            f"{bad_channels:7d} "
            f"{status:<15}"
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        all_results.append({
            "subject": "sub-004",
            "run": run,
            "window": w + 1,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "median_std": median_std,
            "mean_std": mean_std,
            "high_amplitude_percent": high_amp_percent,
            "clip_percent": clip_percent,
            "bad_channel_count": int(bad_channels),
            "status": status
        })

    del raw

# ============================================================
# SAVE CSV
# ============================================================

csv_file = os.path.join(
    OUTPUT_DIR,
    "sub004_temporal_qc.csv"
)

fieldnames = [
    "subject",
    "run",
    "window",
    "start_sec",
    "end_sec",
    "median_std",
    "mean_std",
    "high_amplitude_percent",
    "clip_percent",
    "bad_channel_count",
    "status"
]

with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_results)

# ============================================================
# SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "sub004_temporal_qc_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 80 + "\n")
    f.write("SUB-004 TEMPORAL QC SUMMARY\n")
    f.write("=" * 80 + "\n\n")

    for run in RUNS:

        rows = [
            r for r in all_results
            if r["run"] == run
        ]

        if not rows:
            continue

        normal = sum(
            r["status"] == "NORMAL"
            for r in rows
        )

        review = sum(
            r["status"] == "REVIEW"
            for r in rows
        )

        global_artifact = sum(
            r["status"] == "GLOBAL_ARTIFACT"
            for r in rows
        )

        f.write(f"RUN {run}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Normal windows:          {normal}\n")
        f.write(f"Review windows:          {review}\n")
        f.write(f"Global artifact windows: {global_artifact}\n\n")

        suspicious = [
            r for r in rows
            if r["status"] != "NORMAL"
        ]

        if suspicious:

            f.write("Suspicious windows:\n")

            for r in suspicious:

                f.write(
                    f"  {r['start_sec']:.1f} - "
                    f"{r['end_sec']:.1f} sec | "
                    f"{r['status']} | "
                    f">200={r['high_amplitude_percent']:.3f}% | "
                    f"clip={r['clip_percent']:.3f}% | "
                    f"bad_ch={r['bad_channel_count']}\n"
                )

        else:

            f.write(
                "No suspicious temporal windows detected.\n"
            )

        f.write("\n")

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("TEMPORAL QC COMPLETE")
print("=" * 80)

print()
print("Total windows:", len(all_results))

print()
print("STATUS COUNTS")

from collections import Counter

counts = Counter(
    r["status"]
    for r in all_results
)

for status, count in counts.items():

    print(
        f"{status:<20} {count}"
    )

print()
print("Saved:")
print(csv_file)
print(summary_file)

print()
print("=" * 80)
print("IMPORTANT")
print("=" * 80)
print("This script ONLY reads the FDT files.")
print("RAW DATA WAS NOT MODIFIED.")