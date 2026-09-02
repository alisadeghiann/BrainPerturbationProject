import os
import glob
import numpy as np
import pandas as pd
import mne

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"
INPUT = os.path.join(BASE, "epochs_clean")
OUTDIR = os.path.join(INPUT, "logs", "artifact_localization")

os.makedirs(OUTDIR, exist_ok=True)

CSV = os.path.join(OUTDIR, "artifact_localization_82runs.csv")
SUMMARY = os.path.join(OUTDIR, "artifact_localization_summary.txt")

files = sorted(glob.glob(os.path.join(INPUT, "*_clean_epo.fif")))

print("=" * 80)
print("ARTIFACT LOCALIZATION QC - 82 RUNS")
print("=" * 80)
print(f"Files found: {len(files)}")

records = []

for i, path in enumerate(files, 1):

    fname = os.path.basename(path)

    print("")
    print("=" * 80)
    print(f"[{i}/{len(files)}] {fname}")
    print("=" * 80)

    try:
        epochs = mne.read_epochs(path, preload=True, verbose=False)
        data = epochs.get_data(copy=False)

        n_epochs, n_channels, n_times = data.shape
        ch_names = epochs.ch_names

        # --------------------------------------------------
        # Per sample amplitude
        # --------------------------------------------------

        absdata = np.abs(data)

        # Fraction of samples > thresholds per epoch
        frac200 = np.mean(absdata > 200, axis=(1, 2))
        frac300 = np.mean(absdata > 300, axis=(1, 2))
        frac400 = np.mean(absdata > 400, axis=(1, 2))

        # Maximum amplitude per epoch
        epoch_max = np.max(absdata, axis=(1, 2))

        # Number of channels affected in each epoch
        channel_max = np.max(absdata, axis=2)

        channels_over200 = np.sum(channel_max > 200, axis=1)
        channels_over300 = np.sum(channel_max > 300, axis=1)
        channels_over400 = np.sum(channel_max > 400, axis=1)

        # --------------------------------------------------
        # Classify artifact localization
        # --------------------------------------------------

        for ep in range(n_epochs):

            f200 = frac200[ep]
            f300 = frac300[ep]
            f400 = frac400[ep]

            c200 = channels_over200[ep]
            c300 = channels_over300[ep]
            c400 = channels_over400[ep]

            mx = epoch_max[ep]

            if c400 >= max(10, int(n_channels * 0.50)):
                artifact_type = "GLOBAL_ARTIFACT"

            elif c300 >= max(5, int(n_channels * 0.25)):
                artifact_type = "WIDESPREAD_ARTIFACT"

            elif c200 >= max(3, int(n_channels * 0.10)):
                artifact_type = "MULTI_CHANNEL_ARTIFACT"

            elif c200 >= 1:
                artifact_type = "LOCAL_ARTIFACT"

            else:
                artifact_type = "CLEAN"

            # --------------------------------------------------
            # Identify worst channels
            # --------------------------------------------------

            ch_scores = channel_max[ep]

            worst_idx = np.argsort(ch_scores)[::-1][:5]

            worst_channels = []

            for idx in worst_idx:
                if ch_scores[idx] > 200:
                    worst_channels.append(
                        f"{ch_names[idx]}:{ch_scores[idx]:.1f}"
                    )

            worst_channels_text = ";".join(worst_channels)

            records.append({
                "file": fname,
                "epoch": ep + 1,
                "max_abs": float(mx),
                "percent_samples_gt200": float(f200 * 100),
                "percent_samples_gt300": float(f300 * 100),
                "percent_samples_gt400": float(f400 * 100),
                "channels_gt200": int(c200),
                "channels_gt300": int(c300),
                "channels_gt400": int(c400),
                "artifact_type": artifact_type,
                "worst_channels": worst_channels_text
            })

        # --------------------------------------------------
        # File summary
        # --------------------------------------------------

        subset = records[-n_epochs:]

        counts = pd.Series(
            [r["artifact_type"] for r in subset]
        ).value_counts()

        print(f"Epochs: {n_epochs}")
        print(f"CLEAN: {counts.get('CLEAN', 0)}")
        print(f"LOCAL: {counts.get('LOCAL_ARTIFACT', 0)}")
        print(f"MULTI: {counts.get('MULTI_CHANNEL_ARTIFACT', 0)}")
        print(f"WIDESPREAD: {counts.get('WIDESPREAD_ARTIFACT', 0)}")
        print(f"GLOBAL: {counts.get('GLOBAL_ARTIFACT', 0)}")

    except Exception as e:

        print(f"ERROR: {e}")

# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(records)

df.to_csv(
    CSV,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.append("=" * 80)
summary.append("ARTIFACT LOCALIZATION QC - 82 RUNS")
summary.append("=" * 80)
summary.append("")
summary.append(f"Files found: {len(files)}")
summary.append(f"Epoch records: {len(df)}")
summary.append("")

summary.append("=" * 80)
summary.append("ARTIFACT TYPE COUNTS")
summary.append("=" * 80)

summary.append(
    df["artifact_type"].value_counts().to_string()
)

summary.append("")

summary.append("=" * 80)
summary.append("FILES WITH GLOBAL / WIDESPREAD ARTIFACTS")
summary.append("=" * 80)

file_summary = (
    df.groupby(["file", "artifact_type"])
      .size()
      .unstack(fill_value=0)
)

cols = [
    "GLOBAL_ARTIFACT",
    "WIDESPREAD_ARTIFACT",
    "MULTI_CHANNEL_ARTIFACT",
    "LOCAL_ARTIFACT",
    "CLEAN"
]

for c in cols:
    if c not in file_summary.columns:
        file_summary[c] = 0

file_summary = file_summary[cols]

summary.append(
    file_summary.sort_values(
        ["GLOBAL_ARTIFACT", "WIDESPREAD_ARTIFACT"],
        ascending=False
    ).to_string()
)

summary.append("")

summary.append("=" * 80)
summary.append("MOST FREQUENT ARTIFACT CHANNELS")
summary.append("=" * 80)

channel_counts = {}

for text in df["worst_channels"].dropna():

    for item in str(text).split(";"):

        if not item:
            continue

        ch = item.split(":")[0]

        channel_counts[ch] = channel_counts.get(ch, 0) + 1

if channel_counts:
    channel_series = pd.Series(channel_counts).sort_values(
        ascending=False
    )
    summary.append(channel_series.to_string())
else:
    summary.append("NONE")

summary.append("")
summary.append("=" * 80)
summary.append("IMPORTANT")
summary.append("=" * 80)
summary.append("READ-ONLY QC.")
summary.append("NO RAW DATA WAS MODIFIED.")
summary.append("NO SET FILE WAS MODIFIED.")
summary.append("NO FDT FILE WAS MODIFIED.")
summary.append("NO EPOCH FILE WAS MODIFIED.")
summary.append("NO EPOCHS WERE DELETED.")

with open(SUMMARY, "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("")
print("=" * 80)
print("ARTIFACT LOCALIZATION COMPLETE")
print("=" * 80)

print("")
print("ARTIFACT TYPE COUNTS")
print(df["artifact_type"].value_counts())

print("")
print("Saved:")
print(CSV)
print(SUMMARY)

print("")
print("=" * 80)
print("NO DATA WAS MODIFIED.")
print("=" * 80)