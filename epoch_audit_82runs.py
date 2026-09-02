import os
import glob
import traceback
import numpy as np
import pandas as pd
import mne

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"
EPOCH_DIR = os.path.join(BASE, "epochs")
LOG_DIR = os.path.join(EPOCH_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(LOG_DIR, "epoch_audit_82runs.csv")
OUTPUT_SUMMARY = os.path.join(LOG_DIR, "epoch_audit_82runs_summary.txt")

files = sorted(
    glob.glob(os.path.join(EPOCH_DIR, "*_epo.fif"))
)

print("=" * 80)
print("EPOCH AUDIT - 82 PREPROCESSED RUNS")
print("=" * 80)

print(f"Epoch files found: {len(files)}")
print()

results = []

for i, fname in enumerate(files, 1):

    basename = os.path.basename(fname)

    print("=" * 80)
    print(f"[{i}/{len(files)}] {basename}")
    print("=" * 80)

    row = {
        "file": basename,
        "subject": "",
        "run": "",
        "n_epochs": np.nan,
        "n_channels": np.nan,
        "n_times": np.nan,
        "sfreq": np.nan,
        "tmin": np.nan,
        "tmax": np.nan,
        "duration_sec": np.nan,
        "event_count": np.nan,
        "zero_epoch": False,
        "all_data_finite": False,
        "nan_count": np.nan,
        "inf_count": np.nan,
        "min_value": np.nan,
        "max_value": np.nan,
        "mean_value": np.nan,
        "std_value": np.nan,
        "status": "FAILED",
        "reason": ""
    }

    try:

        # ------------------------------------------------------------
        # SUBJECT / RUN
        # ------------------------------------------------------------

        parts = basename.split("_")

        subject = next(
            (x for x in parts if x.startswith("sub-")),
            ""
        )

        run = ""

        for x in parts:
            if x.startswith("run-"):
                run = x.replace("run-", "")
                break

        row["subject"] = subject
        row["run"] = run

        # ------------------------------------------------------------
        # READ EPOCH FILE
        # ------------------------------------------------------------

        print("Reading epochs...")

        epochs = mne.read_epochs(
            fname,
            preload=True,
            verbose=False
        )

        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        n_times = len(epochs.times)

        row["n_epochs"] = n_epochs
        row["n_channels"] = n_channels
        row["n_times"] = n_times
        row["sfreq"] = epochs.info["sfreq"]
        row["tmin"] = epochs.tmin
        row["tmax"] = epochs.tmax
        row["duration_sec"] = (
            n_times / epochs.info["sfreq"]
            if n_times > 0
            else 0
        )

        # ------------------------------------------------------------
        # EVENTS
        # ------------------------------------------------------------

        try:
            row["event_count"] = len(epochs.events)
        except Exception:
            row["event_count"] = np.nan

        print(f"Epochs:      {n_epochs}")
        print(f"Channels:    {n_channels}")
        print(f"Time points: {n_times}")
        print(f"Sampling:    {epochs.info['sfreq']} Hz")

        # ------------------------------------------------------------
        # ZERO EPOCH CHECK
        # ------------------------------------------------------------

        if n_epochs == 0:

            row["zero_epoch"] = True
            row["status"] = "ZERO_EPOCH"
            row["reason"] = (
                "Epoch file exists but contains zero retained epochs"
            )

            print()
            print("WARNING: ZERO EPOCHS")
            print("This file requires review.")

            results.append(row)

            del epochs

            continue

        # ------------------------------------------------------------
        # DATA CHECK
        # ------------------------------------------------------------

        print("Loading epoch data...")

        data = epochs.get_data()

        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())

        row["nan_count"] = nan_count
        row["inf_count"] = inf_count

        row["all_data_finite"] = (
            nan_count == 0 and inf_count == 0
        )

        row["min_value"] = float(np.min(data))
        row["max_value"] = float(np.max(data))
        row["mean_value"] = float(np.mean(data))
        row["std_value"] = float(np.std(data))

        print(f"NaN:         {nan_count}")
        print(f"Inf:         {inf_count}")
        print(f"Min:         {row['min_value']:.6f}")
        print(f"Max:         {row['max_value']:.6f}")
        print(f"Mean:        {row['mean_value']:.6f}")
        print(f"STD:         {row['std_value']:.6f}")

        # ------------------------------------------------------------
        # FINAL STATUS
        # ------------------------------------------------------------

        if not row["all_data_finite"]:

            row["status"] = "DATA_REVIEW"
            row["reason"] = (
                "NaN or Inf values detected"
            )

        elif n_epochs < 10:

            row["status"] = "LOW_EPOCH_COUNT"
            row["reason"] = (
                f"Only {n_epochs} epochs retained"
            )

        else:

            row["status"] = "PASS"
            row["reason"] = "Epoch file passed basic integrity checks"

        print()
        print(f"STATUS: {row['status']}")

        results.append(row)

        del data
        del epochs

    except Exception as e:

        row["status"] = "FAILED"
        row["reason"] = str(e)

        print()
        print("ERROR:")
        print(str(e))

        results.append(row)

        traceback.print_exc()


# ====================================================================
# SAVE RESULTS
# ====================================================================

df = pd.DataFrame(results)

df = df.sort_values(
    by=["subject", "run"],
    na_position="last"
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ====================================================================
# SUMMARY
# ====================================================================

status_counts = (
    df["status"]
    .value_counts()
    .sort_index()
)

zero_epoch_files = df[
    df["status"] == "ZERO_EPOCH"
]

failed_files = df[
    df["status"] == "FAILED"
]

review_files = df[
    ~df["status"].isin(["PASS"])
]

summary_lines = []

summary_lines.append("=" * 80)
summary_lines.append("EPOCH AUDIT SUMMARY")
summary_lines.append("=" * 80)
summary_lines.append("")

summary_lines.append(
    f"Epoch files found: {len(df)}"
)

summary_lines.append("")

summary_lines.append("STATUS COUNTS")
summary_lines.append("-" * 80)

for status, count in status_counts.items():

    summary_lines.append(
        f"{status:<25} {count}"
    )

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("ZERO-EPOCH FILES")
summary_lines.append("=" * 80)

if len(zero_epoch_files) == 0:

    summary_lines.append("NONE")

else:

    for _, r in zero_epoch_files.iterrows():

        summary_lines.append(
            f"{r['subject']}  run-{r['run']}  {r['file']}"
        )

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("FILES REQUIRING REVIEW")
summary_lines.append("=" * 80)

if len(review_files) == 0:

    summary_lines.append("NONE")

else:

    for _, r in review_files.iterrows():

        summary_lines.append(
            f"{r['subject']}  run-{r['run']}  "
            f"{r['status']}  "
            f"epochs={r['n_epochs']}  "
            f"reason={r['reason']}"
        )

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("FAILED FILES")
summary_lines.append("=" * 80)

if len(failed_files) == 0:

    summary_lines.append("NONE")

else:

    for _, r in failed_files.iterrows():

        summary_lines.append(
            f"{r['subject']}  run-{r['run']}  "
            f"{r['file']}  "
            f"{r['reason']}"
        )

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("EPOCH COUNT DISTRIBUTION")
summary_lines.append("=" * 80)

if len(df) > 0:

    valid_epoch_counts = df[
        pd.notna(df["n_epochs"])
    ]["n_epochs"]

    if len(valid_epoch_counts) > 0:

        summary_lines.append(
            f"Minimum epochs: {int(valid_epoch_counts.min())}"
        )

        summary_lines.append(
            f"Maximum epochs: {int(valid_epoch_counts.max())}"
        )

        summary_lines.append(
            f"Median epochs:  {float(valid_epoch_counts.median()):.1f}"
        )

        summary_lines.append(
            f"Mean epochs:    {float(valid_epoch_counts.mean()):.1f}"
        )

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("IMPORTANT")
summary_lines.append("=" * 80)
summary_lines.append(
    "This audit ONLY reads existing epoch files."
)
summary_lines.append(
    "No raw SET files were modified."
)
summary_lines.append(
    "No FDT files were modified."
)
summary_lines.append(
    "No preprocessed FIF files were modified."
)
summary_lines.append(
    "No epoch files were modified."
)

summary = "\n".join(summary_lines)

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(summary)


# ====================================================================
# TERMINAL OUTPUT
# ====================================================================

print()
print()
print("=" * 80)
print("EPOCH AUDIT COMPLETE")
print("=" * 80)

print()
print("STATUS COUNTS")
print(status_counts)

print()
print(
    f"Zero-epoch files: {len(zero_epoch_files)}"
)

print(
    f"Files requiring review: {len(review_files)}"
)

print(
    f"Failed files: {len(failed_files)}"
)

print()
print("Saved:")
print(OUTPUT_CSV)
print(OUTPUT_SUMMARY)

print()
print("=" * 80)
print("RAW DATA WAS NOT MODIFIED.")
print("NO SET FILES WERE MODIFIED.")
print("NO FDT FILES WERE MODIFIED.")
print("NO PREPROCESSED FIF FILES WERE MODIFIED.")
print("NO EPOCH FILES WERE MODIFIED.")
print("=" * 80)