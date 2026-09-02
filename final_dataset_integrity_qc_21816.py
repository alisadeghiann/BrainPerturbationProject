import os
import glob
import numpy as np
import pandas as pd
import mne
from collections import Counter

# ============================================================
# FINAL DATASET INTEGRITY QC
# ============================================================

PROJECT = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

INPUT_DIR = os.path.join(
    PROJECT,
    "epochs_clean",
    "logs",
    "perturbation_dataset",
    "ELIGIBLE"
)

OUTPUT_DIR = os.path.join(
    PROJECT,
    "epochs_clean",
    "logs",
    "perturbation_dataset",
    "final_integrity_qc"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_OUT = os.path.join(
    OUTPUT_DIR,
    "final_dataset_integrity_qc.csv"
)

SUMMARY_OUT = os.path.join(
    OUTPUT_DIR,
    "final_dataset_integrity_qc_summary.txt"
)

EXPECTED_CHANNELS = 71
EXPECTED_SFREQ = 500.0
EXPECTED_TIMES = 501

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fif")))

print("=" * 80)
print("FINAL DATASET INTEGRITY QC")
print("=" * 80)

print(f"Input directory: {INPUT_DIR}")
print(f"Files found: {len(files)}")
print()

records = []

for i, filepath in enumerate(files, 1):

    fname = os.path.basename(filepath)

    print("=" * 80)
    print(f"[{i}/{len(files)}] {fname}")
    print("=" * 80)

    rec = {
        "file": fname,
        "epochs": 0,
        "channels": 0,
        "times": 0,
        "sfreq": np.nan,
        "duration_sec": np.nan,
        "nan_percent": np.nan,
        "inf_percent": np.nan,
        "global_min": np.nan,
        "global_max": np.nan,
        "global_std": np.nan,
        "max_abs": np.nan,
        "event_count": 0,
        "unique_event_types": 0,
        "channel_consistency": "FAIL",
        "sfreq_check": "FAIL",
        "timepoint_check": "FAIL",
        "nan_check": "FAIL",
        "inf_check": "FAIL",
        "event_check": "FAIL",
        "status": "FAIL",
        "reasons": ""
    }

    reasons = []

    try:
        epochs = mne.read_epochs(
            filepath,
            preload=True,
            verbose=False
        )

        data = epochs.get_data()

        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        n_times = len(epochs.times)
        sfreq = float(epochs.info["sfreq"])

        rec["epochs"] = n_epochs
        rec["channels"] = n_channels
        rec["times"] = n_times
        rec["sfreq"] = sfreq
        rec["duration_sec"] = float(
            epochs.times[-1] - epochs.times[0]
        ) if n_times > 1 else 0

        print(f"Epochs:        {n_epochs}")
        print(f"Channels:      {n_channels}")
        print(f"Time points:   {n_times}")
        print(f"Sampling rate: {sfreq}")

        # ----------------------------------------------------
        # CHANNEL CHECK
        # ----------------------------------------------------

        if n_channels == EXPECTED_CHANNELS:
            rec["channel_consistency"] = "PASS"
        else:
            reasons.append(
                f"CHANNELS_{n_channels}"
            )

        # ----------------------------------------------------
        # SFREQ CHECK
        # ----------------------------------------------------

        if np.isclose(sfreq, EXPECTED_SFREQ, atol=0.1):
            rec["sfreq_check"] = "PASS"
        else:
            reasons.append(
                f"SFREQ_{sfreq}"
            )

        # ----------------------------------------------------
        # TIMEPOINT CHECK
        # ----------------------------------------------------

        if n_times == EXPECTED_TIMES:
            rec["timepoint_check"] = "PASS"
        else:
            reasons.append(
                f"TIMEPOINTS_{n_times}"
            )

        # ----------------------------------------------------
        # NUMERIC CHECKS
        # ----------------------------------------------------

        total_values = data.size

        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())

        nan_percent = (
            100.0 * nan_count / total_values
            if total_values else 0
        )

        inf_percent = (
            100.0 * inf_count / total_values
            if total_values else 0
        )

        rec["nan_percent"] = nan_percent
        rec["inf_percent"] = inf_percent

        finite_data = data[np.isfinite(data)]

        if finite_data.size > 0:
            rec["global_min"] = float(np.min(finite_data))
            rec["global_max"] = float(np.max(finite_data))
            rec["global_std"] = float(np.std(finite_data))
            rec["max_abs"] = float(np.max(np.abs(finite_data)))

        if nan_count == 0:
            rec["nan_check"] = "PASS"
        else:
            reasons.append(
                f"NAN_{nan_percent:.6f}%"
            )

        if inf_count == 0:
            rec["inf_check"] = "PASS"
        else:
            reasons.append(
                f"INF_{inf_percent:.6f}%"
            )

        # ----------------------------------------------------
        # EVENT / CONDITION CHECK
        # ----------------------------------------------------

        event_ids = epochs.events[:, 2]

        rec["event_count"] = len(event_ids)
        rec["unique_event_types"] = len(np.unique(event_ids))

        if len(event_ids) == n_epochs:
            rec["event_check"] = "PASS"
        else:
            reasons.append(
                f"EVENT_EPOCH_MISMATCH_{len(event_ids)}_{n_epochs}"
            )

        # ----------------------------------------------------
        # BASIC EPOCH COUNT CHECK
        # ----------------------------------------------------

        if n_epochs <= 0:
            reasons.append("ZERO_EPOCHS")

        # ----------------------------------------------------
        # FINAL STATUS
        # ----------------------------------------------------

        if len(reasons) == 0:
            rec["status"] = "PASS"
            rec["reasons"] = "NONE"
        else:
            rec["status"] = "FAIL"
            rec["reasons"] = ";".join(reasons)

        print(
            f"NaN: {nan_percent:.6f}% | "
            f"Inf: {inf_percent:.6f}%"
        )

        print(
            f"Range: {rec['global_min']:.3f} "
            f"to {rec['global_max']:.3f}"
        )

        print(
            f"Events: {rec['event_count']} | "
            f"Unique event IDs: {rec['unique_event_types']}"
        )

        print(f"STATUS: {rec['status']}")

        if rec["reasons"] != "NONE":
            print(f"REASONS: {rec['reasons']}")

    except Exception as e:

        rec["status"] = "ERROR"
        rec["reasons"] = (
            "EXCEPTION_" +
            str(e).replace(";", ",")[:500]
        )

        print("ERROR:")
        print(e)

    records.append(rec)


# ============================================================
# MASTER TABLE
# ============================================================

df = pd.DataFrame(records)

df.to_csv(
    CSV_OUT,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

status_counts = df["status"].value_counts()

total_files = len(df)
pass_files = int((df["status"] == "PASS").sum())
fail_files = int((df["status"] == "FAIL").sum())
error_files = int((df["status"] == "ERROR").sum())

total_epochs = int(df["epochs"].sum())

summary_lines = []

summary_lines.append("=" * 80)
summary_lines.append("FINAL DATASET INTEGRITY QC SUMMARY")
summary_lines.append("=" * 80)
summary_lines.append("")

summary_lines.append(f"Files checked:        {total_files}")
summary_lines.append(f"PASS files:           {pass_files}")
summary_lines.append(f"FAIL files:           {fail_files}")
summary_lines.append(f"ERROR files:          {error_files}")
summary_lines.append("")

summary_lines.append(f"Total epochs:         {total_epochs}")
summary_lines.append("")

summary_lines.append("=" * 80)
summary_lines.append("STATUS COUNTS")
summary_lines.append("=" * 80)
summary_lines.append("")

for status, count in status_counts.items():
    summary_lines.append(
        f"{status:<15} {count}"
    )

summary_lines.append("")

summary_lines.append("=" * 80)
summary_lines.append("CHECK RESULTS")
summary_lines.append("=" * 80)
summary_lines.append("")

for col in [
    "channel_consistency",
    "sfreq_check",
    "timepoint_check",
    "nan_check",
    "inf_check",
    "event_check"
]:
    counts = df[col].value_counts()

    summary_lines.append(f"{col}:")
    for key, value in counts.items():
        summary_lines.append(
            f"  {key}: {value}"
        )
    summary_lines.append("")

summary_lines.append("=" * 80)
summary_lines.append("FAILED FILES")
summary_lines.append("=" * 80)
summary_lines.append("")

failed = df[df["status"] != "PASS"]

if len(failed) == 0:
    summary_lines.append("NONE")
else:
    for _, row in failed.iterrows():
        summary_lines.append(
            f"{row['file']} | {row['status']} | "
            f"{row['reasons']}"
        )

summary_lines.append("")

summary_lines.append("=" * 80)
summary_lines.append("FINAL DATASET DECISION")
summary_lines.append("=" * 80)
summary_lines.append("")

if (
    total_files > 0
    and pass_files == total_files
    and total_epochs == 21816
):
    final_status = "PASS - DATASET READY FOR PERTURBATION ANALYSIS"
else:
    final_status = "REVIEW - DATASET REQUIRES INVESTIGATION"

summary_lines.append(final_status)
summary_lines.append("")

summary_lines.append("=" * 80)
summary_lines.append("IMPORTANT")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append("NO RAW DATA WAS MODIFIED.")
summary_lines.append("NO SET FILE WAS MODIFIED.")
summary_lines.append("NO FDT FILE WAS MODIFIED.")
summary_lines.append("NO EPOCH FILE WAS MODIFIED.")
summary_lines.append("ONLY READ-ONLY QC WAS PERFORMED.")
summary_lines.append("")

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(summary_lines))

print()
print("=" * 80)
print("FINAL DATASET INTEGRITY QC COMPLETE")
print("=" * 80)

print()
print("STATUS COUNTS")
print(status_counts)

print()
print(f"Total files:   {total_files}")
print(f"PASS:          {pass_files}")
print(f"FAIL:          {fail_files}")
print(f"ERROR:         {error_files}")
print(f"Total epochs:  {total_epochs}")

print()
print("=" * 80)
print("FINAL DECISION")
print("=" * 80)
print(final_status)

print()
print("Saved:")
print(CSV_OUT)
print(SUMMARY_OUT)

print()
print("NO DATA WAS MODIFIED.")