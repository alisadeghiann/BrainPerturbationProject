import os
import glob
import numpy as np
import pandas as pd
import mne

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
    "final_integrity_qc_v2"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_OUT = os.path.join(
    OUTPUT_DIR,
    "final_dataset_integrity_qc_v2.csv"
)

SUMMARY_OUT = os.path.join(
    OUTPUT_DIR,
    "final_dataset_integrity_qc_v2_summary.txt"
)

EXPECTED_CHANNELS = 71
EXPECTED_TIMES = 501

files = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "*.fif")
    )
)

print("=" * 80)
print("FINAL DATASET INTEGRITY QC V2")
print("=" * 80)

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
        "unique_event_ids": 0,
        "channel_check": "FAIL",
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

        if n_times > 1:
            rec["duration_sec"] = float(
                epochs.times[-1] - epochs.times[0]
            )

        print(f"Epochs:        {n_epochs}")
        print(f"Channels:      {n_channels}")
        print(f"Time points:   {n_times}")
        print(f"Sampling rate: {sfreq}")

        # ====================================================
        # CHANNEL CHECK
        # ====================================================

        if n_channels == EXPECTED_CHANNELS:
            rec["channel_check"] = "PASS"
        else:
            reasons.append(
                f"CHANNELS_{n_channels}"
            )

        # ====================================================
        # TIMEPOINT CHECK
        # ====================================================

        if n_times == EXPECTED_TIMES:
            rec["timepoint_check"] = "PASS"
        else:
            reasons.append(
                f"TIMEPOINTS_{n_times}"
            )

        # ====================================================
        # NAN / INF
        # ====================================================

        total_values = data.size

        nan_count = int(
            np.isnan(data).sum()
        )

        inf_count = int(
            np.isinf(data).sum()
        )

        nan_percent = (
            100.0 * nan_count / total_values
            if total_values > 0 else 0
        )

        inf_percent = (
            100.0 * inf_count / total_values
            if total_values > 0 else 0
        )

        rec["nan_percent"] = nan_percent
        rec["inf_percent"] = inf_percent

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

        # ====================================================
        # DATA RANGE
        # ====================================================

        finite = data[np.isfinite(data)]

        if finite.size > 0:

            rec["global_min"] = float(
                np.min(finite)
            )

            rec["global_max"] = float(
                np.max(finite)
            )

            rec["global_std"] = float(
                np.std(finite)
            )

            rec["max_abs"] = float(
                np.max(np.abs(finite))
            )

        # ====================================================
        # EVENT CHECK
        # ====================================================

        if epochs.events is not None:

            event_ids = epochs.events[:, 2]

            rec["event_count"] = len(event_ids)

            rec["unique_event_ids"] = len(
                np.unique(event_ids)
            )

            if len(event_ids) == n_epochs:
                rec["event_check"] = "PASS"
            else:
                reasons.append(
                    f"EVENT_MISMATCH_{len(event_ids)}_{n_epochs}"
                )

        else:

            reasons.append(
                "NO_EVENTS"
            )

        # ====================================================
        # ZERO EPOCH CHECK
        # ====================================================

        if n_epochs <= 0:

            reasons.append(
                "ZERO_EPOCHS"
            )

        # ====================================================
        # FINAL STATUS
        # ====================================================

        if len(reasons) == 0:

            rec["status"] = "PASS"
            rec["reasons"] = "NONE"

        else:

            rec["status"] = "FAIL"
            rec["reasons"] = ";".join(
                reasons
            )

        print(
            f"NaN: {nan_percent:.6f}% | "
            f"Inf: {inf_percent:.6f}%"
        )

        print(
            f"Range: "
            f"{rec['global_min']:.3f} "
            f"to "
            f"{rec['global_max']:.3f}"
        )

        print(
            f"Events: {rec['event_count']} | "
            f"Unique IDs: {rec['unique_event_ids']}"
        )

        print(
            f"STATUS: {rec['status']}"
        )

        if rec["reasons"] != "NONE":

            print(
                f"REASONS: {rec['reasons']}"
            )

    except Exception as e:

        rec["status"] = "ERROR"

        rec["reasons"] = (
            "EXCEPTION_" +
            str(e)
            .replace(";", ",")
            [:500]
        )

        print("ERROR:")
        print(e)

    records.append(rec)


# ============================================================
# DATAFRAME
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

status_counts = (
    df["status"]
    .value_counts()
)

sfreq_counts = (
    df["sfreq"]
    .round(6)
    .value_counts()
    .sort_index()
)

total_files = len(df)

pass_files = int(
    (df["status"] == "PASS").sum()
)

fail_files = int(
    (df["status"] == "FAIL").sum()
)

error_files = int(
    (df["status"] == "ERROR").sum()
)

total_epochs = int(
    df["epochs"].sum()
)

summary = []

summary.append("=" * 80)
summary.append(
    "FINAL DATASET INTEGRITY QC V2 SUMMARY"
)
summary.append("=" * 80)
summary.append("")

summary.append(
    f"Files checked: {total_files}"
)

summary.append(
    f"PASS:          {pass_files}"
)

summary.append(
    f"FAIL:          {fail_files}"
)

summary.append(
    f"ERROR:         {error_files}"
)

summary.append(
    f"Total epochs:  {total_epochs}"
)

summary.append("")

summary.append("=" * 80)
summary.append("SAMPLING RATE DISTRIBUTION")
summary.append("=" * 80)
summary.append("")

for rate, count in sfreq_counts.items():

    summary.append(
        f"{rate} Hz : {count} files"
    )

summary.append("")

summary.append("=" * 80)
summary.append("CHECK RESULTS")
summary.append("=" * 80)
summary.append("")

for col in [
    "channel_check",
    "timepoint_check",
    "nan_check",
    "inf_check",
    "event_check"
]:

    summary.append(col)

    counts = (
        df[col]
        .value_counts()
    )

    for key, value in counts.items():

        summary.append(
            f"  {key}: {value}"
        )

    summary.append("")

summary.append("=" * 80)
summary.append("FAILED / ERROR FILES")
summary.append("=" * 80)
summary.append("")

failed = df[
    df["status"] != "PASS"
]

if len(failed) == 0:

    summary.append("NONE")

else:

    for _, row in failed.iterrows():

        summary.append(
            f"{row['file']} | "
            f"{row['status']} | "
            f"{row['reasons']}"
        )

summary.append("")

summary.append("=" * 80)
summary.append("FINAL DECISION")
summary.append("=" * 80)
summary.append("")

if (
    total_files == 82
    and total_epochs == 21816
    and fail_files == 0
    and error_files == 0
):

    final_status = (
        "PASS - DATASET READY "
        "FOR PERTURBATION ANALYSIS"
    )

else:

    final_status = (
        "REVIEW - DATASET "
        "REQUIRES INVESTIGATION"
    )

summary.append(final_status)

summary.append("")

summary.append("=" * 80)
summary.append("IMPORTANT")
summary.append("=" * 80)
summary.append("")

summary.append(
    "SAMPLING RATE WAS NOT USED AS AN EXCLUSION CRITERION."
)

summary.append(
    "NO DATA WAS RESAMPLED."
)

summary.append(
    "NO DATA WAS MODIFIED."
)

summary.append(
    "NO RAW DATA WAS MODIFIED."
)

summary.append(
    "NO SET FILE WAS MODIFIED."
)

summary.append(
    "NO FDT FILE WAS MODIFIED."
)

summary.append(
    "NO EPOCH FILE WAS MODIFIED."
)

summary.append(
    "READ-ONLY QC ONLY."
)

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print()
print("=" * 80)
print("FINAL DATASET INTEGRITY QC V2 COMPLETE")
print("=" * 80)

print()
print("STATUS COUNTS")
print(status_counts)

print()
print("SAMPLING RATE DISTRIBUTION")

for rate, count in sfreq_counts.items():

    print(
        f"{rate} Hz : {count} files"
    )

print()
print(f"Total files:  {total_files}")
print(f"PASS:         {pass_files}")
print(f"FAIL:         {fail_files}")
print(f"ERROR:        {error_files}")
print(f"Total epochs: {total_epochs}")

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