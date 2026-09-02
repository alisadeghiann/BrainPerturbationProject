from pathlib import Path
import numpy as np
import pandas as pd
import mne

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

INPUT_DIR = (
    BASE_DIR
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v4"
    / "ELIGIBLE"
)

OUTPUT_DIR = INPUT_DIR / "final_integrity_qc_v4"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_OUT = (
    OUTPUT_DIR
    / "final_dataset_integrity_qc_v4.csv"
)

SUMMARY_OUT = (
    OUTPUT_DIR
    / "final_dataset_integrity_qc_v4_summary.txt"
)

# ============================================================
# EXPECTED STANDARD
# ============================================================

EXPECTED_FILES = 82
EXPECTED_EPOCHS = 21816
EXPECTED_CHANNELS = 71
EXPECTED_SFREQ = 500.0
EXPECTED_TIMES = 501
EXPECTED_TMIN = -0.2
EXPECTED_TMAX = 0.8

SFREQ_TOL = 1e-6
TIME_TOL = 1e-9

# ============================================================
# FIND FILES
# ============================================================

files = sorted(
    INPUT_DIR.glob(
        "*_standardized_epo.fif"
    )
)

print("=" * 80)
print("FINAL DATASET INTEGRITY QC V4")
print("=" * 80)

print()
print("Input directory:")
print(INPUT_DIR)

print()
print(f"Standardized files found: {len(files)}")

records = []

# ============================================================
# PROCESS
# ============================================================

for i, filepath in enumerate(files, 1):

    print()
    print("=" * 80)
    print(
        f"[{i}/{len(files)}] "
        f"{filepath.name}"
    )
    print("=" * 80)

    rec = {
        "file": filepath.name,
        "epochs": np.nan,
        "channels": np.nan,
        "sfreq": np.nan,
        "n_times": np.nan,
        "tmin": np.nan,
        "tmax": np.nan,
        "nan_percent": np.nan,
        "inf_percent": np.nan,
        "global_min": np.nan,
        "global_max": np.nan,
        "global_std": np.nan,
        "max_abs": np.nan,
        "event_count": np.nan,
        "unique_event_ids": np.nan,
        "channel_check": "NOT_CHECKED",
        "sfreq_check": "NOT_CHECKED",
        "timepoint_check": "NOT_CHECKED",
        "time_range_check": "NOT_CHECKED",
        "nan_check": "NOT_CHECKED",
        "inf_check": "NOT_CHECKED",
        "event_check": "NOT_CHECKED",
        "status": "ERROR",
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
        tmin = float(epochs.tmin)
        tmax = float(epochs.tmax)

        rec["epochs"] = n_epochs
        rec["channels"] = n_channels
        rec["n_times"] = n_times
        rec["sfreq"] = sfreq
        rec["tmin"] = tmin
        rec["tmax"] = tmax

        print(f"Epochs:        {n_epochs}")
        print(f"Channels:      {n_channels}")
        print(f"Samples:       {n_times}")
        print(f"Sampling rate: {sfreq}")
        print(
            f"Time range:    "
            f"{tmin:.12f} to {tmax:.12f}"
        )

        # ====================================================
        # EPOCH COUNT
        # ====================================================

        if n_epochs > 0:
            rec["epoch_count_check"] = "PASS"
        else:
            rec["epoch_count_check"] = "FAIL"
            reasons.append("ZERO_EPOCHS")

        # ====================================================
        # CHANNEL CHECK
        # ====================================================

        if n_channels == EXPECTED_CHANNELS:
            rec["channel_check"] = "PASS"
        else:
            rec["channel_check"] = "FAIL"
            reasons.append(
                f"CHANNELS_{n_channels}"
            )

        # ====================================================
        # SAMPLING RATE
        # ====================================================

        if abs(
            sfreq - EXPECTED_SFREQ
        ) <= SFREQ_TOL:

            rec["sfreq_check"] = "PASS"

        else:

            rec["sfreq_check"] = "FAIL"

            reasons.append(
                f"SFREQ_{sfreq}"
            )

        # ====================================================
        # TIMEPOINTS
        # ====================================================

        if n_times == EXPECTED_TIMES:

            rec["timepoint_check"] = "PASS"

        else:

            rec["timepoint_check"] = "FAIL"

            reasons.append(
                f"TIMEPOINTS_{n_times}"
            )

        # ====================================================
        # TIME RANGE
        # ====================================================

        if (
            abs(tmin - EXPECTED_TMIN)
            <= TIME_TOL
            and
            abs(tmax - EXPECTED_TMAX)
            <= TIME_TOL
        ):

            rec["time_range_check"] = "PASS"

        else:

            rec["time_range_check"] = "FAIL"

            reasons.append(
                f"TIME_RANGE_{tmin}_{tmax}"
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
            if total_values > 0
            else 0.0
        )

        inf_percent = (
            100.0 * inf_count / total_values
            if total_values > 0
            else 0.0
        )

        rec["nan_percent"] = nan_percent
        rec["inf_percent"] = inf_percent

        if nan_count == 0:

            rec["nan_check"] = "PASS"

        else:

            rec["nan_check"] = "FAIL"

            reasons.append(
                f"NAN_{nan_percent:.6f}%"
            )

        if inf_count == 0:

            rec["inf_check"] = "PASS"

        else:

            rec["inf_check"] = "FAIL"

            reasons.append(
                f"INF_{inf_percent:.6f}%"
            )

        # ====================================================
        # DATA RANGE
        # ====================================================

        finite = data[
            np.isfinite(data)
        ]

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

        # ====================================================
        # EVENT CHECK
        # ====================================================

        if epochs.events is not None:

            event_ids = epochs.events[:, 2]

            event_count = len(
                event_ids
            )

            unique_ids = len(
                np.unique(event_ids)
            )

            rec["event_count"] = event_count
            rec["unique_event_ids"] = unique_ids

            if event_count == n_epochs:

                rec["event_check"] = "PASS"

            else:

                rec["event_check"] = "FAIL"

                reasons.append(
                    f"EVENT_MISMATCH_"
                    f"{event_count}_{n_epochs}"
                )

            print(
                f"Events: {event_count} | "
                f"Unique IDs: {unique_ids}"
            )

        else:

            rec["event_check"] = "FAIL"

            reasons.append(
                "NO_EVENTS"
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
            f"STATUS: {rec['status']}"
        )

        if rec["reasons"] != "NONE":

            print(
                f"REASONS: "
                f"{rec['reasons']}"
            )

    except Exception as e:

        rec["status"] = "ERROR"

        rec["reasons"] = (
            "EXCEPTION_"
            + str(e)
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
    df["epochs"]
    .fillna(0)
    .sum()
)

status_counts = (
    df["status"]
    .value_counts()
)

summary = []

summary.append("=" * 80)
summary.append(
    "FINAL DATASET INTEGRITY QC V4 SUMMARY"
)
summary.append("=" * 80)
summary.append("")

summary.append(
    f"Files found:       {total_files}"
)

summary.append(
    f"Expected files:    {EXPECTED_FILES}"
)

summary.append(
    f"PASS:               {pass_files}"
)

summary.append(
    f"FAIL:               {fail_files}"
)

summary.append(
    f"ERROR:              {error_files}"
)

summary.append(
    f"Total epochs:       {total_epochs}"
)

summary.append(
    f"Expected epochs:    {EXPECTED_EPOCHS}"
)

summary.append("")

summary.append("=" * 80)
summary.append("STANDARD")
summary.append("=" * 80)
summary.append("")

summary.append(
    f"SFREQ:              {EXPECTED_SFREQ} Hz"
)

summary.append(
    f"TMIN:               {EXPECTED_TMIN}"
)

summary.append(
    f"TMAX:               {EXPECTED_TMAX}"
)

summary.append(
    f"SAMPLES/EPOCH:      {EXPECTED_TIMES}"
)

summary.append(
    f"CHANNELS:           {EXPECTED_CHANNELS}"
)

summary.append("")

summary.append("=" * 80)
summary.append("CHECK COUNTS")
summary.append("=" * 80)
summary.append("")

for col in [
    "channel_check",
    "sfreq_check",
    "timepoint_check",
    "time_range_check",
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

# ============================================================
# FAILED FILES
# ============================================================

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

# ============================================================
# FINAL DECISION
# ============================================================

summary.append("=" * 80)
summary.append("FINAL DECISION")
summary.append("=" * 80)
summary.append("")

if (
    total_files == EXPECTED_FILES
    and total_epochs == EXPECTED_EPOCHS
    and pass_files == EXPECTED_FILES
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
    "READ-ONLY QC."
)

summary.append(
    "NO DATA WAS MODIFIED."
)

summary.append(
    "NO ORIGINAL ELIGIBLE FILE WAS MODIFIED."
)

summary.append(
    "NO EPOCH DATA WAS RESAMPLED."
)

summary.append(
    "NO EPOCH DATA WAS CROPPED."
)

summary.append(
    "NO EPOCH DATA WAS DELETED."
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
# CONSOLE OUTPUT
# ============================================================

print()
print("=" * 80)
print("FINAL DATASET INTEGRITY QC V4 COMPLETE")
print("=" * 80)

print()
print("STATUS COUNTS")
print(status_counts)

print()
print(f"Files found:      {total_files}")
print(f"Expected files:   {EXPECTED_FILES}")
print(f"PASS:              {pass_files}")
print(f"FAIL:              {fail_files}")
print(f"ERROR:             {error_files}")

print()
print(f"Total epochs:      {total_epochs}")
print(f"Expected epochs:   {EXPECTED_EPOCHS}")

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