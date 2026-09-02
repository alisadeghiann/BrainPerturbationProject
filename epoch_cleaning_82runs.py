import os
import re
import glob
import numpy as np
import pandas as pd
import mne

# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

INPUT_DIR = os.path.join(BASE, "epochs_v3")
OUTPUT_DIR = os.path.join(BASE, "epochs_clean")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# The only run excluded by final QC
EXCLUDED_RUNS = {
    ("sub-004", 2)
}

# ============================================================
# OUTPUT FILES
# ============================================================

LOG_CSV = os.path.join(
    LOG_DIR,
    "epoch_cleaning_82runs.csv"
)

SUMMARY_TXT = os.path.join(
    LOG_DIR,
    "epoch_cleaning_summary.txt"
)

# ============================================================
# FIND FILES
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "*_epo.fif"
        )
    )
)

print("=" * 80)
print("EPOCH CLEANING - 82 RUNS")
print("=" * 80)

print(f"Epoch files found: {len(files)}")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================
# PARSE SUBJECT / RUN
# ============================================================

def parse_subject_run(path):

    name = os.path.basename(path)

    subject_match = re.search(
        r"(sub-\d+)",
        name
    )

    run_match = re.search(
        r"_run-(\d+)",
        name
    )

    subject = (
        subject_match.group(1)
        if subject_match
        else None
    )

    run = (
        int(run_match.group(1))
        if run_match
        else None
    )

    return subject, run


# ============================================================
# PROCESS
# ============================================================

records = []

total_before = 0
total_after = 0

processed = 0
failed = 0
excluded = 0

for idx, path in enumerate(files, 1):

    filename = os.path.basename(path)

    subject, run = parse_subject_run(path)

    print()
    print("=" * 80)
    print(f"[{idx}/{len(files)}] {filename}")
    print("=" * 80)

    # --------------------------------------------------------
    # EXCLUSION
    # --------------------------------------------------------

    if (subject, run) in EXCLUDED_RUNS:

        print("FINAL QC EXCLUSION")
        print(f"Excluded: {subject} RUN {run}")

        excluded += 1

        records.append({
            "file": filename,
            "subject": subject,
            "run": run,
            "status": "EXCLUDED_FINAL_QC",
            "n_epochs_before": np.nan,
            "n_epochs_after": np.nan,
            "n_channels": np.nan,
            "n_times": np.nan,
            "sfreq": np.nan,
            "duration_sec": np.nan,
            "nan_percent": np.nan,
            "inf_percent": np.nan,
            "baseline_applied": False,
            "reason": "FINAL_QC_EXCLUSION"
        })

        continue

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    try:

        print("Reading epochs...")

        epochs = mne.read_epochs(
            path,
            preload=True,
            verbose=False
        )

        n_before = len(epochs)
        n_channels = len(epochs.ch_names)
        n_times = len(epochs.times)
        sfreq = float(epochs.info["sfreq"])

        print(f"Epochs before cleaning: {n_before}")
        print(f"Channels: {n_channels}")
        print(f"Time points: {n_times}")
        print(f"Sampling rate: {sfreq}")

        if n_before == 0:

            raise ValueError(
                "ZERO_EPOCHS"
            )

        # ----------------------------------------------------
        # DATA VALIDATION
        # ----------------------------------------------------

        data = epochs.get_data()

        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()

        total_values = data.size

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

        print(
            f"NaN: {nan_percent:.6f}%"
        )

        print(
            f"Inf: {inf_percent:.6f}%"
        )

        if nan_count > 0:
            raise ValueError(
                f"NAN_DATA_{nan_count}"
            )

        if inf_count > 0:
            raise ValueError(
                f"INF_DATA_{inf_count}"
            )

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        print()
        print("Applying baseline correction...")

        baseline_applied = False

        # Only apply baseline if epochs contain a
        # negative-time interval.
        tmin = float(epochs.times[0])

        if tmin < 0:

            epochs.apply_baseline(
                baseline=(None, 0),
                verbose=False
            )

            baseline_applied = True

            print(
                "Baseline: applied using "
                "pre-stimulus interval to 0 sec"
            )

        else:

            print(
                "Baseline: NOT applied "
                "(no pre-stimulus interval)"
            )

        # ----------------------------------------------------
        # POST-BASELINE VALIDATION
        # ----------------------------------------------------

        data_after = epochs.get_data()

        nan_after = np.isnan(data_after).sum()
        inf_after = np.isinf(data_after).sum()

        if nan_after > 0:
            raise ValueError(
                "NAN_AFTER_BASELINE"
            )

        if inf_after > 0:
            raise ValueError(
                "INF_AFTER_BASELINE"
            )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            filename.replace(
                "_epo.fif",
                "_clean_epo.fif"
            )
        )

        print()
        print("Saving:")
        print(output_path)

        epochs.save(
            output_path,
            overwrite=True,
            verbose=False
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        check = mne.read_epochs(
            output_path,
            preload=False,
            verbose=False
        )

        n_after = len(check)

        print()
        print("VALIDATION")
        print("-" * 70)
        print(f"Saved epochs: {n_after}")
        print(f"Saved channels: {len(check.ch_names)}")
        print(f"Saved samples/epoch: {len(check.times)}")
        print(f"Saved sfreq: {check.info['sfreq']}")

        if n_after != n_before:

            raise ValueError(
                f"EPOCH_COUNT_CHANGED_{n_before}_TO_{n_after}"
            )

        total_before += n_before
        total_after += n_after
        processed += 1

        records.append({
            "file": filename,
            "subject": subject,
            "run": run,
            "status": "SUCCESS",
            "n_epochs_before": n_before,
            "n_epochs_after": n_after,
            "n_channels": n_channels,
            "n_times": n_times,
            "sfreq": sfreq,
            "duration_sec": (
                n_times / sfreq
            ),
            "nan_percent": nan_percent,
            "inf_percent": inf_percent,
            "baseline_applied": baseline_applied,
            "reason": ""
        })

        print()
        print("STATUS: SUCCESS")

    except Exception as e:

        failed += 1

        error_text = str(e)

        print()
        print("STATUS: FAILED")
        print(f"ERROR: {error_text}")

        records.append({
            "file": filename,
            "subject": subject,
            "run": run,
            "status": "FAILED",
            "n_epochs_before": np.nan,
            "n_epochs_after": np.nan,
            "n_channels": np.nan,
            "n_times": np.nan,
            "sfreq": np.nan,
            "duration_sec": np.nan,
            "nan_percent": np.nan,
            "inf_percent": np.nan,
            "baseline_applied": False,
            "reason": error_text
        })


# ============================================================
# SAVE LOG
# ============================================================

df = pd.DataFrame(records)

df.to_csv(
    LOG_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

success_count = int(
    (df["status"] == "SUCCESS").sum()
)

failed_count = int(
    (df["status"] == "FAILED").sum()
)

excluded_count = int(
    (df["status"] == "EXCLUDED_FINAL_QC").sum()
)

summary = []

summary.append("=" * 80)
summary.append("EPOCH CLEANING COMPLETE")
summary.append("=" * 80)
summary.append("")

summary.append(
    f"Epoch files found: {len(files)}"
)

summary.append(
    f"Successfully processed: {success_count}"
)

summary.append(
    f"Excluded by final QC: {excluded_count}"
)

summary.append(
    f"Failed: {failed_count}"
)

summary.append("")

summary.append(
    f"Total epochs before cleaning: {total_before}"
)

summary.append(
    f"Total epochs after cleaning:  {total_after}"
)

summary.append(
    f"Epochs removed by cleaning:   "
    f"{total_before - total_after}"
)

summary.append("")

summary.append("=" * 80)
summary.append("FAILED FILES")
summary.append("=" * 80)

failed_df = df[
    df["status"] == "FAILED"
]

if len(failed_df) == 0:

    summary.append("NONE")

else:

    for _, row in failed_df.iterrows():

        summary.append(
            f"{row['file']} | {row['reason']}"
        )

summary.append("")

summary.append("=" * 80)
summary.append("EXCLUDED RUNS")
summary.append("=" * 80)

excluded_df = df[
    df["status"] == "EXCLUDED_FINAL_QC"
]

if len(excluded_df) == 0:

    summary.append("NONE")

else:

    for _, row in excluded_df.iterrows():

        summary.append(
            f"{row['subject']} RUN {int(row['run'])} "
            f"| FINAL_QC_EXCLUSION"
        )

summary.append("")

summary.append("=" * 80)
summary.append("OUTPUT")
summary.append("=" * 80)

summary.append(
    f"Directory: {OUTPUT_DIR}"
)

summary.append(
    f"Log: {LOG_CSV}"
)

summary.append(
    f"Summary: {SUMMARY_TXT}"
)

summary.append("")

summary.append("=" * 80)
summary.append("IMPORTANT")
summary.append("=" * 80)

summary.append(
    "RAW DATA WAS NOT MODIFIED."
)

summary.append(
    "NO SET FILE WAS MODIFIED."
)

summary.append(
    "NO FDT FILE WAS MODIFIED."
)

summary.append(
    "NO EPOCHS_V3 FILE WAS MODIFIED."
)

summary.append(
    "ONLY NEW CLEAN EPOCH FILES WERE CREATED."
)

text = "\n".join(summary)

with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(text)

print()
print(text)