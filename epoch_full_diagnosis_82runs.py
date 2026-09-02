import os
import glob
import re
import traceback
import numpy as np
import pandas as pd
import mne

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

PREPROCESSED_DIR = os.path.join(BASE, "preprocessed")
EPOCH_DIR = os.path.join(BASE, "epochs")
LOG_DIR = os.path.join(EPOCH_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    LOG_DIR,
    "epoch_full_diagnosis_82runs.csv"
)

OUTPUT_SUMMARY = os.path.join(
    LOG_DIR,
    "epoch_full_diagnosis_82runs_summary.txt"
)

# =====================================================================
# FIND PREPROCESSED FILES
# =====================================================================

preprocessed_files = sorted(
    glob.glob(
        os.path.join(
            PREPROCESSED_DIR,
            "*_preprocessed_raw.fif"
        )
    )
)

epoch_files = sorted(
    glob.glob(
        os.path.join(
            EPOCH_DIR,
            "*_epo.fif"
        )
    )
)

print("=" * 80)
print("FULL EPOCH DIAGNOSIS - 82 RUNS")
print("=" * 80)

print()
print("Preprocessed FIF files:", len(preprocessed_files))
print("Epoch FIF files:", len(epoch_files))
print()

# =====================================================================
# HELPER
# =====================================================================

def extract_subject_run(filename):

    subject_match = re.search(
        r"(sub-\d+)",
        filename
    )

    run_match = re.search(
        r"run-(\d+)",
        filename
    )

    subject = (
        subject_match.group(1)
        if subject_match
        else ""
    )

    run = (
        int(run_match.group(1))
        if run_match
        else np.nan
    )

    return subject, run


def get_epoch_filename(preprocessed_filename):

    base = os.path.basename(
        preprocessed_filename
    )

    epoch_name = base.replace(
        "_preprocessed_raw.fif",
        "_epo.fif"
    )

    return os.path.join(
        EPOCH_DIR,
        epoch_name
    )


# =====================================================================
# EXPECTED RUNS
# =====================================================================

expected = {}

for fname in preprocessed_files:

    subject, run = extract_subject_run(
        os.path.basename(fname)
    )

    if subject and not pd.isna(run):

        key = (
            subject,
            int(run)
        )

        expected[key] = fname


# =====================================================================
# DIAGNOSIS
# =====================================================================

results = []

for idx, (key, raw_fname) in enumerate(
    sorted(expected.items()),
    1
):

    subject, run = key

    print()
    print("=" * 80)
    print(
        f"[{idx}/{len(expected)}] "
        f"{subject} RUN {run}"
    )
    print("=" * 80)

    epoch_fname = get_epoch_filename(
        raw_fname
    )

    row = {

        "subject": subject,
        "run": run,

        "preprocessed_file":
            os.path.basename(raw_fname),

        "epoch_file":
            os.path.basename(epoch_fname),

        "epoch_file_exists":
            os.path.exists(epoch_fname),

        "raw_channels": np.nan,
        "raw_eeg_channels": np.nan,
        "raw_sfreq": np.nan,
        "raw_samples": np.nan,
        "raw_duration_sec": np.nan,

        "epoch_exists": False,
        "n_epochs": np.nan,
        "epoch_channels": np.nan,
        "epoch_times": np.nan,
        "epoch_sfreq": np.nan,

        "drop_log_entries": np.nan,
        "dropped_epochs": np.nan,

        "empty_drop_log": False,

        "drop_reasons": "",
        "drop_channels": "",

        "status": "UNKNOWN",
        "reason": ""
    }

    # =================================================================
    # RAW / PREPROCESSED INSPECTION
    # =================================================================

    try:

        print("Reading preprocessed FIF...")

        raw = mne.io.read_raw_fif(
            raw_fname,
            preload=False,
            verbose=False
        )

        row["raw_channels"] = len(
            raw.ch_names
        )

        row["raw_eeg_channels"] = sum(
            ch_type == "eeg"
            for ch_type in raw.get_channel_types()
        )

        row["raw_sfreq"] = raw.info["sfreq"]

        row["raw_samples"] = raw.n_times

        row["raw_duration_sec"] = (
            raw.n_times /
            raw.info["sfreq"]
        )

        print(
            f"Channels: {row['raw_channels']}"
        )

        print(
            f"EEG channels: "
            f"{row['raw_eeg_channels']}"
        )

        print(
            f"Sampling rate: "
            f"{row['raw_sfreq']}"
        )

        print(
            f"Samples: "
            f"{row['raw_samples']}"
        )

        print(
            f"Duration: "
            f"{row['raw_duration_sec']:.3f} sec"
        )

        del raw

    except Exception as e:

        row["status"] = "RAW_READ_FAILED"
        row["reason"] = str(e)

        print()
        print("RAW READ FAILED")
        print(str(e))

        results.append(row)

        continue

    # =================================================================
    # EPOCH FILE EXISTENCE
    # =================================================================

    if not os.path.exists(epoch_fname):

        row["status"] = "EPOCH_FILE_MISSING"

        row["reason"] = (
            "Preprocessed file exists but "
            "corresponding epoch file is missing"
        )

        print()
        print("EPOCH FILE MISSING")

        results.append(row)

        continue

    # =================================================================
    # READ EPOCH FILE
    # =================================================================

    try:

        print()
        print("Reading epoch FIF...")

        epochs = mne.read_epochs(
            epoch_fname,
            preload=False,
            verbose=False
        )

        row["epoch_exists"] = True

        row["n_epochs"] = len(epochs)

        row["epoch_channels"] = len(
            epochs.ch_names
        )

        row["epoch_times"] = len(
            epochs.times
        )

        row["epoch_sfreq"] = (
            epochs.info["sfreq"]
        )

        print(
            f"Epochs retained: "
            f"{row['n_epochs']}"
        )

        print(
            f"Epoch channels: "
            f"{row['epoch_channels']}"
        )

        print(
            f"Epoch time points: "
            f"{row['epoch_times']}"
        )

        # =============================================================
        # DROP LOG
        # =============================================================

        drop_log = epochs.drop_log

        row["drop_log_entries"] = len(
            drop_log
        )

        nonempty_drop_logs = [
            tuple(x)
            for x in drop_log
            if len(x) > 0
        ]

        row["dropped_epochs"] = len(
            nonempty_drop_logs
        )

        row["empty_drop_log"] = (
            len(nonempty_drop_logs) == 0
        )

        # =============================================================
        # DROP REASONS
        # =============================================================

        reason_counter = {}

        channel_counter = {}

        for entry in nonempty_drop_logs:

            for reason in entry:

                reason_counter[reason] = (
                    reason_counter.get(
                        reason,
                        0
                    ) + 1
                )

                # MNE drop reasons sometimes contain
                # channel names directly.
                if reason in epochs.ch_names:

                    channel_counter[reason] = (
                        channel_counter.get(
                            reason,
                            0
                        ) + 1
                    )

        row["drop_reasons"] = ";".join(
            f"{k}:{v}"
            for k, v in sorted(
                reason_counter.items(),
                key=lambda x: -x[1]
            )
        )

        row["drop_channels"] = ";".join(
            f"{k}:{v}"
            for k, v in sorted(
                channel_counter.items(),
                key=lambda x: -x[1]
            )
        )

        # =============================================================
        # STATUS
        # =============================================================

        if len(epochs) == 0:

            row["status"] = "ZERO_EPOCH"

            if len(nonempty_drop_logs) == 0:

                row["reason"] = (
                    "Zero epochs but drop_log "
                    "contains no usable rejection information"
                )

            else:

                row["reason"] = (
                    "All epochs were dropped"
                )

        elif len(epochs) < 10:

            row["status"] = "LOW_EPOCH_COUNT"

            row["reason"] = (
                f"Only {len(epochs)} epochs retained"
            )

        else:

            row["status"] = "PASS"

            row["reason"] = (
                f"{len(epochs)} epochs retained"
            )

        # =============================================================
        # PRINT DROP INFORMATION
        # =============================================================

        print()

        print(
            "Drop-log entries:",
            row["drop_log_entries"]
        )

        print(
            "Dropped epochs:",
            row["dropped_epochs"]
        )

        print(
            "Drop reasons:",
            row["drop_reasons"]
            if row["drop_reasons"]
            else "NONE"
        )

        print(
            "Drop channels:",
            row["drop_channels"]
            if row["drop_channels"]
            else "NONE"
        )

        print()
        print(
            "STATUS:",
            row["status"]
        )

        del epochs

    except Exception as e:

        row["status"] = "EPOCH_READ_FAILED"
        row["reason"] = str(e)

        print()
        print("EPOCH READ FAILED")
        print(str(e))

        traceback.print_exc()

    results.append(row)


# =====================================================================
# CHECK MISSING PREPROCESSED RUNS
# =====================================================================

print()
print("=" * 80)
print("MISSING / EXTRA FILE CHECK")
print("=" * 80)

all_epoch_keys = set()

for fname in epoch_files:

    subject, run = extract_subject_run(
        os.path.basename(fname)
    )

    if subject and not pd.isna(run):

        all_epoch_keys.add(
            (
                subject,
                int(run)
            )
        )

expected_keys = set(
    expected.keys()
)

missing_epoch_keys = sorted(
    expected_keys - all_epoch_keys
)

extra_epoch_keys = sorted(
    all_epoch_keys - expected_keys
)

print()
print(
    "Expected preprocessed runs:",
    len(expected_keys)
)

print(
    "Epoch files:",
    len(all_epoch_keys)
)

print(
    "Missing epoch runs:",
    len(missing_epoch_keys)
)

for subject, run in missing_epoch_keys:

    print(
        f"  {subject} RUN {run}"
    )

print()
print(
    "Extra epoch runs:",
    len(extra_epoch_keys)
)

for subject, run in extra_epoch_keys:

    print(
        f"  {subject} RUN {run}"
    )


# =====================================================================
# DATAFRAME
# =====================================================================

df = pd.DataFrame(results)

df = df.sort_values(
    by=["subject", "run"]
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# =====================================================================
# SUMMARY
# =====================================================================

status_counts = (
    df["status"]
    .value_counts()
    .sort_index()
)

zero_epoch = df[
    df["status"] == "ZERO_EPOCH"
]

missing = df[
    df["status"] == "EPOCH_FILE_MISSING"
]

failed = df[
    df["status"].isin(
        [
            "RAW_READ_FAILED",
            "EPOCH_READ_FAILED"
        ]
    )
]

low_epoch = df[
    df["status"] == "LOW_EPOCH_COUNT"
]

passed = df[
    df["status"] == "PASS"
]


summary = []

summary.append("=" * 80)
summary.append(
    "FULL EPOCH DIAGNOSIS SUMMARY"
)
summary.append("=" * 80)
summary.append("")

summary.append(
    f"Preprocessed runs found: "
    f"{len(preprocessed_files)}"
)

summary.append(
    f"Epoch files found: "
    f"{len(epoch_files)}"
)

summary.append("")

summary.append("STATUS COUNTS")
summary.append("-" * 80)

for status, count in status_counts.items():

    summary.append(
        f"{status:<25} {count}"
    )

summary.append("")
summary.append("=" * 80)
summary.append("ZERO-EPOCH RUNS")
summary.append("=" * 80)

if len(zero_epoch) == 0:

    summary.append("NONE")

else:

    for _, r in zero_epoch.iterrows():

        summary.append(
            f"{r['subject']} RUN {r['run']} | "
            f"dropped={r['dropped_epochs']} | "
            f"reasons={r['drop_reasons']}"
        )


summary.append("")
summary.append("=" * 80)
summary.append("MISSING EPOCH FILES")
summary.append("=" * 80)

if len(missing_epoch_keys) == 0:

    summary.append("NONE")

else:

    for subject, run in missing_epoch_keys:

        summary.append(
            f"{subject} RUN {run}"
        )


summary.append("")
summary.append("=" * 80)
summary.append("LOW EPOCH COUNT")
summary.append("=" * 80)

if len(low_epoch) == 0:

    summary.append("NONE")

else:

    for _, r in low_epoch.iterrows():

        summary.append(
            f"{r['subject']} RUN {r['run']} | "
            f"epochs={r['n_epochs']}"
        )


summary.append("")
summary.append("=" * 80)
summary.append("FAILED RUNS")
summary.append("=" * 80)

if len(failed) == 0:

    summary.append("NONE")

else:

    for _, r in failed.iterrows():

        summary.append(
            f"{r['subject']} RUN {r['run']} | "
            f"{r['status']} | "
            f"{r['reason']}"
        )


summary.append("")
summary.append("=" * 80)
summary.append("PASSED RUNS")
summary.append("=" * 80)

summary.append(
    f"Runs with >=10 retained epochs: "
    f"{len(passed)}"
)


summary.append("")
summary.append("=" * 80)
summary.append("DROP REASON FREQUENCY")
summary.append("=" * 80)

all_reason_counts = {}

for value in df["drop_reasons"].dropna():

    if not value:
        continue

    for item in value.split(";"):

        if ":" not in item:
            continue

        reason, count = item.rsplit(
            ":",
            1
        )

        try:

            count = int(count)

        except Exception:

            continue

        all_reason_counts[reason] = (
            all_reason_counts.get(
                reason,
                0
            ) + count
        )


if all_reason_counts:

    for reason, count in sorted(
        all_reason_counts.items(),
        key=lambda x: -x[1]
    ):

        summary.append(
            f"{reason:<30} {count}"
        )

else:

    summary.append("NONE")


summary.append("")
summary.append("=" * 80)
summary.append("IMPORTANT")
summary.append("=" * 80)
summary.append(
    "This script ONLY reads files."
)
summary.append(
    "No raw SET files were modified."
)
summary.append(
    "No FDT files were modified."
)
summary.append(
    "No preprocessed FIF files were modified."
)
summary.append(
    "No epoch FIF files were modified."
)
summary.append(
    "No channels were removed."
)
summary.append(
    "No samples were deleted."
)
summary.append(
    "No interpolation was performed."
)

summary_text = "\n".join(summary)

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(summary_text)


# =====================================================================
# FINAL TERMINAL OUTPUT
# =====================================================================

print()
print()
print("=" * 80)
print("FULL EPOCH DIAGNOSIS COMPLETE")
print("=" * 80)

print()
print("STATUS COUNTS")
print(status_counts)

print()
print(
    "Preprocessed runs:",
    len(preprocessed_files)
)

print(
    "Epoch files:",
    len(epoch_files)
)

print(
    "Missing epoch runs:",
    len(missing_epoch_keys)
)

print(
    "Zero-epoch runs:",
    len(zero_epoch)
)

print(
    "Low-epoch runs:",
    len(low_epoch)
)

print(
    "Failed runs:",
    len(failed)
)

print(
    "Passed runs:",
    len(passed)
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