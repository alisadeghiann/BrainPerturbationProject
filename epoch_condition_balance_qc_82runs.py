import os
import glob
import pandas as pd
import numpy as np
import mne

# ============================================================
# EPOCH CONDITION BALANCE QC - 82 RUNS
# READ-ONLY
# NO DATA MODIFICATION
# ============================================================

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

EPOCH_DIR = os.path.join(BASE, "epochs_clean")

FINAL_SELECTION = os.path.join(
    EPOCH_DIR,
    "logs",
    "final_selection",
    "epoch_keep_recommendations_82runs.csv"
)

OUT_DIR = os.path.join(
    EPOCH_DIR,
    "logs",
    "condition_balance_qc"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(
    OUT_DIR,
    "epoch_condition_balance_qc_82runs.csv"
)

OUT_CONDITION_CSV = os.path.join(
    OUT_DIR,
    "epoch_condition_counts_82runs.csv"
)

OUT_SUBJECT_CSV = os.path.join(
    OUT_DIR,
    "epoch_subject_condition_balance_82runs.csv"
)

SUMMARY = os.path.join(
    OUT_DIR,
    "epoch_condition_balance_qc_82runs_summary.txt"
)

print("=" * 80)
print("EPOCH CONDITION BALANCE QC - 82 RUNS")
print("=" * 80)

# ------------------------------------------------------------
# CHECK INPUT
# ------------------------------------------------------------

if not os.path.exists(FINAL_SELECTION):
    raise FileNotFoundError(
        f"Final selection file not found:\n{FINAL_SELECTION}"
    )

selection = pd.read_csv(FINAL_SELECTION)

print("\nSelection records:", len(selection))

# ------------------------------------------------------------
# FIND COLUMNS
# ------------------------------------------------------------

file_col = None

for c in ["file", "epoch_file", "filename"]:
    if c in selection.columns:
        file_col = c
        break

if file_col is None:
    raise ValueError(
        "File column not found.\n"
        f"Available: {list(selection.columns)}"
    )

recommendation_col = None

for c in ["recommendation", "final_recommendation"]:
    if c in selection.columns:
        recommendation_col = c
        break

if recommendation_col is None:
    raise ValueError(
        "Recommendation column not found.\n"
        f"Available: {list(selection.columns)}"
    )

# ------------------------------------------------------------
# EPOCH FILES
# ------------------------------------------------------------

epoch_files = sorted(
    glob.glob(
        os.path.join(
            EPOCH_DIR,
            "*_clean_epo.fif"
        )
    )
)

print("Epoch files found:", len(epoch_files))

if len(epoch_files) != 82:
    print(
        "\nWARNING: Expected 82 epoch files "
        f"but found {len(epoch_files)}."
    )

# ------------------------------------------------------------
# HELPER
# ------------------------------------------------------------

def decode_event_name(name):
    """
    Convert possible numpy / byte / array representations
    into a readable string.
    """

    if isinstance(name, bytes):
        try:
            return name.decode("utf-8")
        except Exception:
            return str(name)

    if isinstance(name, np.ndarray):
        try:
            arr = np.asarray(name).flatten()

            if arr.dtype.kind in ["i", "u"]:
                return "".join(
                    chr(int(x)) for x in arr
                )

            if arr.dtype.kind == "S":
                return b"".join(
                    arr.tolist()
                ).decode("utf-8", errors="replace")

        except Exception:
            pass

    return str(name)


def get_condition_from_epoch(epochs, index):
    """
    Determine event/condition associated with an epoch.
    """

    try:
        event_id = epochs.event_id

        inverse = {
            int(v): k
            for k, v in event_id.items()
        }

        code = int(
            epochs.events[index, 2]
        )

        if code in inverse:
            return decode_event_name(
                inverse[code]
            )

    except Exception:
        pass

    return "UNKNOWN"


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

run_rows = []
condition_rows = []
subject_condition_rows = []

total_epochs = 0
total_keep = 0
total_review = 0
total_exclude = 0

# ------------------------------------------------------------
# PROCESS
# ------------------------------------------------------------

for i, path in enumerate(epoch_files, start=1):

    filename = os.path.basename(path)

    print("\n" + "=" * 80)
    print(f"[{i}/{len(epoch_files)}] {filename}")
    print("=" * 80)

    try:

        epochs = mne.read_epochs(
            path,
            preload=False,
            verbose=False
        )

        n_epochs = len(epochs)

        print("Epochs:", n_epochs)
        print("Channels:", len(epochs.ch_names))
        print("Sampling rate:", epochs.info["sfreq"])

        # ----------------------------------------------------
        # PARSE SUBJECT / RUN
        # ----------------------------------------------------

        subject = "UNKNOWN"
        run = "UNKNOWN"

        parts = filename.split("_")

        for p in parts:
            if p.startswith("sub-"):
                subject = p

        for p in parts:
            if p.startswith("run-"):
                run = p.replace(
                    "run-", ""
                )

        # ----------------------------------------------------
        # EVENT COUNTS
        # ----------------------------------------------------

        counts = {}

        for ep_idx in range(n_epochs):

            condition = get_condition_from_epoch(
                epochs,
                ep_idx
            )

            counts[condition] = (
                counts.get(condition, 0) + 1
            )

        # ----------------------------------------------------
        # SELECTION INFORMATION
        # ----------------------------------------------------

        sel = selection[
            selection[file_col] == filename
        ]

        # If filename matching fails, try contains
        if len(sel) == 0:

            sel = selection[
                selection[file_col]
                .astype(str)
                .str.contains(
                    filename.replace(
                        "_clean_epo.fif",
                        ""
                    ),
                    regex=False
                )
            ]

        # ----------------------------------------------------
        # EXPECTED RECOMMENDATION COUNTS
        # ----------------------------------------------------

        keep_count = 0
        brief_count = 0
        moderate_count = 0
        exclude_count = 0

        if len(sel) > 0:

            rec = (
                sel[recommendation_col]
                .astype(str)
                .str.upper()
            )

            keep_count = int(
                (rec == "KEEP").sum()
            )

            brief_count = int(
                (rec == "KEEP_REVIEW").sum()
            )

            moderate_count = int(
                (rec == "REVIEW").sum()
            )

            exclude_count = int(
                (rec == "EXCLUDE_RECOMMENDED").sum()
            )

        total_epochs += n_epochs
        total_keep += keep_count
        total_review += (
            brief_count +
            moderate_count
        )
        total_exclude += exclude_count

        # ----------------------------------------------------
        # CONDITION ROWS
        # ----------------------------------------------------

        for condition, count in sorted(
            counts.items()
        ):

            condition_rows.append({
                "subject": subject,
                "run": run,
                "file": filename,
                "condition": condition,
                "epoch_count": count,
                "percent_of_run":
                    100 * count / n_epochs
                    if n_epochs else 0
            })

        # ----------------------------------------------------
        # BALANCE METRICS
        # ----------------------------------------------------

        condition_values = list(
            counts.values()
        )

        if len(condition_values) > 0:

            min_count = min(
                condition_values
            )

            max_count = max(
                condition_values
            )

            mean_count = np.mean(
                condition_values
            )

            balance_ratio = (
                min_count / max_count
                if max_count > 0
                else 0
            )

        else:

            min_count = 0
            max_count = 0
            mean_count = 0
            balance_ratio = 0

        # ----------------------------------------------------
        # RUN STATUS
        # ----------------------------------------------------

        if n_epochs == 0:

            balance_status = "BAD"

        elif len(counts) < 2:

            balance_status = "REVIEW"

        elif balance_ratio >= 0.70:

            balance_status = "PASS"

        elif balance_ratio >= 0.40:

            balance_status = "REVIEW"

        else:

            balance_status = "BAD"

        run_rows.append({

            "subject": subject,
            "run": run,
            "file": filename,

            "total_epochs": n_epochs,

            "n_conditions": len(counts),

            "min_condition_count": min_count,
            "max_condition_count": max_count,
            "mean_condition_count":
                mean_count,

            "balance_ratio":
                balance_ratio,

            "selection_KEEP":
                keep_count,

            "selection_KEEP_REVIEW":
                brief_count,

            "selection_REVIEW":
                moderate_count,

            "selection_EXCLUDE":
                exclude_count,

            "balance_status":
                balance_status
        })

        # ----------------------------------------------------
        # SUBJECT CONDITION
        # ----------------------------------------------------

        for condition, count in sorted(
            counts.items()
        ):

            subject_condition_rows.append({

                "subject": subject,
                "condition": condition,
                "run": run,
                "epoch_count": count

            })

        print("\nCONDITIONS")

        for condition, count in sorted(
            counts.items()
        ):

            print(
                f"{condition:25s} {count:5d}"
            )

        print(
            f"\nBalance ratio: "
            f"{balance_ratio:.3f}"
        )

        print(
            f"STATUS: {balance_status}"
        )

    except Exception as e:

        print("\nERROR:")
        print(str(e))

        run_rows.append({

            "subject": subject,
            "run": run,
            "file": filename,

            "total_epochs": 0,

            "n_conditions": 0,

            "min_condition_count": 0,
            "max_condition_count": 0,
            "mean_condition_count": 0,

            "balance_ratio": 0,

            "selection_KEEP": 0,
            "selection_KEEP_REVIEW": 0,
            "selection_REVIEW": 0,
            "selection_EXCLUDE": 0,

            "balance_status": "ERROR"
        })


# ------------------------------------------------------------
# DATAFRAMES
# ------------------------------------------------------------

run_df = pd.DataFrame(
    run_rows
)

condition_df = pd.DataFrame(
    condition_rows
)

subject_condition_df = pd.DataFrame(
    subject_condition_rows
)

# ------------------------------------------------------------
# SUBJECT AGGREGATION
# ------------------------------------------------------------

if len(subject_condition_df) > 0:

    subject_summary = (
        subject_condition_df
        .groupby(
            ["subject", "condition"],
            as_index=False
        )["epoch_count"]
        .sum()
    )

else:

    subject_summary = pd.DataFrame()

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

run_df.to_csv(
    OUT_CSV,
    index=False
)

condition_df.to_csv(
    OUT_CONDITION_CSV,
    index=False
)

subject_summary.to_csv(
    OUT_SUBJECT_CSV,
    index=False
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

status_counts = (
    run_df["balance_status"]
    .value_counts()
)

with open(
    SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "EPOCH CONDITION BALANCE QC - 82 RUNS\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Epoch files: {len(epoch_files)}\n"
    )

    f.write(
        f"Total epochs: {total_epochs}\n\n"
    )

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "RUN BALANCE STATUS\n"
    )

    f.write(
        "=" * 80 + "\n"
    )

    for status, count in status_counts.items():

        f.write(
            f"{status:10s} {count:5d}\n"
        )

    f.write("\n")

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "SELECTION COUNTS\n"
    )

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        f"KEEP:             {total_keep}\n"
    )

    f.write(
        f"KEEP_REVIEW:      {total_review}\n"
    )

    f.write(
        f"EXCLUDE:          {total_exclude}\n"
    )

    f.write("\n")

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "RUNS REQUIRING REVIEW\n"
    )

    f.write(
        "=" * 80 + "\n"
    )

    review_df = run_df[
        run_df["balance_status"].isin(
            ["REVIEW", "BAD", "ERROR"]
        )
    ]

    if len(review_df) == 0:

        f.write("NONE\n")

    else:

        f.write(
            review_df.to_string(
                index=False
            )
        )

        f.write("\n")

    f.write("\n")

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "IMPORTANT\n"
    )

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "This script ONLY reads existing epoch files.\n"
    )

    f.write(
        "NO EPOCH FILE WAS MODIFIED.\n"
    )

    f.write(
        "NO EPOCH WAS DELETED.\n"
    )

    f.write(
        "NO RAW DATA WAS MODIFIED.\n"
    )

# ------------------------------------------------------------
# FINAL CONSOLE
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("CONDITION BALANCE QC COMPLETE")
print("=" * 80)

print("\nSTATUS COUNTS")
print(status_counts)

print("\nTOTAL EPOCHS:")
print(total_epochs)

print("\nSELECTION COUNTS")
print("KEEP:", total_keep)
print("KEEP_REVIEW:", total_review)
print("EXCLUDE:", total_exclude)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUT_CSV)
print(OUT_CONDITION_CSV)
print(OUT_SUBJECT_CSV)
print(SUMMARY)

print("\n" + "=" * 80)
print("NO DATA WAS MODIFIED.")
print("=" * 80)