import os
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

PROJECT = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

FINAL_DIR = os.path.join(
    PROJECT,
    "qc",
    "final_qc"
)

CHANNEL_FILE = os.path.join(
    FINAL_DIR,
    "final_channel_qc_83runs.csv"
)

RUN_FILE = os.path.join(
    FINAL_DIR,
    "final_run_qc_83runs.csv"
)

OUTPUT_CSV = os.path.join(
    FINAL_DIR,
    "qc_decision_report_83runs.csv"
)

OUTPUT_SUMMARY = os.path.join(
    FINAL_DIR,
    "qc_decision_report_83runs_summary.txt"
)

# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("QC DECISION REPORT - 83 RUNS")
print("=" * 75)

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(CHANNEL_FILE):
    raise FileNotFoundError(
        f"\nChannel QC file not found:\n{CHANNEL_FILE}"
    )

if not os.path.exists(RUN_FILE):
    raise FileNotFoundError(
        f"\nRun QC file not found:\n{RUN_FILE}"
    )

# ============================================================
# LOAD
# ============================================================

channel_df = pd.read_csv(
    CHANNEL_FILE
)

run_df = pd.read_csv(
    RUN_FILE
)

print(
    f"Channel QC records: {len(channel_df)}"
)

print(
    f"Run QC records:     {len(run_df)}"
)

print("\nChannel columns:")
print(list(channel_df.columns))

print("\nRun columns:")
print(list(run_df.columns))

# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

channel_df.columns = [
    str(c).strip().lower()
    for c in channel_df.columns
]

run_df.columns = [
    str(c).strip().lower()
    for c in run_df.columns
]

# ============================================================
# FINAL STATUS COLUMN
# ============================================================

if "final_status" not in channel_df.columns:

    raise ValueError(
        "\nfinal_status column is missing from channel QC."
    )

channel_df["final_status"] = (
    channel_df["final_status"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_channel = [
    "subject",
    "run",
    "channel",
    "final_status"
]

required_run = [
    "subject",
    "run",
    "bad_channels",
    "review_channels",
    "pass_channels"
]

for col in required_channel:

    if col not in channel_df.columns:

        raise ValueError(
            f"\nMissing channel column: {col}\n"
            f"Available:\n{list(channel_df.columns)}"
        )

for col in required_run:

    if col not in run_df.columns:

        raise ValueError(
            f"\nMissing run column: {col}\n"
            f"Available:\n{list(run_df.columns)}"
        )

# ============================================================
# NORMALIZE SUBJECT / RUN
# ============================================================

channel_df["subject"] = (
    channel_df["subject"]
    .astype(str)
    .str.strip()
)

run_df["subject"] = (
    run_df["subject"]
    .astype(str)
    .str.strip()
)

channel_df["run"] = pd.to_numeric(
    channel_df["run"],
    errors="coerce"
)

run_df["run"] = pd.to_numeric(
    run_df["run"],
    errors="coerce"
)

channel_df = channel_df.dropna(
    subset=["subject", "run"]
)

run_df = run_df.dropna(
    subset=["subject", "run"]
)

channel_df["run"] = (
    channel_df["run"]
    .astype(int)
)

run_df["run"] = (
    run_df["run"]
    .astype(int)
)

# ============================================================
# RE-CALCULATE CHANNEL COUNTS
# ============================================================

channel_counts = (
    channel_df
    .groupby(
        ["subject", "run", "final_status"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

for status in [
    "PASS",
    "REVIEW",
    "BAD"
]:

    if status not in channel_counts.columns:

        channel_counts[status] = 0

channel_counts = (
    channel_counts
    .reset_index()
)

channel_counts = channel_counts.rename(
    columns={
        "PASS": "calculated_pass_channels",
        "REVIEW": "calculated_review_channels",
        "BAD": "calculated_bad_channels"
    }
)

# ============================================================
# MERGE CALCULATED COUNTS
# ============================================================

report = run_df.copy()

report = report.merge(
    channel_counts[
        [
            "subject",
            "run",
            "calculated_pass_channels",
            "calculated_review_channels",
            "calculated_bad_channels"
        ]
    ],
    on=[
        "subject",
        "run"
    ],
    how="left"
)

# ============================================================
# FILL COUNTS
# ============================================================

count_columns = [
    "calculated_pass_channels",
    "calculated_review_channels",
    "calculated_bad_channels",
    "bad_channels",
    "review_channels",
    "pass_channels"
]

for col in count_columns:

    report[col] = pd.to_numeric(
        report[col],
        errors="coerce"
    ).fillna(0).astype(int)

# ============================================================
# CHECK CONSISTENCY
# ============================================================

report["bad_count_match"] = (
    report["bad_channels"]
    ==
    report["calculated_bad_channels"]
)

report["review_count_match"] = (
    report["review_channels"]
    ==
    report["calculated_review_channels"]
)

report["pass_count_match"] = (
    report["pass_channels"]
    ==
    report["calculated_pass_channels"]
)

report["counts_consistent"] = (
    report["bad_count_match"]
    &
    report["review_count_match"]
    &
    report["pass_count_match"]
)

# ============================================================
# EEG CHANNEL COUNT
# ============================================================

if "eeg_channels_x" in report.columns:

    report["eeg_channels"] = pd.to_numeric(
        report["eeg_channels_x"],
        errors="coerce"
    )

elif "eeg_channels" in report.columns:

    report["eeg_channels"] = pd.to_numeric(
        report["eeg_channels"],
        errors="coerce"
    )

else:

    report["eeg_channels"] = 69

report["eeg_channels"] = (
    report["eeg_channels"]
    .fillna(69)
    .astype(int)
)

# ============================================================
# DECISION ENGINE
# ============================================================

def decision(row):

    bad = int(row["bad_channels"])
    review = int(row["review_channels"])
    eeg = int(row["eeg_channels"])

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if eeg > 0:

        if bad >= max(
            20,
            int(0.50 * eeg)
        ):

            return (
                "EXCLUDE_RUN",
                ">=50% of EEG channels are BAD"
            )

    # --------------------------------------------------------
    # MANY BAD CHANNELS
    # --------------------------------------------------------

    if bad >= 5:

        return (
            "RUN_REVIEW",
            "5 or more BAD EEG channels"
        )

    # --------------------------------------------------------
    # 2-4 BAD CHANNELS
    # --------------------------------------------------------

    if bad >= 2:

        return (
            "CHANNEL_REVIEW",
            "2-4 BAD EEG channels"
        )

    # --------------------------------------------------------
    # ONE BAD CHANNEL
    # --------------------------------------------------------

    if bad == 1:

        if review >= 15:

            return (
                "RUN_REVIEW",
                "1 BAD channel + >=15 REVIEW channels"
            )

        return (
            "INTERPOLATE_CHANNEL",
            "1 BAD EEG channel"
        )

    # --------------------------------------------------------
    # MANY REVIEW CHANNELS
    # --------------------------------------------------------

    if review >= 20:

        return (
            "RUN_REVIEW",
            ">=20 REVIEW EEG channels"
        )

    # --------------------------------------------------------
    # MODERATE REVIEW
    # --------------------------------------------------------

    if review >= 10:

        return (
            "CHANNEL_REVIEW",
            "10-19 REVIEW EEG channels"
        )

    # --------------------------------------------------------
    # FEW REVIEW
    # --------------------------------------------------------

    if review > 0:

        return (
            "KEEP_REVIEW",
            "1-9 REVIEW EEG channels"
        )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    return (
        "KEEP",
        "No BAD or REVIEW EEG channels"
    )

# ============================================================
# APPLY DECISION
# ============================================================

decisions = report.apply(
    decision,
    axis=1,
    result_type="expand"
)

decisions.columns = [
    "recommended_action",
    "decision_reason"
]

report = pd.concat(
    [
        report,
        decisions
    ],
    axis=1
)

# ============================================================
# MANUAL REVIEW FLAG
# ============================================================

report["needs_manual_review"] = (
    report["recommended_action"]
    .isin(
        [
            "RUN_REVIEW",
            "CHANNEL_REVIEW",
            "KEEP_REVIEW"
        ]
    )
)

report["critical_run"] = (
    report["recommended_action"]
    ==
    "EXCLUDE_RUN"
)

# ============================================================
# SORT
# ============================================================

report = report.sort_values(
    [
        "subject",
        "run"
    ]
).reset_index(
    drop=True
)

# ============================================================
# SAVE COMPLETE REPORT
# ============================================================

report.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# COUNTS
# ============================================================

action_counts = (
    report[
        "recommended_action"
    ]
    .value_counts()
)

# ============================================================
# PRINT FINAL ACTIONS
# ============================================================

print("\n")
print("=" * 75)
print("RECOMMENDED ACTIONS")
print("=" * 75)

print(
    action_counts.to_string()
)

# ============================================================
# RUNS REQUIRING REVIEW
# ============================================================

print("\n")
print("=" * 75)
print("RUNS REQUIRING MANUAL REVIEW")
print("=" * 75)

review_report = report[
    report["needs_manual_review"]
][
    [
        "subject",
        "run",
        "recommended_action",
        "bad_channels",
        "review_channels",
        "pass_channels",
        "final_run_status",
        "decision_reason"
    ]
]

if len(review_report) > 0:

    print(
        review_report.to_string(
            index=False
        )
    )

else:

    print("NONE")

# ============================================================
# EXCLUDE CANDIDATES
# ============================================================

print("\n")
print("=" * 75)
print("EXCLUSION CANDIDATES")
print("=" * 75)

exclude_report = report[
    report["recommended_action"]
    ==
    "EXCLUDE_RUN"
][
    [
        "subject",
        "run",
        "bad_channels",
        "review_channels",
        "final_run_status",
        "decision_reason"
    ]
]

if len(exclude_report) > 0:

    print(
        exclude_report.to_string(
            index=False
        )
    )

else:

    print("NONE")

# ============================================================
# INTERPOLATION CANDIDATES
# ============================================================

print("\n")
print("=" * 75)
print("CHANNEL INTERPOLATION CANDIDATES")
print("=" * 75)

interpolate_report = report[
    report["recommended_action"]
    ==
    "INTERPOLATE_CHANNEL"
][
    [
        "subject",
        "run",
        "bad_channels",
        "review_channels"
    ]
]

if len(interpolate_report) > 0:

    print(
        interpolate_report.to_string(
            index=False
        )
    )

else:

    print("NONE")

# ============================================================
# INCONSISTENT COUNTS
# ============================================================

print("\n")
print("=" * 75)
print("COUNT CONSISTENCY CHECK")
print("=" * 75)

inconsistent = report[
    ~report["counts_consistent"]
]

if len(inconsistent) > 0:

    print(
        "WARNING: Count mismatch detected."
    )

    print(
        inconsistent[
            [
                "subject",
                "run",
                "bad_channels",
                "calculated_bad_channels",
                "review_channels",
                "calculated_review_channels",
                "pass_channels",
                "calculated_pass_channels"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "ALL RUNS CONSISTENT"
    )

# ============================================================
# TEXT SUMMARY
# ============================================================

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 75 + "\n"
    )

    f.write(
        "QC DECISION REPORT - 83 RUNS\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Total runs: {len(report)}\n"
    )

    f.write(
        f"Total channel records: {len(channel_df)}\n\n"
    )

    f.write(
        "RECOMMENDED ACTION COUNTS\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        action_counts.to_string()
    )

    f.write("\n\n")

    f.write(
        "RUNS REQUIRING MANUAL REVIEW\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    if len(review_report) > 0:

        f.write(
            review_report.to_string(
                index=False
            )
        )

    else:

        f.write("NONE")

    f.write("\n\n")

    f.write(
        "EXCLUSION CANDIDATES\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    if len(exclude_report) > 0:

        f.write(
            exclude_report.to_string(
                index=False
            )
        )

    else:

        f.write("NONE")

    f.write("\n\n")

    f.write(
        "CHANNEL INTERPOLATION CANDIDATES\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    if len(interpolate_report) > 0:

        f.write(
            interpolate_report.to_string(
                index=False
            )
        )

    else:

        f.write("NONE")

    f.write("\n\n")

    f.write(
        "COUNT CONSISTENCY\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    if len(inconsistent) == 0:

        f.write(
            "ALL RUNS CONSISTENT\n"
        )

    else:

        f.write(
            "COUNT MISMATCH DETECTED\n"
        )

    f.write("\n\n")

    f.write(
        "IMPORTANT\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        "This script ONLY analyzes QC CSV files.\n"
    )

    f.write(
        "RAW SET/FDT DATA WAS NOT MODIFIED.\n"
    )

    f.write(
        "NO CHANNELS WERE REMOVED.\n"
    )

    f.write(
        "NO SAMPLES WERE DELETED.\n"
    )

    f.write(
        "NO INTERPOLATION WAS PERFORMED.\n"
    )

# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 75)
print("COMPLETE")
print("=" * 75)

print("\nSaved:")
print(
    OUTPUT_CSV
)

print(
    OUTPUT_SUMMARY
)

print("\nRAW DATA WAS NOT MODIFIED.")
print("NO SET FILES WERE MODIFIED.")
print("NO FDT FILES WERE MODIFIED.")
print("NO CHANNELS WERE REMOVED.")
print("NO SAMPLES WERE DELETED.")
print("NO INTERPOLATION WAS PERFORMED.")