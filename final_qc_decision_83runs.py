from pathlib import Path
import pandas as pd
import numpy as np
import re

# ============================================================
# FINAL QC DECISION ENGINE - FIXED
# Brain Perturbation Project
# ============================================================

PROJECT = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

QC = PROJECT / "qc"

CHANNEL_QC = QC / "automated_channel_qc_83runs.csv"
RUN_QC = QC / "automated_run_qc_83runs.csv"

OUT = QC / "final_qc"
OUT.mkdir(exist_ok=True)

CHANNEL_OUT = OUT / "final_channel_qc_83runs.csv"
RUN_OUT = OUT / "final_run_qc_83runs.csv"
SUMMARY_OUT = OUT / "final_qc_summary.txt"

print("=" * 75)
print("FINAL QC DECISION ENGINE - FIXED VERSION")
print("=" * 75)

# ============================================================
# LOAD FILES
# ============================================================

if not CHANNEL_QC.exists():
    raise FileNotFoundError(
        f"\nMissing channel QC file:\n{CHANNEL_QC}"
    )

if not RUN_QC.exists():
    raise FileNotFoundError(
        f"\nMissing run QC file:\n{RUN_QC}"
    )

channels = pd.read_csv(CHANNEL_QC)
runs = pd.read_csv(RUN_QC)

print(f"\nChannel QC records: {len(channels)}")
print(f"Run QC records:     {len(runs)}")

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

channels.columns = [
    str(c).strip()
    for c in channels.columns
]

runs.columns = [
    str(c).strip()
    for c in runs.columns
]

# ============================================================
# EXTRACT RUN NUMBER FROM FILE NAME
# ============================================================

def extract_run(value):

    if pd.isna(value):
        return np.nan

    text = str(value)

    match = re.search(
        r"_run-(\d+)_",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    # fallback
    match = re.search(
        r"run[-_]?(\d+)",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return np.nan


# ============================================================
# CHANNEL FILE
# ============================================================

if "run" not in channels.columns:

    if "file" not in channels.columns:
        raise ValueError(
            "\nChannel QC has neither 'run' nor 'file'.\n"
            f"Available columns:\n{list(channels.columns)}"
        )

    channels["run"] = channels["file"].apply(
        extract_run
    )

# ============================================================
# RUN FILE
# ============================================================

if "run" not in runs.columns:

    if "file" not in runs.columns:
        raise ValueError(
            "\nRun QC has neither 'run' nor 'file'.\n"
            f"Available columns:\n{list(runs.columns)}"
        )

    runs["run"] = runs["file"].apply(
        extract_run
    )

# ============================================================
# CHECK RUN EXTRACTION
# ============================================================

bad_channel_runs = channels["run"].isna().sum()
bad_run_runs = runs["run"].isna().sum()

print(
    f"\nChannel rows with unknown run: {bad_channel_runs}"
)

print(
    f"Run rows with unknown run:     {bad_run_runs}"
)

if bad_channel_runs > 0:

    print("\nWARNING: Some channel rows have no run number.")

if bad_run_runs > 0:

    print("\nWARNING: Some run rows have no run number.")

# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_channel_cols = [
    "std",
    "median_eeg_std",
    "std_ratio",
    "mean",
    "min",
    "max",
    "clip_percent",
    "high_amplitude_percent",
    "channel_index",
]

for col in numeric_channel_cols:

    if col in channels.columns:

        channels[col] = pd.to_numeric(
            channels[col],
            errors="coerce"
        )

channels["run"] = pd.to_numeric(
    channels["run"],
    errors="coerce"
)

runs["run"] = pd.to_numeric(
    runs["run"],
    errors="coerce"
)

# ============================================================
# EOG CHANNELS
# ============================================================

EOG_NAMES = {
    "LEYE",
    "REYE",
    "EOG",
    "HEOG",
    "VEOG",
}

channels["channel_clean"] = (
    channels["channel"]
    .astype(str)
    .str.strip()
    .str.upper()
)

channels["is_eog"] = (
    channels["channel_clean"]
    .isin(EOG_NAMES)
)

eeg = channels.loc[
    ~channels["is_eog"]
].copy()

print(
    f"\nEEG channel records: {len(eeg)}"
)

print(
    f"EOG records ignored: "
    f"{channels['is_eog'].sum()}"
)

# ============================================================
# FINAL CHANNEL DECISION
# ============================================================

def channel_decision(row):

    ratio = row["std_ratio"]
    clip = row["clip_percent"]
    amp = row["high_amplitude_percent"]

    reasons = []

    # --------------------------------------------------------
    # CLIPPING
    # --------------------------------------------------------

    if pd.notna(clip):

        if clip >= 5:

            reasons.append(
                "SEVERE_CLIPPING"
            )

        elif clip >= 1:

            reasons.append(
                "CLIPPING_REVIEW"
            )

    # --------------------------------------------------------
    # AMPLITUDE
    # --------------------------------------------------------

    if pd.notna(amp):

        if amp >= 10:

            reasons.append(
                "SEVERE_AMPLITUDE"
            )

        elif amp >= 5:

            reasons.append(
                "AMPLITUDE_REVIEW"
            )

    # --------------------------------------------------------
    # STD RATIO
    # --------------------------------------------------------

    if pd.notna(ratio):

        if ratio >= 5:

            reasons.append(
                "EXTREME_STD_RATIO"
            )

        elif ratio >= 3:

            reasons.append(
                "HIGH_STD_RATIO"
            )

        elif ratio >= 2:

            reasons.append(
                "ELEVATED_STD"
            )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    if (
        (pd.notna(ratio) and ratio >= 5)
        or
        (pd.notna(clip) and clip >= 5)
        or
        (pd.notna(amp) and amp >= 10)
    ):

        decision = "BAD"

    elif (
        (pd.notna(ratio) and ratio >= 2)
        or
        (pd.notna(clip) and clip >= 1)
        or
        (pd.notna(amp) and amp >= 5)
    ):

        decision = "REVIEW"

    else:

        decision = "PASS"

    return pd.Series(
        [decision, ";".join(reasons)]
    )


# ============================================================
# APPLY CHANNEL QC
# ============================================================

eeg[
    [
        "final_status",
        "final_reasons"
    ]
] = eeg.apply(
    channel_decision,
    axis=1
)

# ============================================================
# ACTION
# ============================================================

eeg["action"] = np.select(

    [
        eeg["final_status"] == "BAD",
        eeg["final_status"] == "REVIEW",
        eeg["final_status"] == "PASS",
    ],

    [
        "REMOVE_OR_INTERPOLATE_AFTER_MANUAL_REVIEW",
        "KEEP_FOR_NOW",
        "KEEP",
    ],

    default="REVIEW"
)

# ============================================================
# SAVE CHANNEL QC
# ============================================================

channel_columns = [

    "subject",
    "run",
    "file",
    "channel",
    "channel_index",
    "std",
    "median_eeg_std",
    "std_ratio",
    "mean",
    "min",
    "max",
    "clip_percent",
    "high_amplitude_percent",
    "final_status",
    "final_reasons",
    "action",
]

channel_columns = [
    c
    for c in channel_columns
    if c in eeg.columns
]

eeg[channel_columns].to_csv(
    CHANNEL_OUT,
    index=False
)

# ============================================================
# RUN-LEVEL CHANNEL SUMMARY
# ============================================================

run_channel_summary = (

    eeg
    .groupby(
        ["subject", "run"],
        dropna=False
    )
    .agg(

        eeg_channels=(
            "channel",
            "count"
        ),

        bad_channels=(
            "final_status",
            lambda x:
                (x == "BAD").sum()
        ),

        review_channels=(
            "final_status",
            lambda x:
                (x == "REVIEW").sum()
        ),

        pass_channels=(
            "final_status",
            lambda x:
                (x == "PASS").sum()
        ),

    )

    .reset_index()
)

# ============================================================
# MERGE WITH ORIGINAL RUN QC
# ============================================================

original_run_columns = [

    "subject",
    "run",
    "file",
    "channels_total",
    "eeg_channels",
    "srate",
    "duration_seconds",
    "median_eeg_std",
]

original_run_columns = [
    c
    for c in original_run_columns
    if c in runs.columns
]

original_runs = runs[
    original_run_columns
].copy()

run_summary = original_runs.merge(

    run_channel_summary,

    on=[
        "subject",
        "run"
    ],

    how="left"
)

# ============================================================
# FILL MISSING COUNTS
# ============================================================

for col in [
    "bad_channels",
    "review_channels",
    "pass_channels",
    "eeg_channels_y",
]:

    if col in run_summary.columns:

        run_summary[col] = (
            run_summary[col]
            .fillna(0)
        )

# Avoid duplicated eeg_channels name if merge created one
if "eeg_channels_y" in run_summary.columns:

    run_summary["eeg_channels_checked"] = (
        run_summary["eeg_channels_y"]
    )

    run_summary.drop(
        columns=["eeg_channels_y"],
        inplace=True
    )

# ============================================================
# RUN-LEVEL DECISION
# ============================================================

def run_decision(row):

    bad = row["bad_channels"]
    review = row["review_channels"]

    # --------------------------------------------------------
    # IMPORTANT:
    # A run is NOT automatically rejected because it contains
    # temporal/global artifacts.
    #
    # This is channel-level QC only.
    # --------------------------------------------------------

    if pd.isna(bad):

        return "REVIEW"

    if bad >= 2:

        return "RUN_REVIEW"

    if bad == 1:

        return "CHANNEL_REVIEW"

    if review >= 10:

        return "RUN_REVIEW"

    return "PASS"


run_summary[
    "final_run_status"
] = run_summary.apply(
    run_decision,
    axis=1
)

# ============================================================
# SAVE RUN QC
# ============================================================

run_summary.to_csv(
    RUN_OUT,
    index=False
)

# ============================================================
# COUNTS
# ============================================================

channel_counts = (
    eeg["final_status"]
    .value_counts()
)

run_counts = (
    run_summary["final_run_status"]
    .value_counts()
)

# ============================================================
# BAD CHANNEL TABLE
# ============================================================

bad_rows = eeg[
    eeg["final_status"] == "BAD"
].copy()

review_rows = eeg[
    eeg["final_status"] == "REVIEW"
].copy()

review_runs = run_summary[
    run_summary["final_run_status"] != "PASS"
].copy()

# ============================================================
# SUMMARY FILE
# ============================================================

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 75 + "\n"
    )

    f.write(
        "FINAL QC SUMMARY\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Channel QC records: {len(eeg)}\n"
    )

    f.write(
        f"Run QC records: {len(run_summary)}\n"
    )

    f.write(
        f"Subjects: {eeg['subject'].nunique()}\n\n"
    )

    # --------------------------------------------------------
    # CHANNEL STATUS
    # --------------------------------------------------------

    f.write(
        "CHANNEL STATUS\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    for status, count in channel_counts.items():

        f.write(
            f"{status:10s} {count:6d}\n"
        )

    # --------------------------------------------------------
    # RUN STATUS
    # --------------------------------------------------------

    f.write(
        "\nRUN STATUS\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    for status, count in run_counts.items():

        f.write(
            f"{status:25s} {count:6d}\n"
        )

    # --------------------------------------------------------
    # BAD CHANNELS
    # --------------------------------------------------------

    f.write(
        "\n" + "=" * 75 + "\n"
    )

    f.write(
        "BAD CHANNELS\n"
    )

    f.write(
        "=" * 75 + "\n"
    )

    if len(bad_rows) == 0:

        f.write(
            "NONE\n"
        )

    else:

        for _, row in bad_rows.iterrows():

            f.write(
                f"{row['subject']} "
                f"run-{int(row['run'])} "
                f"{row['channel']} "
                f"ratio={row['std_ratio']:.3f} "
                f"clip={row['clip_percent']:.3f}% "
                f"amp={row['high_amplitude_percent']:.3f}% "
                f"{row['final_reasons']}\n"
            )

    # --------------------------------------------------------
    # REVIEW CHANNELS
    # --------------------------------------------------------

    f.write(
        "\n" + "=" * 75 + "\n"
    )

    f.write(
        "REVIEW CHANNELS\n"
    )

    f.write(
        "=" * 75 + "\n"
    )

    f.write(
        f"Total REVIEW channels: "
        f"{len(review_rows)}\n"
    )

    # --------------------------------------------------------
    # REVIEW RUNS
    # --------------------------------------------------------

    f.write(
        "\n" + "=" * 75 + "\n"
    )

    f.write(
        "RUNS REQUIRING REVIEW\n"
    )

    f.write(
        "=" * 75 + "\n"
    )

    if len(review_runs) == 0:

        f.write(
            "NONE\n"
        )

    else:

        for _, row in review_runs.iterrows():

            f.write(
                f"{row['subject']} "
                f"run-{int(row['run'])} "
                f"status={row['final_run_status']} "
                f"bad={int(row['bad_channels'])} "
                f"review={int(row['review_channels'])}\n"
            )

    f.write(
        "\n" + "=" * 75 + "\n"
    )

    f.write(
        "RAW DATA WAS NOT MODIFIED.\n"
    )

    f.write(
        "NO SET FILES WERE MODIFIED.\n"
    )

    f.write(
        "NO FDT FILES WERE MODIFIED.\n"
    )

    f.write(
        "=" * 75 + "\n"
    )

# ============================================================
# TERMINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "FINAL CHANNEL STATUS"
)

print(
    "=" * 75
)

print(channel_counts)

print(
    "\n" + "=" * 75
)

print(
    "FINAL RUN STATUS"
)

print(
    "=" * 75
)

print(run_counts)

print(
    "\n" + "=" * 75
)

print(
    "BAD CHANNELS"
)

print(
    "=" * 75
)

if len(bad_rows) == 0:

    print("NONE")

else:

    print(
        bad_rows[
            [
                "subject",
                "run",
                "channel",
                "std_ratio",
                "clip_percent",
                "high_amplitude_percent",
                "final_reasons",
            ]
        ].to_string(
            index=False
        )
    )

print(
    "\n" + "=" * 75
)

print(
    "RUNS REQUIRING REVIEW"
)

print(
    "=" * 75
)

if len(review_runs) == 0:

    print("NONE")

else:

    print(
        review_runs[
            [
                "subject",
                "run",
                "final_run_status",
                "bad_channels",
                "review_channels",
            ]
        ].to_string(
            index=False
        )
    )

print(
    "\n" + "=" * 75
)

print(
    "COMPLETE"
)

print(
    "=" * 75
)

print("\nSaved:")
print(CHANNEL_OUT)
print(RUN_OUT)
print(SUMMARY_OUT)

print(
    "\nRAW DATA WAS NOT MODIFIED."
)

print(
    "NO SET FILES WERE MODIFIED."
)

print(
    "NO FDT FILES WERE MODIFIED."
)