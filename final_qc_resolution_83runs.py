import os
import pandas as pd
import numpy as np

PROJECT = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

EPOCH_LOG = os.path.join(
    PROJECT,
    "epochs_v3",
    "logs",
    "epoch_final_qc_83runs.csv"
)

FINAL_DECISION = os.path.join(
    PROJECT,
    "epochs_v3",
    "logs",
    "FINAL_DATASET_DECISION_83runs.csv"
)

CHANNEL_QC = os.path.join(
    PROJECT,
    "qc",
    "final_qc",
    "final_channel_qc_83runs.csv"
)

RUN_QC = os.path.join(
    PROJECT,
    "qc",
    "final_qc",
    "final_run_qc_83runs.csv"
)

ARTIFACT_OVERLAP = os.path.join(
    PROJECT,
    "qc",
    "sub004_temporal",
    "sub004_artifact_event_overlap_v2.csv"
)

DUPLICATE_AUDIT = os.path.join(
    PROJECT,
    "epochs_v2",
    "logs",
    "duplicate_event_audit.csv"
)

OUT_DIR = os.path.join(
    PROJECT,
    "epochs_v3",
    "logs"
)

OUT_CSV = os.path.join(
    OUT_DIR,
    "FINAL_QC_RESOLUTION_83runs.csv"
)

OUT_SUMMARY = os.path.join(
    OUT_DIR,
    "FINAL_QC_RESOLUTION_83runs_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def norm_subject(x):
    x = str(x)
    if x.startswith("sub-"):
        return x
    try:
        return f"sub-{int(x):03d}"
    except Exception:
        return x


def get_run_from_file(filename):
    filename = os.path.basename(str(filename))

    if "_run-" not in filename:
        return np.nan

    try:
        return int(filename.split("_run-")[1].split("_")[0])
    except Exception:
        return np.nan


def find_column(df, candidates):

    for c in candidates:
        if c in df.columns:
            return c

    return None


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FINAL QC RESOLUTION - 83 RUNS")
print("=" * 80)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD MAIN EPOCH QC
# ============================================================

if not os.path.exists(EPOCH_LOG):
    raise FileNotFoundError(
        f"\nMissing epoch QC:\n{EPOCH_LOG}"
    )

epoch = pd.read_csv(EPOCH_LOG)

print("\nEpoch QC:")
print(f"Records: {len(epoch)}")
print(list(epoch.columns))


file_col = find_column(
    epoch,
    ["file", "filename", "epoch_file"]
)

status_col = find_column(
    epoch,
    ["status", "final_status"]
)

if file_col is None:
    raise ValueError(
        "Epoch QC does not contain a file column."
    )

if status_col is None:
    raise ValueError(
        "Epoch QC does not contain a status column."
    )


epoch["file_key"] = epoch[file_col].astype(str)
epoch["subject"] = epoch["file_key"].apply(
    lambda x: os.path.basename(x).split("_")[0]
)

epoch["run"] = epoch["file_key"].apply(
    get_run_from_file
)

epoch["epoch_qc_status"] = (
    epoch[status_col]
    .astype(str)
    .str.upper()
)


# ============================================================
# LOAD FINAL DATASET DECISION
# ============================================================

decision = None

if os.path.exists(FINAL_DECISION):

    decision = pd.read_csv(FINAL_DECISION)

    decision["file_key"] = (
        decision["file"]
        .astype(str)
    )

    if "subject" not in decision.columns:
        decision["subject"] = decision["file_key"].apply(
            lambda x: os.path.basename(x).split("_")[0]
        )

    if "run" not in decision.columns:
        decision["run"] = decision["file_key"].apply(
            get_run_from_file
        )

    print(
        f"\nPrevious final decision records: {len(decision)}"
    )

else:

    print(
        "\nPrevious FINAL_DATASET_DECISION file not found."
    )


# ============================================================
# LOAD CHANNEL QC
# ============================================================

channel = None

if os.path.exists(CHANNEL_QC):

    channel = pd.read_csv(CHANNEL_QC)

    print(
        f"\nChannel QC records: {len(channel)}"
    )

    if "subject" in channel.columns:
        channel["subject"] = channel["subject"].apply(
            norm_subject
        )

    if "run" in channel.columns:
        channel["run"] = pd.to_numeric(
            channel["run"],
            errors="coerce"
        )

else:

    print("\nWARNING: Channel QC file not found.")


# ============================================================
# CHANNEL SUMMARY
# ============================================================

channel_summary = pd.DataFrame()

if channel is not None:

    status_column = find_column(
        channel,
        [
            "final_status",
            "status"
        ]
    )

    if status_column is not None:

        channel[status_column] = (
            channel[status_column]
            .astype(str)
            .str.upper()
        )

        channel_summary = (
            channel
            .groupby(
                ["subject", "run"],
                dropna=False
            )
            .agg(
                bad_channels=(
                    status_column,
                    lambda x: int(
                        (x == "BAD").sum()
                    )
                ),
                review_channels=(
                    status_column,
                    lambda x: int(
                        (x == "REVIEW").sum()
                    )
                ),
                pass_channels=(
                    status_column,
                    lambda x: int(
                        (x == "PASS").sum()
                    )
                )
            )
            .reset_index()
        )


# ============================================================
# LOAD RUN QC
# ============================================================

runqc = None

if os.path.exists(RUN_QC):

    runqc = pd.read_csv(RUN_QC)

    print(
        f"Run QC records: {len(runqc)}"
    )

    if "subject" in runqc.columns:
        runqc["subject"] = runqc["subject"].apply(
            norm_subject
        )

    if "run" in runqc.columns:
        runqc["run"] = pd.to_numeric(
            runqc["run"],
            errors="coerce"
        )


# ============================================================
# MERGE
# ============================================================

df = epoch.copy()

df["subject"] = df["subject"].apply(
    norm_subject
)

if channel_summary is not None and len(channel_summary) > 0:

    df = df.merge(
        channel_summary,
        on=["subject", "run"],
        how="left"
    )

else:

    df["bad_channels"] = 0
    df["review_channels"] = 0
    df["pass_channels"] = 0


df["bad_channels"] = (
    pd.to_numeric(
        df["bad_channels"],
        errors="coerce"
    )
    .fillna(0)
)

df["review_channels"] = (
    pd.to_numeric(
        df["review_channels"],
        errors="coerce"
    )
    .fillna(0)
)


# ============================================================
# EPOCH COUNT
# ============================================================

epoch_count_col = find_column(
    df,
    [
        "epochs",
        "epoch_count",
        "n_epochs",
        "count"
    ]
)

if epoch_count_col is not None:

    df["epoch_count"] = pd.to_numeric(
        df[epoch_count_col],
        errors="coerce"
    )

else:

    df["epoch_count"] = np.nan


# ============================================================
# SFREQ
# ============================================================

sfreq_col = find_column(
    df,
    [
        "sfreq",
        "sampling_rate",
        "sampling_frequency"
    ]
)

if sfreq_col is not None:

    df["sfreq_value"] = pd.to_numeric(
        df[sfreq_col],
        errors="coerce"
    )

else:

    df["sfreq_value"] = np.nan


# ============================================================
# DUPLICATE EVENTS
# ============================================================

df["duplicate_event_count"] = 0

if os.path.exists(DUPLICATE_AUDIT):

    dup = pd.read_csv(DUPLICATE_AUDIT)

    print(
        f"Duplicate audit records: {len(dup)}"
    )

    if len(dup) > 0:

        dup_file_col = find_column(
            dup,
            [
                "file",
                "filename",
                "epoch_file"
            ]
        )

        if dup_file_col is not None:

            dup["file_key"] = (
                dup[dup_file_col]
                .astype(str)
                .apply(os.path.basename)
            )

            dup_counts = (
                dup
                .groupby("file_key")
                .size()
                .reset_index(
                    name="duplicate_event_count"
                )
            )

            df["file_basename"] = (
                df["file_key"]
                .astype(str)
                .apply(os.path.basename)
            )

            df = df.merge(
                dup_counts,
                on="file_basename",
                how="left",
                suffixes=("", "_dup")
            )

            if "duplicate_event_count_dup" in df.columns:

                df["duplicate_event_count"] = (
                    pd.to_numeric(
                        df[
                            "duplicate_event_count_dup"
                        ],
                        errors="coerce"
                    )
                    .fillna(0)
                )

                df.drop(
                    columns=[
                        "duplicate_event_count_dup"
                    ],
                    inplace=True
                )


# ============================================================
# DECISION ENGINE
# ============================================================

def resolve(row):

    status = str(
        row["epoch_qc_status"]
    ).upper()

    bad = int(
        row["bad_channels"]
    )

    review = int(
        row["review_channels"]
    )

    dup = int(
        row["duplicate_event_count"]
    )

    epochs = row["epoch_count"]

    sfreq = row["sfreq_value"]

    reasons = []

    # --------------------------------------------------------
    # HARD EXCLUSION
    # --------------------------------------------------------

    if status in [
        "ZERO_EPOCH",
        "FAILED",
        "ERROR",
        "EPOCH_FILE_MISSING"
    ]:

        return (
            "EXCLUDE",
            "EPOCH_QC_FAILURE"
        )

    # --------------------------------------------------------
    # LOW EPOCH COUNT
    # --------------------------------------------------------

    if pd.notna(epochs):

        if epochs < 100:

            return (
                "EXCLUDE",
                f"VERY_LOW_EPOCH_COUNT_{int(epochs)}"
            )

        elif epochs < 300:

            reasons.append(
                f"LOW_EPOCH_COUNT_{int(epochs)}"
            )

    # --------------------------------------------------------
    # SAMPLING RATE
    # --------------------------------------------------------

    if pd.notna(sfreq):

        if abs(float(sfreq) - 500.0) > 0.01:

            reasons.append(
                f"RESAMPLE_REQUIRED_{float(sfreq):.6f}"
            )

    # --------------------------------------------------------
    # DUPLICATE EVENTS
    # --------------------------------------------------------

    if dup > 0:

        reasons.append(
            f"DUPLICATE_EVENTS_{dup}"
        )

    # --------------------------------------------------------
    # BAD CHANNELS
    # --------------------------------------------------------

    if bad >= 50:

        return (
            "EXCLUDE",
            f"TOO_MANY_BAD_CHANNELS_{bad}"
        )

    elif bad > 0:

        reasons.append(
            f"INTERPOLATION_REQUIRED_{bad}"
        )

    # --------------------------------------------------------
    # REVIEW CHANNELS
    # --------------------------------------------------------

    if review >= 30:

        reasons.append(
            f"MANY_REVIEW_CHANNELS_{review}"
        )

    elif review > 0:

        reasons.append(
            f"REVIEW_CHANNELS_{review}"
        )

    # --------------------------------------------------------
    # ORIGINAL STATUS
    # --------------------------------------------------------

    if status == "BAD":

        reasons.append(
            "ORIGINAL_EPOCH_QC_BAD"
        )

    elif status == "REVIEW":

        reasons.append(
            "ORIGINAL_EPOCH_QC_REVIEW"
        )

    # --------------------------------------------------------
    # FINAL CLASSIFICATION
    # --------------------------------------------------------

    if len(reasons) == 0:

        return (
            "KEEP",
            "QC_PASS"
        )

    # Important:
    # REVIEW does NOT mean exclusion.

    return (
        "KEEP_REVIEW",
        ";".join(reasons)
    )


resolved = df.apply(
    lambda row: pd.Series(
        resolve(row),
        index=[
            "resolution",
            "resolution_reason"
        ]
    ),
    axis=1
)

df = pd.concat(
    [df, resolved],
    axis=1
)


# ============================================================
# SPECIAL CASES KNOWN FROM PREVIOUS QC
# ============================================================

# sub-004 run-2 was identified previously as
# having all EEG channels BAD.
# This is the only automatic hard exclusion here.

mask_sub004_run2 = (
    (df["subject"] == "sub-004")
    &
    (df["run"] == 2)
)

if mask_sub004_run2.any():

    df.loc[
        mask_sub004_run2,
        "resolution"
    ] = "EXCLUDE"

    df.loc[
        mask_sub004_run2,
        "resolution_reason"
    ] = (
        "PREVIOUS_FINAL_QC_69_BAD_CHANNELS"
    )


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    ["subject", "run"]
).reset_index(drop=True)


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

counts = (
    df["resolution"]
    .value_counts()
)

summary = []

summary.append("=" * 80)
summary.append(
    "FINAL QC RESOLUTION - 83 RUNS"
)
summary.append("=" * 80)
summary.append("")

summary.append(
    f"Total records: {len(df)}"
)

summary.append("")

summary.append(
    "RESOLUTION COUNTS"
)

summary.append(
    "-" * 80
)

for name in [
    "KEEP",
    "KEEP_REVIEW",
    "EXCLUDE"
]:

    summary.append(
        f"{name:15s} "
        f"{int(counts.get(name, 0))}"
    )


# ============================================================
# EXCLUSIONS
# ============================================================

summary.append("")
summary.append("=" * 80)
summary.append("EXCLUSIONS")
summary.append("=" * 80)

excluded = df[
    df["resolution"] == "EXCLUDE"
]

if len(excluded) == 0:

    summary.append("NONE")

else:

    for _, r in excluded.iterrows():

        summary.append(
            f"{r['subject']} "
            f"RUN {int(r['run'])} | "
            f"{r['resolution_reason']}"
        )


# ============================================================
# KEEP_REVIEW
# ============================================================

summary.append("")
summary.append("=" * 80)
summary.append("KEEP_REVIEW")
summary.append("=" * 80)

kr = df[
    df["resolution"] == "KEEP_REVIEW"
]

if len(kr) == 0:

    summary.append("NONE")

else:

    for _, r in kr.iterrows():

        summary.append(
            f"{r['subject']} "
            f"RUN {int(r['run'])} | "
            f"{r['resolution_reason']}"
        )


# ============================================================
# KEEP
# ============================================================

summary.append("")
summary.append("=" * 80)
summary.append("KEEP")
summary.append("=" * 80)

keep = df[
    df["resolution"] == "KEEP"
]

summary.append(
    f"Total KEEP: {len(keep)}"
)

for _, r in keep.iterrows():

    summary.append(
        f"{r['subject']} "
        f"RUN {int(r['run'])}"
    )


# ============================================================
# IMPORTANT
# ============================================================

summary.append("")
summary.append("=" * 80)
summary.append("IMPORTANT")
summary.append("=" * 80)
summary.append("")
summary.append(
    "This script ONLY reads QC files."
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
    "NO FIF FILE WAS MODIFIED."
)
summary.append(
    "NO EPOCH FILE WAS MODIFIED."
)
summary.append(
    "NO CHANNEL WAS INTERPOLATED."
)
summary.append(
    "NO SAMPLE WAS DELETED."
)
summary.append("")
summary.append(
    "KEEP_REVIEW means the run is NOT excluded."
)
summary.append(
    "It requires controlled preprocessing "
    "before final analysis."
)


with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("")
print("=" * 80)
print("RESOLUTION COUNTS")
print("=" * 80)

print(
    counts
)

print("")
print("=" * 80)
print("EXCLUSIONS")
print("=" * 80)

if len(excluded) == 0:

    print("NONE")

else:

    print(
        excluded[
            [
                "subject",
                "run",
                "resolution_reason"
            ]
        ].to_string(index=False)
    )


print("")
print("=" * 80)
print("KEEP_REVIEW")
print("=" * 80)

if len(kr) == 0:

    print("NONE")

else:

    print(
        kr[
            [
                "subject",
                "run",
                "resolution_reason"
            ]
        ].to_string(index=False)
    )


print("")
print("=" * 80)
print("COMPLETE")
print("=" * 80)

print("")
print("Saved:")
print(OUT_CSV)
print(OUT_SUMMARY)

print("")
print("RAW DATA WAS NOT MODIFIED.")
print("NO SET FILE WAS MODIFIED.")
print("NO FDT FILE WAS MODIFIED.")
print("NO FIF FILE WAS MODIFIED.")
print("NO EPOCH FILE WAS MODIFIED.")