import os
import pandas as pd
import numpy as np

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

EPOCH_QC = os.path.join(
    BASE, "epochs_v3", "logs", "epoch_final_qc_83runs.csv"
)

PREV_DECISION = os.path.join(
    BASE, "epochs_v3", "logs", "FINAL_DATASET_DECISION_83runs.csv"
)

CHANNEL_QC = os.path.join(
    BASE, "qc", "final_qc", "final_channel_qc_83runs.csv"
)

RUN_QC = os.path.join(
    BASE, "qc", "final_qc", "final_run_qc_83runs.csv"
)

OUT_DIR = os.path.join(BASE, "epochs_v3", "logs")

OUT_CSV = os.path.join(
    OUT_DIR, "FINAL_DATASET_DECISION_v2_83runs.csv"
)

OUT_SUMMARY = os.path.join(
    OUT_DIR, "FINAL_DATASET_DECISION_v2_83runs_summary.txt"
)

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 80)
print("FINAL QC RESOLUTION V2 - 83 RUNS")
print("=" * 80)

# ----------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------

epoch = pd.read_csv(EPOCH_QC)
prev = pd.read_csv(PREV_DECISION)
channel = pd.read_csv(CHANNEL_QC)
run = pd.read_csv(RUN_QC)

print(f"Epoch QC records: {len(epoch)}")
print(f"Previous decision records: {len(prev)}")
print(f"Channel QC records: {len(channel)}")
print(f"Run QC records: {len(run)}")

# ----------------------------------------------------------------------
# NORMALIZE SUBJECT / RUN
# ----------------------------------------------------------------------

def normalize_subject(df):
    if "subject" in df.columns:
        df["subject"] = df["subject"].astype(str).str.strip()
    return df


for df in [prev, channel, run]:
    normalize_subject(df)

# Epoch file -> subject/run
if "file" in epoch.columns:
    epoch["file"] = epoch["file"].astype(str)

    epoch["subject"] = epoch["file"].str.extract(
        r"(sub-\d+)", expand=False
    )

    epoch["run"] = pd.to_numeric(
        epoch["file"].str.extract(
            r"_run-(\d+)", expand=False
        ),
        errors="coerce"
    )

# Previous decision
if "run" in prev.columns:
    prev["run"] = pd.to_numeric(prev["run"], errors="coerce")

if "run" in channel.columns:
    channel["run"] = pd.to_numeric(channel["run"], errors="coerce")

if "run" in run.columns:
    run["run"] = pd.to_numeric(run["run"], errors="coerce")

# ----------------------------------------------------------------------
# BUILD CHANNEL SUMMARY
# ----------------------------------------------------------------------

print("\nBuilding channel summary...")

channel["is_bad"] = (
    channel["final_status"]
    .astype(str)
    .str.upper()
    .eq("BAD")
)

channel["is_review"] = (
    channel["final_status"]
    .astype(str)
    .str.upper()
    .eq("REVIEW")
)

channel_summary = (
    channel
    .groupby(["subject", "run"], as_index=False)
    .agg(
        qc_bad_channels=("is_bad", "sum"),
        qc_review_channels=("is_review", "sum")
    )
)

# ----------------------------------------------------------------------
# BUILD RUN SUMMARY
# ----------------------------------------------------------------------

print("Building run summary...")

run_keep = [
    c for c in [
        "subject",
        "run",
        "file",
        "channels_total",
        "eeg_channels_x",
        "eeg_channels",
        "srate",
        "duration_seconds",
        "median_eeg_std",
        "bad_channels",
        "review_channels",
        "pass_channels",
        "eeg_channels_checked",
        "final_run_status"
    ]
    if c in run.columns
]

run_summary = run[run_keep].copy()

# Avoid duplicate column names after merge
rename_map = {}

if "bad_channels" in run_summary.columns:
    rename_map["bad_channels"] = "run_bad_channels"

if "review_channels" in run_summary.columns:
    rename_map["review_channels"] = "run_review_channels"

if "final_run_status" in run_summary.columns:
    rename_map["final_run_status"] = "original_run_status"

run_summary = run_summary.rename(columns=rename_map)

# ----------------------------------------------------------------------
# PREVIOUS DECISION CLEANUP
# ----------------------------------------------------------------------

prev_keep = [
    c for c in [
        "subject",
        "run",
        "final_dataset_status",
        "decision_reason"
    ]
    if c in prev.columns
]

prev_summary = prev[prev_keep].copy()

if "final_dataset_status" not in prev_summary.columns:
    prev_summary["final_dataset_status"] = "REVIEW"

if "decision_reason" not in prev_summary.columns:
    prev_summary["decision_reason"] = ""

# ----------------------------------------------------------------------
# MERGE
# ----------------------------------------------------------------------

print("\nMerging all QC layers...")

df = epoch.merge(
    channel_summary,
    on=["subject", "run"],
    how="left"
)

df = df.merge(
    run_summary,
    on=["subject", "run"],
    how="left",
    suffixes=("", "_RUN")
)

df = df.merge(
    prev_summary,
    on=["subject", "run"],
    how="left",
    suffixes=("", "_PREVIOUS")
)

# ----------------------------------------------------------------------
# NUMERIC CLEANUP
# ----------------------------------------------------------------------

numeric_cols = [
    "n_epochs",
    "n_channels",
    "n_times",
    "sfreq",
    "epoch_duration_sec",
    "nan_percent",
    "inf_percent",
    "global_std",
    "max_abs",
    "high_amplitude_percent",
    "bad_amplitude_percent",
    "suspicious_epochs",
    "bad_epochs",
    "suspicious_channels",
    "bad_channels",
    "event_count",
    "qc_bad_channels",
    "qc_review_channels",
    "run_bad_channels",
    "run_review_channels"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------------------------------------------------
# FIND ACTUAL BAD/REVIEW COUNTS
# ----------------------------------------------------------------------

if "run_bad_channels" in df.columns:
    bad_count = df["run_bad_channels"].fillna(0)
else:
    bad_count = pd.Series(0, index=df.index)

if "run_review_channels" in df.columns:
    review_count = df["run_review_channels"].fillna(0)
else:
    review_count = pd.Series(0, index=df.index)

# Prefer the epoch-level counts if they exist
if "bad_channels" in df.columns:
    epoch_bad = df["bad_channels"].fillna(0)
    bad_count = np.maximum(bad_count, epoch_bad)

if "suspicious_channels" in df.columns:
    suspicious = df["suspicious_channels"].fillna(0)
else:
    suspicious = pd.Series(0, index=df.index)

# ----------------------------------------------------------------------
# FINAL DECISION LOGIC
# ----------------------------------------------------------------------

final_status = []
final_reason = []

for i, row in df.iterrows():

    reasons = []

    n_epochs = row.get("n_epochs", np.nan)
    n_channels = row.get("n_channels", np.nan)
    sfreq = row.get("sfreq", np.nan)

    b = float(bad_count.loc[i]) if pd.notna(bad_count.loc[i]) else 0
    r = float(review_count.loc[i]) if pd.notna(review_count.loc[i]) else 0
    s = float(suspicious.loc[i]) if pd.notna(suspicious.loc[i]) else 0

    # --------------------------------------------------------------
    # HARD FAIL CONDITIONS
    # --------------------------------------------------------------

    if pd.notna(n_epochs) and n_epochs <= 0:
        reasons.append("ZERO_EPOCHS")

    if pd.notna(n_channels) and n_channels < 60:
        reasons.append("TOO_FEW_CHANNELS")

    if pd.notna(row.get("nan_percent", np.nan)):
        if row["nan_percent"] > 0:
            reasons.append("NAN_DATA")

    if pd.notna(row.get("inf_percent", np.nan)):
        if row["inf_percent"] > 0:
            reasons.append("INF_DATA")

    # --------------------------------------------------------------
    # BAD CHANNELS
    # --------------------------------------------------------------

    if b >= 35:
        reasons.append(f"BAD_CHANNELS_{int(b)}")

    elif b > 0:
        reasons.append(f"BAD_CHANNELS_{int(b)}")

    # --------------------------------------------------------------
    # SUSPICIOUS CHANNELS
    # --------------------------------------------------------------

    if s > 0:
        reasons.append(f"SUSPICIOUS_CHANNELS_{int(s)}")

    # --------------------------------------------------------------
    # SAMPLING RATE
    # --------------------------------------------------------------

    if pd.notna(sfreq):
        if abs(float(sfreq) - 500.0) > 0.1:
            reasons.append(f"SFREQ_REVIEW_{sfreq}")

    # --------------------------------------------------------------
    # EPOCH COUNT
    # --------------------------------------------------------------

    if pd.notna(n_epochs):
        if n_epochs < 300:
            reasons.append(f"LOW_EPOCH_COUNT_{int(n_epochs)}")

    # --------------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------------

    if any(
        x in reasons
        for x in [
            "ZERO_EPOCHS",
            "TOO_FEW_CHANNELS",
            "NAN_DATA",
            "INF_DATA"
        ]
    ):
        status = "BAD"

    elif b >= 35:
        status = "BAD"

    elif len(reasons) > 0:
        status = "REVIEW"

    else:
        status = "PASS"

    final_status.append(status)
    final_reason.append(";".join(reasons) if reasons else "CLEAN")

df["final_resolved_status"] = final_status
df["final_resolved_reason"] = final_reason

# ----------------------------------------------------------------------
# IMPORTANT: DO NOT AUTOMATICALLY EXCLUDE REVIEW
# ----------------------------------------------------------------------

df["final_dataset_status"] = df["final_resolved_status"]

# Only definite catastrophic cases are excluded
df["final_dataset_status"] = np.where(
    df["final_resolved_status"].eq("BAD")
    & (
        df["final_resolved_reason"].str.contains(
            "ZERO_EPOCHS|TOO_FEW_CHANNELS|NAN_DATA|INF_DATA",
            regex=True,
            na=False
        )
        | (
            pd.to_numeric(
                df.get("run_bad_channels", 0),
                errors="coerce"
            ).fillna(0) >= 50
        )
    ),
    "EXCLUDE",
    df["final_dataset_status"]
)

# ----------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------

df.to_csv(OUT_CSV, index=False)

# ----------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------

status_counts = df["final_dataset_status"].value_counts()

summary_lines = []

summary_lines.append("=" * 80)
summary_lines.append("FINAL QC RESOLUTION V2 - 83 RUNS")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append(f"Total records: {len(df)}")
summary_lines.append("")
summary_lines.append("FINAL DATASET STATUS")
summary_lines.append("--------------------")

for status, count in status_counts.items():
    summary_lines.append(f"{status:15s} {count}")

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("RUNS REQUIRING REVIEW")
summary_lines.append("=" * 80)

review_df = df[df["final_dataset_status"] == "REVIEW"]

if len(review_df) == 0:
    summary_lines.append("NONE")
else:
    for _, row in review_df.iterrows():
        summary_lines.append(
            f"{row['subject']} RUN {int(row['run'])} | "
            f"{row['final_resolved_reason']}"
        )

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("EXCLUDED RUNS")
summary_lines.append("=" * 80)

excluded = df[df["final_dataset_status"] == "EXCLUDE"]

if len(excluded) == 0:
    summary_lines.append("NONE")
else:
    for _, row in excluded.iterrows():
        summary_lines.append(
            f"{row['subject']} RUN {int(row['run'])} | "
            f"{row['final_resolved_reason']}"
        )

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("COMPLETE")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append(f"Saved:")
summary_lines.append(OUT_CSV)
summary_lines.append(OUT_SUMMARY)
summary_lines.append("")
summary_lines.append("RAW DATA WAS NOT MODIFIED.")
summary_lines.append("NO SET FILE WAS MODIFIED.")
summary_lines.append("NO FDT FILE WAS MODIFIED.")
summary_lines.append("NO PREPROCESSED FIF FILE WAS MODIFIED.")
summary_lines.append("NO EPOCH FILE WAS MODIFIED.")

with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\n".join(summary_lines))