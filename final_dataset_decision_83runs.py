import os
import pandas as pd
import numpy as np

PROJECT = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

EPOCH_DIR = os.path.join(PROJECT, "epochs_v3")
LOG_DIR = os.path.join(EPOCH_DIR, "logs")

QC_FILE = os.path.join(LOG_DIR, "epoch_final_qc_83runs.csv")

OUT_CSV = os.path.join(LOG_DIR, "FINAL_DATASET_DECISION_83runs.csv")
OUT_SUMMARY = os.path.join(LOG_DIR, "FINAL_DATASET_DECISION_83runs_summary.txt")


print("=" * 80)
print("FINAL DATASET DECISION - 83 RUNS")
print("=" * 80)

if not os.path.exists(QC_FILE):
    raise FileNotFoundError(
        f"\nQC file not found:\n{QC_FILE}\n"
    )

df = pd.read_csv(QC_FILE)

print(f"\nQC records found: {len(df)}")
print("\nColumns:")
print(list(df.columns))

# ---------------------------------------------------------
# Helper: find column
# ---------------------------------------------------------

def find_col(possible):
    for c in possible:
        if c in df.columns:
            return c
    return None


file_col = find_col(["file", "filename", "epoch_file"])
status_col = find_col(["status", "final_status"])
bad_col = find_col(["bad_channels", "bad_channel_count"])
review_col = find_col(["review_channels", "suspicious_channels"])
sfreq_col = find_col(["sfreq", "sampling_rate"])

if file_col is None:
    raise ValueError("Could not find epoch filename column.")

if status_col is None:
    raise ValueError("Could not find status column.")

print(f"\nUsing file column:   {file_col}")
print(f"Using status column: {status_col}")

# ---------------------------------------------------------
# Normalize
# ---------------------------------------------------------

df["file"] = df[file_col].astype(str)
df["original_qc_status"] = df[status_col].astype(str).str.upper()

if bad_col is not None:
    df["bad_channels_num"] = pd.to_numeric(
        df[bad_col], errors="coerce"
    ).fillna(0)
else:
    df["bad_channels_num"] = 0

if review_col is not None:
    df["review_channels_num"] = pd.to_numeric(
        df[review_col], errors="coerce"
    ).fillna(0)
else:
    df["review_channels_num"] = 0

if sfreq_col is not None:
    df["sfreq_num"] = pd.to_numeric(
        df[sfreq_col], errors="coerce"
    )
else:
    df["sfreq_num"] = np.nan

# ---------------------------------------------------------
# Decision logic
# IMPORTANT:
# This script does NOT modify any FIF file.
# It only creates a decision table.
# ---------------------------------------------------------

def decide(row):

    status = row["original_qc_status"]
    bad = row["bad_channels_num"]
    review = row["review_channels_num"]
    sfreq = row["sfreq_num"]

    reasons = []

    # Missing / broken QC
    if status in ["FAILED", "ERROR", "ZERO_EPOCH", "EPOCH_FILE_MISSING"]:
        return "EXCLUDE", f"ORIGINAL_QC_{status}"

    # Explicit BAD channels
    if bad >= 3:
        reasons.append(f"BAD_CHANNELS_{int(bad)}")

    # Extremely suspicious channel count
    if bad >= 1 and bad >= 0.10 * 71:
        reasons.append("HIGH_BAD_CHANNEL_FRACTION")

    # Sampling frequency mismatch
    if pd.notna(sfreq):
        if abs(float(sfreq) - 500.0) > 0.01:
            reasons.append(f"SFREQ_REVIEW_{sfreq:.6f}")

    # Very low epoch count
    if "count" in row.index:
        try:
            n = float(row["count"])
            if n < 300:
                reasons.append(f"LOW_EPOCH_COUNT_{int(n)}")
        except:
            pass

    # Decision
    if len(reasons) > 0:
        return "REVIEW", ";".join(reasons)

    if status == "BAD":
        return "REVIEW", "BAD_STATUS_REQUIRES_MANUAL_DECISION"

    if status == "REVIEW":
        return "REVIEW", "ORIGINAL_QC_REVIEW"

    return "PASS", "QC_PASS"

results = df.apply(
    lambda row: pd.Series(
        decide(row),
        index=["final_dataset_status", "decision_reason"]
    ),
    axis=1
)

df = pd.concat([df, results], axis=1)

# ---------------------------------------------------------
# Extract subject/run
# ---------------------------------------------------------

def extract_subject(filename):
    base = os.path.basename(filename)
    if base.startswith("sub-"):
        return base.split("_")[0]
    return "UNKNOWN"

def extract_run(filename):
    base = os.path.basename(filename)

    marker = "_run-"

    if marker in base:
        try:
            rest = base.split(marker, 1)[1]
            return int(rest.split("_")[0])
        except:
            return -1

    return -1

df["subject"] = df["file"].apply(extract_subject)
df["run"] = df["file"].apply(extract_run)

# ---------------------------------------------------------
# Sort
# ---------------------------------------------------------

df = df.sort_values(
    ["subject", "run", "file"]
).reset_index(drop=True)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    OUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

status_counts = (
    df["final_dataset_status"]
    .value_counts()
    .reindex(["PASS", "REVIEW", "EXCLUDE"], fill_value=0)
)

summary = []

summary.append("=" * 80)
summary.append("FINAL DATASET DECISION - 83 RUNS")
summary.append("=" * 80)
summary.append("")

summary.append(f"Total QC records: {len(df)}")
summary.append("")

summary.append("FINAL STATUS COUNTS")
summary.append("-" * 80)

for status, count in status_counts.items():
    summary.append(f"{status:10s} {int(count)}")

summary.append("")
summary.append("=" * 80)
summary.append("REVIEW RUNS")
summary.append("=" * 80)

review_df = df[df["final_dataset_status"] == "REVIEW"]

if len(review_df) == 0:
    summary.append("NONE")
else:
    for _, r in review_df.iterrows():
        summary.append(
            f"{r['subject']:8s} "
            f"RUN {int(r['run']):2d} | "
            f"{r['decision_reason']}"
        )

summary.append("")
summary.append("=" * 80)
summary.append("EXCLUDED RUNS")
summary.append("=" * 80)

exclude_df = df[df["final_dataset_status"] == "EXCLUDE"]

if len(exclude_df) == 0:
    summary.append("NONE")
else:
    for _, r in exclude_df.iterrows():
        summary.append(
            f"{r['subject']:8s} "
            f"RUN {int(r['run']):2d} | "
            f"{r['decision_reason']}"
        )

summary.append("")
summary.append("=" * 80)
summary.append("PASS RUNS")
summary.append("=" * 80)

pass_df = df[df["final_dataset_status"] == "PASS"]

summary.append(f"Total PASS runs: {len(pass_df)}")
summary.append("")

for _, r in pass_df.iterrows():
    summary.append(
        f"{r['subject']:8s} RUN {int(r['run']):2d}"
    )

summary.append("")
summary.append("=" * 80)
summary.append("IMPORTANT")
summary.append("=" * 80)
summary.append("")
summary.append("This script ONLY reads the epoch QC CSV.")
summary.append("NO FIF FILE WAS MODIFIED.")
summary.append("NO RAW DATA WAS MODIFIED.")
summary.append("NO EPOCH DATA WAS DELETED.")
summary.append("")
summary.append(
    "REVIEW runs are NOT automatically excluded."
)
summary.append(
    "They require the next QC decision step before Perturbation."
)

with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(summary))

# ---------------------------------------------------------
# Terminal output
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("FINAL STATUS COUNTS")
print("=" * 80)
print(status_counts)

print("\n")
print("=" * 80)
print("REVIEW RUNS")
print("=" * 80)

if len(review_df) == 0:
    print("NONE")
else:
    print(
        review_df[
            [
                "subject",
                "run",
                "final_dataset_status",
                "decision_reason"
            ]
        ].to_string(index=False)
    )

print("\n")
print("=" * 80)
print("EXCLUDED RUNS")
print("=" * 80)

if len(exclude_df) == 0:
    print("NONE")
else:
    print(
        exclude_df[
            [
                "subject",
                "run",
                "decision_reason"
            ]
        ].to_string(index=False)
    )

print("\n")
print("=" * 80)
print("COMPLETE")
print("=" * 80)

print("\nSaved:")
print(OUT_CSV)
print(OUT_SUMMARY)

print("\nRAW DATA WAS NOT MODIFIED.")
print("NO FIF FILE WAS MODIFIED.")
print("NO EPOCH FILE WAS MODIFIED.")