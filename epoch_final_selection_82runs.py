import os
import glob
import pandas as pd
import numpy as np

# ============================================================
# FINAL EPOCH SELECTION - 82 RUNS
# READ-ONLY / NO DATA MODIFICATION
# ============================================================

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

EPOCH_DIR = os.path.join(BASE, "epochs_clean")

ARTIFACT_CSV = os.path.join(
    EPOCH_DIR,
    "logs",
    "artifact_duration_qc",
    "epoch_artifact_duration_qc_82runs.csv"
)

OUT_DIR = os.path.join(
    EPOCH_DIR,
    "logs",
    "final_selection"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(
    OUT_DIR,
    "epoch_final_selection_82runs.csv"
)

OUT_EPOCH_CSV = os.path.join(
    OUT_DIR,
    "epoch_keep_recommendations_82runs.csv"
)

SUMMARY = os.path.join(
    OUT_DIR,
    "epoch_final_selection_82runs_summary.txt"
)

print("=" * 80)
print("FINAL EPOCH SELECTION - 82 RUNS")
print("=" * 80)

# ------------------------------------------------------------
# LOAD ARTIFACT DURATION QC
# ------------------------------------------------------------

if not os.path.exists(ARTIFACT_CSV):
    raise FileNotFoundError(
        f"Artifact duration QC file not found:\n{ARTIFACT_CSV}"
    )

df = pd.read_csv(ARTIFACT_CSV)

print(f"\nLoaded records: {len(df)}")
print("Columns:")
print(list(df.columns))

# ------------------------------------------------------------
# FIND IMPORTANT COLUMNS
# ------------------------------------------------------------

def find_column(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"Required column not found.\n"
            f"Candidates: {candidates}\n"
            f"Available: {list(df.columns)}"
        )
    return None


file_col = find_column(
    df,
    ["file", "epoch_file", "filename"]
)

class_col = find_column(
    df,
    ["diagnostic_class", "artifact_class"]
)

# Epoch index
epoch_col = find_column(
    df,
    ["epoch", "epoch_index", "epoch_idx"],
    required=False
)

# ------------------------------------------------------------
# NORMALIZE
# ------------------------------------------------------------

df[class_col] = df[class_col].astype(str).str.upper().str.strip()

if epoch_col is not None:
    df["epoch_index_final"] = pd.to_numeric(
        df[epoch_col],
        errors="coerce"
    )
else:
    print("\nWARNING: Epoch index column not found.")
    print("Creating sequential epoch index within each file.")
    df["epoch_index_final"] = (
        df.groupby(file_col).cumcount()
    )

# ------------------------------------------------------------
# CLASS COUNTS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("GLOBAL ARTIFACT CLASS COUNTS")
print("=" * 80)

global_counts = df[class_col].value_counts()

print(global_counts)

# ------------------------------------------------------------
# CONSERVATIVE KEEP POLICY
# ------------------------------------------------------------
#
# CLEAN  -> KEEP
# BRIEF  -> KEEP_REVIEW
# MODERATE -> REVIEW
# HIGH -> EXCLUDE
# SEVERE -> EXCLUDE
#
# We DO NOT delete anything.
# This is only a recommendation.
# ------------------------------------------------------------

def classify(x):
    if x == "CLEAN":
        return "KEEP"
    elif x == "BRIEF":
        return "KEEP_REVIEW"
    elif x == "MODERATE":
        return "REVIEW"
    elif x in ["HIGH", "SEVERE"]:
        return "EXCLUDE_RECOMMENDED"
    else:
        return "REVIEW"

df["recommendation"] = df[class_col].apply(classify)

# ------------------------------------------------------------
# RUN-LEVEL SUMMARY
# ------------------------------------------------------------

rows = []

for filename, g in df.groupby(file_col, sort=True):

    counts = g[class_col].value_counts()

    total = len(g)

    clean = int(counts.get("CLEAN", 0))
    brief = int(counts.get("BRIEF", 0))
    moderate = int(counts.get("MODERATE", 0))
    high = int(counts.get("HIGH", 0))
    severe = int(counts.get("SEVERE", 0))

    keep = clean
    keep_review = brief
    review = moderate
    exclude = high + severe

    usable_strict = keep
    usable_extended = keep + brief

    clean_pct = 100 * clean / total if total else 0
    problematic_pct = 100 * exclude / total if total else 0
    non_clean_pct = 100 * (total - clean) / total if total else 0

    # --------------------------------------------------------
    # RUN DECISION
    # --------------------------------------------------------

    if severe >= max(1, int(0.10 * total)):
        run_status = "BAD"

    elif high >= max(1, int(0.20 * total)):
        run_status = "REVIEW"

    elif clean_pct >= 50:
        run_status = "PASS"

    else:
        run_status = "REVIEW"

    rows.append({
        "file": filename,
        "total_epochs": total,
        "clean_epochs": clean,
        "brief_epochs": brief,
        "moderate_epochs": moderate,
        "high_epochs": high,
        "severe_epochs": severe,
        "strict_keep_epochs": usable_strict,
        "extended_keep_epochs": usable_extended,
        "exclude_recommended_epochs": exclude,
        "clean_percent": clean_pct,
        "non_clean_percent": non_clean_pct,
        "high_severe_percent": problematic_pct,
        "run_status": run_status
    })

run_df = pd.DataFrame(rows)

# ------------------------------------------------------------
# SAVE EPOCH-LEVEL RECOMMENDATIONS
# ------------------------------------------------------------

df.to_csv(
    OUT_EPOCH_CSV,
    index=False
)

# ------------------------------------------------------------
# SAVE RUN-LEVEL SUMMARY
# ------------------------------------------------------------

run_df.to_csv(
    OUT_CSV,
    index=False
)

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

with open(SUMMARY, "w", encoding="utf-8") as f:

    f.write("=" * 80 + "\n")
    f.write("FINAL EPOCH SELECTION - 82 RUNS\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total epoch records: {len(df)}\n")
    f.write(f"Total runs: {len(run_df)}\n\n")

    f.write("=" * 80 + "\n")
    f.write("GLOBAL CLASS COUNTS\n")
    f.write("=" * 80 + "\n")

    for cls, count in global_counts.items():
        pct = 100 * count / len(df)
        f.write(f"{cls:15s} {count:8d} ({pct:6.2f}%)\n")

    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("GLOBAL RECOMMENDATIONS\n")
    f.write("=" * 80 + "\n")

    rec_counts = df["recommendation"].value_counts()

    for rec, count in rec_counts.items():
        pct = 100 * count / len(df)
        f.write(f"{rec:25s} {count:8d} ({pct:6.2f}%)\n")

    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("RUN STATUS COUNTS\n")
    f.write("=" * 80 + "\n")

    for status, count in run_df["run_status"].value_counts().items():
        f.write(f"{status:15s} {count:8d}\n")

    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("RUNS REQUIRING REVIEW\n")
    f.write("=" * 80 + "\n")

    review_runs = run_df[
        run_df["run_status"].isin(["REVIEW", "BAD"])
    ].sort_values(
        ["run_status", "clean_percent"]
    )

    if len(review_runs) == 0:
        f.write("NONE\n")
    else:
        f.write(
            review_runs.to_string(index=False)
        )
        f.write("\n")

    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("STRICT DATASET SIZE\n")
    f.write("=" * 80 + "\n")

    strict_total = int(run_df["strict_keep_epochs"].sum())
    extended_total = int(run_df["extended_keep_epochs"].sum())
    excluded_total = int(run_df["exclude_recommended_epochs"].sum())

    f.write(
        f"CLEAN-only epochs:        {strict_total}\n"
    )
    f.write(
        f"CLEAN + BRIEF epochs:     {extended_total}\n"
    )
    f.write(
        f"HIGH + SEVERE suggested:   {excluded_total}\n"
    )

    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("IMPORTANT\n")
    f.write("=" * 80 + "\n")
    f.write(
        "This script ONLY analyzes existing QC results.\n"
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
    f.write(
        "The recommendations require scientific review before exclusion.\n"
    )

# ------------------------------------------------------------
# CONSOLE SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL SELECTION COMPLETE")
print("=" * 80)

print("\nGLOBAL CLASS COUNTS")
print(global_counts)

print("\nRECOMMENDATION COUNTS")
print(df["recommendation"].value_counts())

print("\nRUN STATUS COUNTS")
print(run_df["run_status"].value_counts())

print("\nSTRICT CLEAN EPOCHS:")
print(int(run_df["strict_keep_epochs"].sum()))

print("\nCLEAN + BRIEF EPOCHS:")
print(int(run_df["extended_keep_epochs"].sum()))

print("\nHIGH + SEVERE:")
print(int(run_df["exclude_recommended_epochs"].sum()))

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUT_CSV)
print(OUT_EPOCH_CSV)
print(SUMMARY)

print("\n" + "=" * 80)
print("NO DATA WAS MODIFIED.")
print("=" * 80)