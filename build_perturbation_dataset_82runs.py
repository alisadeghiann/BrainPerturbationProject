import os
import re
import shutil
import traceback
import pandas as pd
import numpy as np
import mne

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

EPOCH_DIR = os.path.join(BASE_DIR, "epochs_clean")

MASTER_CSV = os.path.join(
    EPOCH_DIR,
    "logs",
    "perturbation_master",
    "perturbation_master_82runs.csv"
)

OUTPUT_DIR = os.path.join(
    EPOCH_DIR,
    "logs",
    "perturbation_dataset"
)

ELIGIBLE_DIR = os.path.join(OUTPUT_DIR, "ELIGIBLE")
REVIEW_DIR = os.path.join(OUTPUT_DIR, "ELIGIBLE_REVIEW")

LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ELIGIBLE_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MANIFEST_OUT = os.path.join(
    OUTPUT_DIR,
    "perturbation_dataset_manifest.csv"
)

SUMMARY_OUT = os.path.join(
    LOG_DIR,
    "perturbation_dataset_summary.txt"
)

# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_run_from_filename(filename):
    m = re.search(r"_run-(\d+)_", filename)
    if m:
        return int(m.group(1))
    return None


def extract_subject_from_filename(filename):
    m = re.search(r"(sub-\d+)", filename)
    if m:
        return m.group(1)
    return None


# ============================================================
# START
# ============================================================

print("=" * 80)
print("BUILD PERTURBATION DATASET - 82 RUNS")
print("=" * 80)

print("\nMaster CSV:")
print(MASTER_CSV)

if not os.path.exists(MASTER_CSV):
    raise FileNotFoundError(
        f"Master dataset not found:\n{MASTER_CSV}"
    )

# ============================================================
# LOAD MASTER
# ============================================================

print("\nLoading perturbation master dataset...")

master = pd.read_csv(MASTER_CSV)

print(f"Master records: {len(master)}")

print("\nMASTER COLUMNS")
print("-" * 80)
print(list(master.columns))

# ============================================================
# REQUIRED COLUMNS
# ============================================================

subject_col = find_column(
    master,
    ["subject", "sub", "participant"]
)

run_col = find_column(
    master,
    ["run", "run_number"]
)

epoch_col = find_column(
    master,
    ["epoch", "epoch_index", "epoch_idx"]
)

status_col = find_column(
    master,
    ["perturbation_status", "status"]
)

recommendation_col = find_column(
    master,
    ["recommendation", "final_recommendation"]
)

condition_col = find_column(
    master,
    ["condition", "event_type", "event"]
)

file_col = find_column(
    master,
    ["file", "epoch_file", "source_file"]
)

required = {
    "subject": subject_col,
    "run": run_col,
    "epoch": epoch_col,
    "perturbation_status": status_col,
    "recommendation": recommendation_col,
    "condition": condition_col,
}

print("\nCOLUMN MAPPING")
print("-" * 80)

for name, col in required.items():
    print(f"{name:25s} -> {col}")

missing = [
    name for name, col in required.items()
    if col is None
]

if missing:
    raise ValueError(
        "Required columns missing from master CSV: "
        + ", ".join(missing)
    )

# ============================================================
# NORMALIZE
# ============================================================

master[subject_col] = master[subject_col].astype(str)

master[run_col] = pd.to_numeric(
    master[run_col],
    errors="coerce"
).astype("Int64")

master[epoch_col] = pd.to_numeric(
    master[epoch_col],
    errors="coerce"
).astype("Int64")

master[status_col] = master[status_col].astype(str)
master[recommendation_col] = master[recommendation_col].astype(str)
master[condition_col] = master[condition_col].astype(str)

# ============================================================
# BASIC COUNTS
# ============================================================

print("\nPERTURBATION STATUS")
print("-" * 80)
print(master[status_col].value_counts())

print("\nRECOMMENDATION")
print("-" * 80)
print(master[recommendation_col].value_counts())

# ============================================================
# DEFINE GROUPS
# ============================================================

eligible_mask = (
    master[status_col] == "ELIGIBLE"
)

eligible_review_mask = (
    master[status_col] == "ELIGIBLE_REVIEW"
)

review_mask = (
    master[status_col] == "REVIEW"
)

exclude_mask = (
    master[status_col] == "EXCLUDE"
)

eligible_master = master[eligible_mask].copy()
eligible_review_master = master[eligible_review_mask].copy()

print("\nDATASET GROUPS")
print("-" * 80)

print(f"ELIGIBLE:        {len(eligible_master)}")
print(f"ELIGIBLE_REVIEW: {len(eligible_review_master)}")
print(f"REVIEW:          {review_mask.sum()}")
print(f"EXCLUDE:         {exclude_mask.sum()}")

# ============================================================
# FIND EPOCH FILES
# ============================================================

print("\nSearching clean epoch files...")

epoch_files = {}

for filename in os.listdir(EPOCH_DIR):

    if not filename.endswith("_clean_epo.fif"):
        continue

    subject = extract_subject_from_filename(filename)
    run = extract_run_from_filename(filename)

    if subject is None or run is None:
        continue

    key = (subject, run)

    epoch_files[key] = os.path.join(
        EPOCH_DIR,
        filename
    )

print(f"Epoch files found: {len(epoch_files)}")

# ============================================================
# VALIDATE MASTER ↔ FILES
# ============================================================

all_keys = set(
    zip(
        master[subject_col],
        master[run_col].astype(int)
    )
)

missing_files = []

for key in sorted(all_keys):

    if key not in epoch_files:
        missing_files.append(key)

if missing_files:

    print("\nWARNING - MISSING EPOCH FILES")
    for key in missing_files:
        print(key)

# ============================================================
# OUTPUT RECORDS
# ============================================================

manifest_records = []
run_summary = []

total_input_epochs = 0
total_eligible = 0
total_eligible_review = 0

processed_runs = 0
failed_runs = 0

# ============================================================
# PROCESS EACH RUN
# ============================================================

print("\n")
print("=" * 80)
print("PROCESSING RUNS")
print("=" * 80)

for idx, key in enumerate(sorted(epoch_files.keys()), 1):

    subject, run = key

    source_file = epoch_files[key]

    print("\n" + "=" * 80)
    print(f"[{idx}/{len(epoch_files)}] {os.path.basename(source_file)}")
    print("=" * 80)

    try:

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        epochs = mne.read_epochs(
            source_file,
            preload=True,
            verbose=False
        )

        n_epochs = len(epochs)

        total_input_epochs += n_epochs

        print(f"Input epochs: {n_epochs}")
        print(f"Channels:      {len(epochs.ch_names)}")
        print(f"Time points:   {len(epochs.times)}")
        print(f"Sampling rate: {epochs.info['sfreq']}")

        # ----------------------------------------------------
        # MASTER FOR THIS RUN
        # ----------------------------------------------------

        run_master = master[
            (master[subject_col] == subject) &
            (master[run_col].astype(int) == int(run))
        ].copy()

        if len(run_master) != n_epochs:

            raise ValueError(
                f"Master/file epoch mismatch: "
                f"master={len(run_master)}, "
                f"file={n_epochs}"
            )

        # ----------------------------------------------------
        # SORT BY EPOCH INDEX
        # ----------------------------------------------------

        run_master = run_master.sort_values(
            epoch_col
        ).reset_index(drop=True)

        # Epoch numbering may be 0-based or 1-based.
        # We only use row order after validation.

        # ----------------------------------------------------
        # STATUS MASKS
        # ----------------------------------------------------

        eligible_rows = np.where(
            run_master[status_col].values == "ELIGIBLE"
        )[0]

        eligible_review_rows = np.where(
            run_master[status_col].values == "ELIGIBLE_REVIEW"
        )[0]

        review_rows = np.where(
            run_master[status_col].values == "REVIEW"
        )[0]

        exclude_rows = np.where(
            run_master[status_col].values == "EXCLUDE"
        )[0]

        print("\nSELECTION")
        print("-" * 80)
        print(f"ELIGIBLE:        {len(eligible_rows)}")
        print(f"ELIGIBLE_REVIEW: {len(eligible_review_rows)}")
        print(f"REVIEW:          {len(review_rows)}")
        print(f"EXCLUDE:         {len(exclude_rows)}")

        # ----------------------------------------------------
        # SAVE ELIGIBLE
        # ----------------------------------------------------

        eligible_count = 0
        review_count = 0

        if len(eligible_rows) > 0:

            selected = epochs[eligible_rows]

            output_name = (
                f"{subject}_run-{run:02d}_"
                "perturbation_eligible_epo.fif"
            )

            output_path = os.path.join(
                ELIGIBLE_DIR,
                output_name
            )

            print("\nSaving ELIGIBLE:")
            print(output_path)

            selected.save(
                output_path,
                overwrite=True
            )

            eligible_count = len(selected)

        # ----------------------------------------------------
        # SAVE ELIGIBLE_REVIEW
        # ----------------------------------------------------

        if len(eligible_review_rows) > 0:

            selected_review = epochs[
                eligible_review_rows
            ]

            output_name = (
                f"{subject}_run-{run:02d}_"
                "perturbation_review_epo.fif"
            )

            output_path_review = os.path.join(
                REVIEW_DIR,
                output_name
            )

            print("\nSaving ELIGIBLE_REVIEW:")
            print(output_path_review)

            selected_review.save(
                output_path_review,
                overwrite=True
            )

            review_count = len(selected_review)

        # ----------------------------------------------------
        # MANIFEST
        # ----------------------------------------------------

        for local_index, row in run_master.iterrows():

            status = row[status_col]

            if status not in [
                "ELIGIBLE",
                "ELIGIBLE_REVIEW",
                "REVIEW",
                "EXCLUDE"
            ]:
                continue

            if status == "ELIGIBLE":
                dataset_group = "ELIGIBLE"
            elif status == "ELIGIBLE_REVIEW":
                dataset_group = "ELIGIBLE_REVIEW"
            else:
                dataset_group = "NOT_SELECTED"

            manifest_records.append({
                "subject": subject,
                "run": int(run),
                "epoch_index": int(local_index),
                "master_epoch": row[epoch_col],
                "condition": row[condition_col],
                "perturbation_status": status,
                "recommendation": row[recommendation_col],
                "dataset_group": dataset_group,
                "source_file": os.path.basename(source_file),
            })

        run_summary.append({
            "subject": subject,
            "run": int(run),
            "input_epochs": n_epochs,
            "eligible_epochs": eligible_count,
            "eligible_review_epochs": review_count,
            "review_epochs": len(review_rows),
            "exclude_epochs": len(exclude_rows),
            "sfreq": float(epochs.info["sfreq"]),
            "channels": len(epochs.ch_names),
            "times": len(epochs.times),
            "status": "SUCCESS",
        })

        total_eligible += eligible_count
        total_eligible_review += review_count

        processed_runs += 1

        print("\nSTATUS: SUCCESS")

    except Exception as e:

        failed_runs += 1

        print("\nSTATUS: FAILED")
        print(str(e))

        run_summary.append({
            "subject": subject,
            "run": int(run),
            "input_epochs": np.nan,
            "eligible_epochs": np.nan,
            "eligible_review_epochs": np.nan,
            "review_epochs": np.nan,
            "exclude_epochs": np.nan,
            "sfreq": np.nan,
            "channels": np.nan,
            "times": np.nan,
            "status": "FAILED",
            "error": str(e),
        })

        traceback.print_exc()

# ============================================================
# SAVE MANIFEST
# ============================================================

print("\n")
print("=" * 80)
print("SAVING MASTER MANIFEST")
print("=" * 80)

manifest_df = pd.DataFrame(manifest_records)

manifest_df.to_csv(
    MANIFEST_OUT,
    index=False
)

run_summary_df = pd.DataFrame(
    run_summary
)

run_summary_path = os.path.join(
    OUTPUT_DIR,
    "perturbation_run_summary.csv"
)

run_summary_df.to_csv(
    run_summary_path,
    index=False
)

# ============================================================
# CONDITION SUMMARY
# ============================================================

selected_manifest = manifest_df[
    manifest_df["dataset_group"].isin(
        ["ELIGIBLE", "ELIGIBLE_REVIEW"]
    )
].copy()

condition_summary = (
    selected_manifest
    .groupby(
        ["dataset_group", "condition"]
    )
    .size()
    .reset_index(
        name="epochs"
    )
)

condition_summary_path = os.path.join(
    OUTPUT_DIR,
    "perturbation_condition_summary.csv"
)

condition_summary.to_csv(
    condition_summary_path,
    index=False
)

# ============================================================
# SUBJECT SUMMARY
# ============================================================

subject_summary = (
    selected_manifest
    .groupby(
        ["subject", "dataset_group"]
    )
    .size()
    .reset_index(
        name="epochs"
    )
)

subject_summary_path = os.path.join(
    OUTPUT_DIR,
    "perturbation_subject_summary.csv"
)

subject_summary.to_csv(
    subject_summary_path,
    index=False
)

# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 80)
print("PERTURBATION DATASET VALIDATION")
print("=" * 80)

print(f"\nInput epochs:           {total_input_epochs}")
print(f"ELIGIBLE epochs:       {total_eligible}")
print(f"ELIGIBLE_REVIEW:       {total_eligible_review}")
print(f"Processed runs:        {processed_runs}")
print(f"Failed runs:            {failed_runs}")

print("\nEXPECTED FROM MASTER")
print("-" * 80)
print(
    "ELIGIBLE:",
    int((master[status_col] == "ELIGIBLE").sum())
)

print(
    "ELIGIBLE_REVIEW:",
    int((master[status_col] == "ELIGIBLE_REVIEW").sum())
)

print("\nMANIFEST")
print("-" * 80)
print(f"Total manifest records: {len(manifest_df)}")

print("\nDATASET GROUPS")
print("-" * 80)
print(
    manifest_df["dataset_group"].value_counts()
)

# ============================================================
# WRITE SUMMARY
# ============================================================

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "PERTURBATION DATASET SUMMARY\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Input epochs: {total_input_epochs}\n"
    )

    f.write(
        f"ELIGIBLE epochs: {total_eligible}\n"
    )

    f.write(
        f"ELIGIBLE_REVIEW epochs: "
        f"{total_eligible_review}\n"
    )

    f.write(
        f"Processed runs: {processed_runs}\n"
    )

    f.write(
        f"Failed runs: {failed_runs}\n\n"
    )

    f.write(
        "DATASET GROUP COUNTS\n"
    )

    f.write(
        str(
            manifest_df[
                "dataset_group"
            ].value_counts()
        )
    )

    f.write("\n\nCONDITION COUNTS\n")

    f.write(
        str(
            selected_manifest[
                "condition"
            ].value_counts()
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "RAW DATA WAS NOT MODIFIED.\n"
    )

    f.write(
        "NO SET FILE WAS MODIFIED.\n"
    )

    f.write(
        "NO FDT FILE WAS MODIFIED.\n"
    )

    f.write(
        "NO EPOCHS_CLEAN FILE WAS MODIFIED.\n"
    )

# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("PERTURBATION DATASET COMPLETE")
print("=" * 80)

print(f"\nELIGIBLE epochs:       {total_eligible}")
print(f"ELIGIBLE_REVIEW:       {total_eligible_review}")
print(f"Failed runs:            {failed_runs}")

print("\nOUTPUT")
print("-" * 80)
print(f"Main dataset:")
print(ELIGIBLE_DIR)

print("\nReview dataset:")
print(REVIEW_DIR)

print("\nManifest:")
print(MANIFEST_OUT)

print("\nRun summary:")
print(run_summary_path)

print("\nCondition summary:")
print(condition_summary_path)

print("\nSubject summary:")
print(subject_summary_path)

print("\nSummary:")
print(SUMMARY_OUT)

print("\n")
print("=" * 80)
print("NO ORIGINAL DATA WAS MODIFIED.")
print("=" * 80)