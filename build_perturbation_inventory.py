
from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# PERTURBATION DATA INVENTORY + ALIGNMENT QC
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

PERTURBATION_DIR = BASE / "final_dataset" / "perturbation"

OUTPUT_DIR = (
    BASE
    / "features"
    / "ml_results"
    / "perturbation_qc"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PERTURBATION DATA INVENTORY + ALIGNMENT QC")
print("=" * 80)

# =============================================================================
# FIND FILES
# =============================================================================

if not PERTURBATION_DIR.exists():
    raise RuntimeError(
        f"Perturbation directory does not exist:\n{PERTURBATION_DIR}"
    )

files = sorted(
    PERTURBATION_DIR.rglob("*")
)

files = [
    f for f in files
    if f.is_file()
]

print(f"Perturbation directory: {PERTURBATION_DIR}")
print(f"Files found:             {len(files)}")

if len(files) == 0:
    raise RuntimeError("No files found in perturbation directory.")

# =============================================================================
# FILE INVENTORY
# =============================================================================

inventory_rows = []

for f in files:

    suffix = f.suffix.lower()

    try:
        size_mb = f.stat().st_size / (1024 * 1024)
    except Exception:
        size_mb = np.nan

    inventory_rows.append(
        {
            "file": f.name,
            "relative_path": str(f.relative_to(PERTURBATION_DIR)),
            "suffix": suffix,
            "size_mb": round(size_mb, 4),
        }
    )

inventory = pd.DataFrame(inventory_rows)

print()
print("=" * 80)
print("FILE TYPES")
print("=" * 80)

print(
    inventory["suffix"]
    .value_counts()
    .sort_index()
)

print()
print("=" * 80)
print("FILE INVENTORY SAMPLE")
print("=" * 80)

print(
    inventory.head(30).to_string(index=False)
)

# =============================================================================
# FILENAME STRUCTURE ANALYSIS
# =============================================================================

print()
print("=" * 80)
print("FILENAME STRUCTURE")
print("=" * 80)

filename_records = []

for f in files:

    name = f.name

    subject = None
    run = None
    perturbation = None

    # subject
    parts = name.split("_")

    for part in parts:
        if part.startswith("sub-"):
            subject = part

        if part.startswith("run-"):
            run = part

    # perturbation keywords
    lower = name.lower()

    if "perturb" in lower:
        perturbation = "perturbation"

    elif "control" in lower:
        perturbation = "control"

    elif "baseline" in lower:
        perturbation = "baseline"

    filename_records.append(
        {
            "file": name,
            "subject": subject,
            "run": run,
            "perturbation_hint": perturbation,
            "suffix": f.suffix.lower(),
        }
    )

filename_df = pd.DataFrame(filename_records)

print(
    filename_df.head(30).to_string(index=False)
)

# =============================================================================
# SUBJECT COVERAGE
# =============================================================================

print()
print("=" * 80)
print("SUBJECT COVERAGE")
print("=" * 80)

subject_counts = (
    filename_df["subject"]
    .value_counts(dropna=False)
    .sort_index()
)

print(subject_counts)

# =============================================================================
# RUN COVERAGE
# =============================================================================

print()
print("=" * 80)
print("RUN COVERAGE")
print("=" * 80)

run_counts = (
    filename_df["run"]
    .value_counts(dropna=False)
    .sort_index()
)

print(run_counts)

# =============================================================================
# POSSIBLE PERTURBATION LABELS
# =============================================================================

print()
print("=" * 80)
print("PERTURBATION LABEL HINTS")
print("=" * 80)

print(
    filename_df["perturbation_hint"]
    .value_counts(dropna=False)
)

# =============================================================================
# SEARCH FOR TABULAR METADATA
# =============================================================================

tabular_files = [
    f for f in files
    if f.suffix.lower() in [".csv", ".tsv", ".xlsx", ".xls"]
]

print()
print("=" * 80)
print("TABULAR METADATA FILES")
print("=" * 80)

print(f"Tabular files: {len(tabular_files)}")

for f in tabular_files:
    print(f" - {f.relative_to(PERTURBATION_DIR)}")

# =============================================================================
# READ CSV / TSV METADATA IF AVAILABLE
# =============================================================================

metadata_summaries = []

for f in tabular_files:

    if f.suffix.lower() == ".csv":

        try:
            df = pd.read_csv(f)

        except Exception as e:

            print()
            print(f"Could not read CSV: {f.name}")
            print(f"Reason: {e}")
            continue

    elif f.suffix.lower() == ".tsv":

        try:
            df = pd.read_csv(f, sep="\t")

        except Exception as e:

            print()
            print(f"Could not read TSV: {f.name}")
            print(f"Reason: {e}")
            continue

    else:
        continue

    print()
    print("-" * 80)
    print(f"METADATA FILE: {f.name}")
    print("-" * 80)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("Columns:")
    print(list(df.columns))

    metadata_summaries.append(
        {
            "file": f.name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": "|".join(map(str, df.columns)),
        }
    )

# =============================================================================
# EEG FILES
# =============================================================================

eeg_files = [
    f for f in files
    if f.suffix.lower() in [".fif", ".set", ".edf", ".bdf", ".vhdr"]
]

print()
print("=" * 80)
print("EEG FILES")
print("=" * 80)

print(f"EEG files: {len(eeg_files)}")

for f in eeg_files[:50]:
    print(f" - {f.relative_to(PERTURBATION_DIR)}")

if len(eeg_files) > 50:
    print(f"... and {len(eeg_files) - 50} more")

# =============================================================================
# CHECK EXPECTED FINAL EPOCH FILES
# =============================================================================

final_epoch_files = [
    f for f in eeg_files
    if "_final_epo.fif" in f.name
]

print()
print("=" * 80)
print("FINAL EPOCH FILE CHECK")
print("=" * 80)

print(f"*_final_epo.fif files: {len(final_epoch_files)}")

if final_epoch_files:
    print("Sample:")
    for f in final_epoch_files[:20]:
        print(f" - {f.name}")

# =============================================================================
# DUPLICATE FILENAME CHECK
# =============================================================================

print()
print("=" * 80)
print("DUPLICATE FILE CHECK")
print("=" * 80)

duplicate_names = (
    inventory["file"]
    .duplicated(keep=False)
)

duplicate_count = int(duplicate_names.sum())

print(f"Duplicate filenames: {duplicate_count}")

if duplicate_count > 0:
    print(
        inventory.loc[
            duplicate_names
        ].sort_values("file").to_string(index=False)
    )

# =============================================================================
# MISSING SUBJECT / RUN PARSING
# =============================================================================

print()
print("=" * 80)
print("PARSING QC")
print("=" * 80)

missing_subject = int(filename_df["subject"].isna().sum())
missing_run = int(filename_df["run"].isna().sum())

print(f"Files without subject ID: {missing_subject}")
print(f"Files without run ID:     {missing_run}")

if missing_subject > 0:
    print()
    print("Files with missing subject:")
    print(
        filename_df.loc[
            filename_df["subject"].isna(),
            "file"
        ].head(50).to_string(index=False)
    )

if missing_run > 0:
    print()
    print("Files with missing run:")
    print(
        filename_df.loc[
            filename_df["run"].isna(),
            "file"
        ].head(50).to_string(index=False)
    )

# =============================================================================
# SAVE INVENTORY
# =============================================================================

inventory_path = OUTPUT_DIR / "perturbation_file_inventory.csv"

filename_path = OUTPUT_DIR / "perturbation_filename_structure.csv"

metadata_path = OUTPUT_DIR / "perturbation_metadata_inventory.csv"

inventory.to_csv(
    inventory_path,
    index=False
)

filename_df.to_csv(
    filename_path,
    index=False
)

pd.DataFrame(
    metadata_summaries
).to_csv(
    metadata_path,
    index=False
)

# =============================================================================
# QC SUMMARY
# =============================================================================

qc = {
    "perturbation_directory": str(PERTURBATION_DIR),
    "total_files": len(files),
    "eeg_files": len(eeg_files),
    "final_epoch_files": len(final_epoch_files),
    "tabular_files": len(tabular_files),
    "subjects_detected": filename_df["subject"].nunique(dropna=True),
    "runs_detected": filename_df["run"].nunique(dropna=True),
    "duplicate_filenames": duplicate_count,
    "missing_subject": missing_subject,
    "missing_run": missing_run,
}

qc_df = pd.DataFrame(
    [qc]
)

qc_path = OUTPUT_DIR / "perturbation_inventory_qc.csv"

qc_df.to_csv(
    qc_path,
    index=False
)

# =============================================================================
# FINAL STATUS
# =============================================================================

print()
print("=" * 80)
print("PERTURBATION INVENTORY COMPLETE")
print("=" * 80)

print(f"Total files:          {len(files)}")
print(f"EEG files:            {len(eeg_files)}")
print(f"Final epoch files:    {len(final_epoch_files)}")
print(f"Subjects detected:    {filename_df['subject'].nunique(dropna=True)}")
print(f"Runs detected:        {filename_df['run'].nunique(dropna=True)}")
print(f"Duplicate filenames:  {duplicate_count}")
print(f"Missing subject:      {missing_subject}")
print(f"Missing run:          {missing_run}")

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(inventory_path)
print(filename_path)
print(metadata_path)
print(qc_path)

print()
print("=" * 80)
print("STATUS: INVENTORY COMPLETE - REVIEW OUTPUT BEFORE PERTURBATION MODELING")
print("=" * 80)