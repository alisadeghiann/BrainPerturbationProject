# ============================================================
# FIND THE SINGLE REVIEW RUN IN V5
# ============================================================

from pathlib import Path
import pandas as pd

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

AUDIT_DIR = (
    BASE_DIR
    / "qc"
    / "behavioral_mapping_audit_v5"
)

RUN_AUDIT = (
    AUDIT_DIR
    / "run_level_mapping_audit_v5.csv"
)

CONDITION_AUDIT = (
    AUDIT_DIR
    / "condition_summary_v5.csv"
)

# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("V5 REVIEW RUN INVESTIGATION")
print("=" * 100)
print()

print("Project:")
print(BASE_DIR)
print()

print("Run-level audit:")
print(RUN_AUDIT)
print()

# ============================================================
# CHECK FILE
# ============================================================

if not RUN_AUDIT.exists():
    raise FileNotFoundError(
        f"Run-level audit file not found:\n{RUN_AUDIT}"
    )

# ============================================================
# READ RUN AUDIT
# ============================================================

df = pd.read_csv(RUN_AUDIT)

print("Columns:")
print(list(df.columns))
print()

print("Total rows:")
print(len(df))
print()

# ============================================================
# SHOW COMPLETE RUN AUDIT
# ============================================================

print("=" * 100)
print("COMPLETE RUN-LEVEL AUDIT")
print("=" * 100)
print()

print(df.to_string(index=False))
print()

# ============================================================
# FIND NON-PASS RUNS
# ============================================================

possible_status_columns = [
    "status",
    "final_status",
    "mapping_status",
    "qc_status",
]

status_col = None

for col in possible_status_columns:
    if col in df.columns:
        status_col = col
        break

print("=" * 100)
print("STATUS ANALYSIS")
print("=" * 100)
print()

if status_col is not None:

    print(f"Status column detected: {status_col}")
    print()

    print(
        df[status_col]
        .value_counts(dropna=False)
        .to_string()
    )

    print()

    review_df = df[
        df[status_col]
        .astype(str)
        .str.upper()
        != "PASS"
    ]

else:

    print(
        "No standard status column detected."
    )
    print()

    print(
        "Please inspect the complete table above."
    )

    review_df = pd.DataFrame()

# ============================================================
# PRINT REVIEW RUNS
# ============================================================

print("=" * 100)
print("RUN(S) REQUIRING REVIEW")
print("=" * 100)
print()

if len(review_df) == 0:

    print(
        "No non-PASS run was detected from the available "
        "status column."
    )

else:

    print(
        f"Number of review runs: {len(review_df)}"
    )
    print()

    print(
        review_df.to_string(index=False)
    )

print()

# ============================================================
# LOOK FOR IMPORTANT NUMERICAL ANOMALIES
# ============================================================

print("=" * 100)
print("POSSIBLE ANOMALIES")
print("=" * 100)
print()

numeric_checks = [
    "epochs",
    "BIDS_events",
    "bids_events",
    "matched",
    "unmatched",
    "remember",
    "ignore",
    "probe_target",
    "probe_not_shown",
    "work_memory",
]

for col in numeric_checks:

    if col in df.columns:

        print()
        print(f"--- {col} ---")
        print(
            df[col]
            .describe()
            .to_string()
        )

# ============================================================
# SHOW RUNS WITH UNMATCHED EVENTS
# ============================================================

print()
print("=" * 100)
print("UNMATCHED EVENTS")
print("=" * 100)
print()

unmatched_col = None

for col in ["unmatched", "unmatched_events", "n_unmatched"]:
    if col in df.columns:
        unmatched_col = col
        break

if unmatched_col:

    bad_unmatched = df[
        pd.to_numeric(
            df[unmatched_col],
            errors="coerce"
        ).fillna(0) > 0
    ]

    if len(bad_unmatched) == 0:
        print("NO UNMATCHED EVENTS FOUND.")
    else:
        print(
            bad_unmatched.to_string(index=False)
        )

else:

    print(
        "No unmatched column found."
    )

# ============================================================
# CONDITION SUMMARY
# ============================================================

print()
print("=" * 100)
print("CONDITION SUMMARY")
print("=" * 100)
print()

if CONDITION_AUDIT.exists():

    cond = pd.read_csv(
        CONDITION_AUDIT
    )

    print(
        cond.to_string(index=False)
    )

else:

    print(
        "Condition summary file not found:"
    )
    print(CONDITION_AUDIT)

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print("INVESTIGATION COMPLETE")
print("=" * 100)
print()

print(
    "IMPORTANT:"
)
print(
    "This script is READ-ONLY."
)
print(
    "No EEG/FIF/TSV/CSV files were modified."
)
print(
    "No files were deleted."
)
print(
    "No files were regenerated."
)
print()

print(
    "NEXT STEP:"
)
print(
    "Send me the COMPLETE output of this script."
)
print(
    "We will identify exactly why 1 of the 82 runs "
    "is not PASS before changing anything."
)
print()