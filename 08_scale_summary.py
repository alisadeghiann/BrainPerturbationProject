import os
import pandas as pd
import numpy as np


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

REPORT_FILE = os.path.join(
    PROJECT_ROOT,
    "qc",
    "subject_scale_inspection",
    "subject_scale_file_summary.csv"
)


# ============================================================
# READ REPORT
# ============================================================

print("=" * 80)
print("FINAL SCALE SUMMARY")
print("=" * 80)

df = pd.read_csv(
    REPORT_FILE
)


# ============================================================
# SHOW ALL FILES
# ============================================================

print()
print("=" * 80)
print("ALL INSPECTED FILES")
print("=" * 80)

columns = [
    "subject",
    "file",
    "sampling_rate",
    "median_std",
    "max_std",
    "median_ptp",
    "max_ptp",
    "min_value",
    "max_value",
    "channels_over_5x",
    "channels_over_10x"
]

available_columns = [
    c for c in columns
    if c in df.columns
]

print(
    df[available_columns].to_string(
        index=False
    )
)


# ============================================================
# MOST SUSPICIOUS FILES
# ============================================================

print()
print("=" * 80)
print("MOST SUSPICIOUS FILES")
print("=" * 80)

print()

suspicious = df.sort_values(
    [
        "channels_over_10x",
        "channels_over_5x",
        "median_std"
    ],
    ascending=False
)

print(
    suspicious[
        available_columns
    ].head(15).to_string(
        index=False
    )
)


# ============================================================
# SUB-004 SPECIFIC
# ============================================================

print()
print("=" * 80)
print("SUB-004 INSPECTION")
print("=" * 80)

print()

sub004 = df[
    df["subject"] == "sub-004"
]

if len(sub004) == 0:

    print(
        "sub-004 was not found in the report."
    )

else:

    print(
        sub004[
            available_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# SUB-003 SPECIFIC
# ============================================================

print()
print("=" * 80)
print("SUB-003 INSPECTION")
print("=" * 80)

print()

sub003 = df[
    df["subject"] == "sub-003"
]

if len(sub003) == 0:

    print(
        "sub-003 was not found in the report."
    )

else:

    print(
        sub003[
            available_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# SUB-008 SPECIFIC
# ============================================================

print()
print("=" * 80)
print("SUB-008 INSPECTION")
print("=" * 80)

print()

sub008 = df[
    df["subject"] == "sub-008"
]

if len(sub008) == 0:

    print(
        "sub-008 was not found in the report."
    )

else:

    print(
        sub008[
            available_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("SUMMARY COMPLETE")
print("=" * 80)

print()
print(
    "Report used:"
)

print(
    REPORT_FILE
)

print()
print(
    "DONE."
)