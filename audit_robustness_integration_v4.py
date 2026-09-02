from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "robustness_integration_v4"
    / "perturbation_robustness_integration_v4.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "robustness_integration_v4"
    / "audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_COLUMNS = OUTPUT_DIR / "v4_nan_column_audit.csv"
OUTPUT_ROWS = OUTPUT_DIR / "v4_nan_row_audit.csv"
OUTPUT_QC = OUTPUT_DIR / "v4_nan_audit_qc.csv"

# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("ROBUSTNESS INTEGRATION V4 - NaN AUDIT")
print("=" * 80)

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT}")

df = pd.read_csv(INPUT)

print(f"Rows:       {len(df):,}")
print(f"Columns:    {len(df):,}")

# ============================================================
# NUMERIC COLUMNS
# ============================================================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print()
print("=" * 80)
print("NUMERIC QC")
print("=" * 80)

print(f"Numeric columns: {len(numeric_cols)}")

nan_counts = df[numeric_cols].isna().sum()
inf_counts = np.isinf(df[numeric_cols]).sum()

nan_columns = (
    nan_counts[nan_counts > 0]
    .sort_values(ascending=False)
)

inf_columns = (
    inf_counts[inf_counts > 0]
    .sort_values(ascending=False)
)

print()
print("Columns containing NaN:")
if len(nan_columns) == 0:
    print("None")
else:
    print(nan_columns.to_string())

print()
print("Columns containing Inf:")
if len(inf_columns) == 0:
    print("None")
else:
    print(inf_columns.to_string())

# ============================================================
# COLUMN-LEVEL AUDIT
# ============================================================

column_audit = pd.DataFrame({
    "column": numeric_cols,
    "nan_count": [nan_counts[c] for c in numeric_cols],
    "nan_percent": [
        100 * nan_counts[c] / len(df)
        for c in numeric_cols
    ],
    "inf_count": [inf_counts[c] for c in numeric_cols],
})

column_audit["status"] = np.where(
    column_audit["nan_count"] > 0,
    "HAS_NAN",
    "OK"
)

column_audit = column_audit.sort_values(
    ["nan_count", "column"],
    ascending=[False, True]
)

# ============================================================
# ROW-LEVEL AUDIT
# ============================================================

df["_nan_count"] = df[numeric_cols].isna().sum(axis=1)
df["_inf_count"] = np.isinf(df[numeric_cols]).sum(axis=1)

row_audit = df.loc[
    (df["_nan_count"] > 0) |
    (df["_inf_count"] > 0)
].copy()

print()
print("=" * 80)
print("ROW-LEVEL NAN AUDIT")
print("=" * 80)

print(f"Rows with NaN/Inf: {len(row_audit):,}")

if len(row_audit) > 0:

    display_cols = []

    for c in [
        "target",
        "feature",
        "effect_magnitude",
        "direction",
        "robustness_classification",
        "robustness_status",
    ]:
        if c in row_audit.columns:
            display_cols.append(c)

    display_cols += ["_nan_count", "_inf_count"]

    print()
    print(row_audit[display_cols].to_string(index=False))

# ============================================================
# TARGET / FEATURE AUDIT
# ============================================================

print()
print("=" * 80)
print("TARGET / FEATURE COVERAGE")
print("=" * 80)

if "target" in df.columns:
    print("\nTargets:")
    print(df["target"].value_counts(dropna=False).to_string())

if "feature" in df.columns:
    print("\nFeatures:")
    print(f"Unique features: {df['feature'].nunique()}")

# ============================================================
# EXACT NAN LOCATIONS
# ============================================================

nan_locations = []

for idx, row in df.iterrows():

    for col in numeric_cols:

        if pd.isna(row[col]):

            record = {
                "row_index": idx,
                "target": row["target"] if "target" in df.columns else None,
                "feature": row["feature"] if "feature" in df.columns else None,
                "column": col,
            }

            nan_locations.append(record)

nan_locations_df = pd.DataFrame(nan_locations)

# ============================================================
# SUMMARY
# ============================================================

total_nan = int(df[numeric_cols].isna().sum().sum())
total_inf = int(np.isinf(df[numeric_cols]).sum().sum())

print()
print("=" * 80)
print("FINAL NAN AUDIT SUMMARY")
print("=" * 80)

print(f"Total rows:             {len(df):,}")
print(f"Numeric columns:        {len(numeric_cols):,}")
print(f"Total NaN cells:        {total_nan:,}")
print(f"Total Inf cells:        {total_inf:,}")
print(f"Rows with NaN/Inf:      {len(row_audit):,}")
print(f"Columns with NaN:       {len(nan_columns):,}")

if total_nan == 0 and total_inf == 0:
    status = "PASS"
else:
    status = "REVIEW_REQUIRED"

# ============================================================
# SAVE
# ============================================================

column_audit.to_csv(
    OUTPUT_COLUMNS,
    index=False
)

row_audit.drop(
    columns=["_nan_count", "_inf_count"],
    errors="ignore"
).to_csv(
    OUTPUT_ROWS,
    index=False
)

qc = pd.DataFrame([{
    "rows": len(df),
    "columns": len(df.columns) - 2,
    "numeric_columns": len(numeric_cols),
    "total_nan_cells": total_nan,
    "total_inf_cells": total_inf,
    "rows_with_nan_or_inf": len(row_audit),
    "columns_with_nan": len(nan_columns),
    "status": status
}])

qc.to_csv(
    OUTPUT_QC,
    index=False
)

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_COLUMNS)
print(OUTPUT_ROWS)
print(OUTPUT_QC)

print()
print("=" * 80)
print(f"STATUS: {status}")
print("=" * 80)