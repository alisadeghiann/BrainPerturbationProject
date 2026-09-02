from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

V4 = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "robustness_integration_v4"
    / "perturbation_robustness_integration_v4.csv"
)

V3 = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "robustness_v3"
    / "subject_level_perturbation_robustness_v3.csv"
)

OUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "robustness_integration_v4"
    / "audit"
    / "v4_nan_exact_diagnosis.csv"
)

print("=" * 80)
print("V4 ROBUSTNESS - EXACT NaN DIAGNOSIS")
print("=" * 80)

# ============================================================
# LOAD V4
# ============================================================

if not V4.exists():
    raise FileNotFoundError(f"V4 file not found:\n{V4}")

df = pd.read_csv(V4)

print(f"V4 rows:    {len(df)}")
print(f"V4 columns: {len(df.columns)}")

# ============================================================
# FIND NaN COLUMNS
# ============================================================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

nan_summary = []

for col in numeric_cols:

    n_nan = int(df[col].isna().sum())

    if n_nan > 0:

        nan_summary.append({
            "column": col,
            "nan_count": n_nan,
            "total_rows": len(df),
            "nan_percent": 100 * n_nan / len(df),
            "unique_non_nan": df[col].dropna().nunique()
        })

nan_df = pd.DataFrame(nan_summary)

print()
print("=" * 80)
print("EXACT NaN COLUMNS")
print("=" * 80)

if len(nan_df) == 0:

    print("No NaN numeric columns found.")

else:

    print(nan_df.to_string(index=False))

# ============================================================
# CHECK ALL VALUES FOR EACH NaN COLUMN
# ============================================================

if len(nan_df) > 0:

    for col in nan_df["column"]:

        print()
        print("-" * 80)
        print(f"COLUMN: {col}")
        print("-" * 80)

        print(
            f"NaN: {df[col].isna().sum()} / {len(df)}"
        )

        print(
            f"Non-NaN: {df[col].notna().sum()} / {len(df)}"
        )

# ============================================================
# V3 COMPARISON
# ============================================================

print()
print("=" * 80)
print("V3 COMPARISON")
print("=" * 80)

if not V3.exists():

    print("V3 file not found:")
    print(V3)

else:

    v3 = pd.read_csv(V3)

    print(f"V3 rows:    {len(v3)}")
    print(f"V3 columns: {len(v3.columns)}")

    print()
    print("Potential matching columns:")

    for col in nan_df["column"] if len(nan_df) > 0 else []:

        if col in v3.columns:

            print(f"{col} -> EXISTS in V3")

        else:

            print(f"{col} -> NOT FOUND in V3")

    print()
    print("V3 columns:")

    print(v3.columns.tolist())

# ============================================================
# TARGET / FEATURE STRUCTURE
# ============================================================

print()
print("=" * 80)
print("V4 STRUCTURE")
print("=" * 80)

for col in [
    "target",
    "feature",
    "effect_magnitude",
    "direction",
    "robustness_classification",
    "robustness_status"
]:

    if col in df.columns:

        print()
        print(f"{col}:")
        print(df[col].value_counts(dropna=False).head(20).to_string())

# ============================================================
# SAVE
# ============================================================

nan_df.to_csv(
    OUT,
    index=False
)

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUT)

print()
print("=" * 80)
print("STATUS: DIAGNOSIS COMPLETE")
print("=" * 80)