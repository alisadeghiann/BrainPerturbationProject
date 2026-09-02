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

print("=" * 80)
print("V4 NaN SOURCE DIAGNOSTIC")
print("=" * 80)

df = pd.read_csv(V4)

# ------------------------------------------------------------
# EXACT NaN COLUMNS
# ------------------------------------------------------------

numeric_cols = df.select_dtypes(include=[np.number]).columns

nan_cols = [
    col for col in numeric_cols
    if df[col].isna().all()
]

print()
print("=" * 80)
print("COLUMNS THAT ARE 100% NaN")
print("=" * 80)

print(f"Count: {len(nan_cols)}")

for col in nan_cols:
    print(f"  {col}")

# ------------------------------------------------------------
# ALL NUMERIC COLUMNS
# ------------------------------------------------------------

print()
print("=" * 80)
print("NUMERIC COLUMN STATUS")
print("=" * 80)

for col in numeric_cols:

    n_nan = int(df[col].isna().sum())
    n_valid = int(df[col].notna().sum())

    print(
        f"{col:45s} "
        f"valid={n_valid:4d} "
        f"NaN={n_nan:4d}"
    )

# ------------------------------------------------------------
# V3 CHECK
# ------------------------------------------------------------

print()
print("=" * 80)
print("V3 SOURCE CHECK")
print("=" * 80)

if not V3.exists():

    print("V3 FILE NOT FOUND")
    print(V3)

else:

    v3 = pd.read_csv(V3)

    print(f"V3 rows: {len(v3)}")

    print()
    print("NaN columns vs V3:")

    for col in nan_cols:

        if col in v3.columns:

            valid = int(v3[col].notna().sum())
            nan = int(v3[col].isna().sum())

            print(
                f"{col:45s} "
                f"EXISTS IN V3 | "
                f"valid={valid} | NaN={nan}"
            )

        else:

            print(
                f"{col:45s} "
                f"NOT FOUND IN V3"
            )

# ------------------------------------------------------------
# ROBUSTNESS-RELATED COLUMN NAMES
# ------------------------------------------------------------

print()
print("=" * 80)
print("ROBUSTNESS-RELATED COLUMNS")
print("=" * 80)

keywords = [
    "robust",
    "agreement",
    "stability",
    "consistency",
    "subject",
    "direction",
    "classification",
    "score"
]

for col in df.columns:

    name = col.lower()

    if any(k in name for k in keywords):

        print(col)

# ------------------------------------------------------------
# SAMPLE
# ------------------------------------------------------------

print()
print("=" * 80)
print("SAMPLE OF NaN COLUMNS")
print("=" * 80)

if nan_cols:

    sample_cols = [
        c for c in
        ["target", "feature"] + nan_cols
        if c in df.columns
    ]

    print(
        df[sample_cols]
        .head(10)
        .to_string(index=False)
    )

print()
print("=" * 80)
print("STATUS: SOURCE DIAGNOSTIC COMPLETE")
print("=" * 80)