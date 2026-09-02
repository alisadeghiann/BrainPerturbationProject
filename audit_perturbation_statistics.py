from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# PERTURBATION STATISTICAL AUDIT
# =============================================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "statistical"
    / "perturbation_statistical_results.csv"
)

OUTPUT_DIR = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "statistical"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_FILE = OUTPUT_DIR / "perturbation_statistical_audit.csv"
QC_FILE = OUTPUT_DIR / "perturbation_statistical_audit_qc.csv"


print("=" * 80)
print("PERTURBATION STATISTICAL AUDIT")
print("=" * 80)

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT}")

df = pd.read_csv(INPUT)

print(f"Input rows:    {len(df):,}")
print(f"Input columns: {len(df.columns):,}")

# =============================================================================
# BASIC INFORMATION
# =============================================================================

print("\n" + "=" * 80)
print("COLUMN INVENTORY")
print("=" * 80)

print(list(df.columns))

# =============================================================================
# NUMERIC COLUMNS
# =============================================================================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\n" + "=" * 80)
print("NUMERIC QC")
print("=" * 80)

nan_counts = df[numeric_cols].isna().sum()
inf_counts = np.isinf(
    df[numeric_cols].to_numpy(dtype=float, copy=True)
).sum()

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Total NaN cells: {int(nan_counts.sum())}")
print(f"Total Inf cells: {int(inf_counts)}")

if nan_counts.sum() > 0:
    print("\nColumns containing NaN:")
    print(
        nan_counts[nan_counts > 0]
        .sort_values(ascending=False)
        .to_string()
    )

# =============================================================================
# IDENTIFY EFFECT / STATISTICAL COLUMNS
# =============================================================================

keywords = [
    "effect",
    "delta",
    "change",
    "difference",
    "p_value",
    "pvalue",
    "q_value",
    "fdr",
    "corrected",
    "cohen",
    "t_stat",
    "statistic",
    "robust",
    "median",
]

effect_cols = []

for col in numeric_cols:
    name = col.lower()
    if any(k in name for k in keywords):
        effect_cols.append(col)

print("\n" + "=" * 80)
print("CANDIDATE STATISTICAL COLUMNS")
print("=" * 80)

print(f"Candidate columns: {len(effect_cols)}")

for col in effect_cols:
    print(col)

# =============================================================================
# COLUMN-LEVEL QC
# =============================================================================

audit_rows = []

for col in effect_cols:

    series = pd.to_numeric(df[col], errors="coerce")

    n_total = len(series)
    n_valid = int(series.notna().sum())
    n_nan = int(series.isna().sum())

    finite = series.replace([np.inf, -np.inf], np.nan)

    n_finite = int(finite.notna().sum())

    if n_finite > 0:
        mean_val = float(finite.mean())
        median_val = float(finite.median())
        std_val = float(finite.std())
        min_val = float(finite.min())
        max_val = float(finite.max())
    else:
        mean_val = np.nan
        median_val = np.nan
        std_val = np.nan
        min_val = np.nan
        max_val = np.nan

    audit_rows.append(
        {
            "column": col,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_nan": n_nan,
            "n_finite": n_finite,
            "nan_percent": 100 * n_nan / n_total if n_total else np.nan,
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
        }
    )

audit = pd.DataFrame(audit_rows)

# =============================================================================
# PRINT NaN DIAGNOSIS
# =============================================================================

print("\n" + "=" * 80)
print("NaN DIAGNOSIS")
print("=" * 80)

nan_audit = audit[audit["n_nan"] > 0].copy()

if len(nan_audit) == 0:
    print("PASS - No NaN values in candidate statistical columns.")
else:
    print(
        nan_audit[
            [
                "column",
                "n_total",
                "n_valid",
                "n_nan",
                "nan_percent",
            ]
        ].to_string(index=False)
    )

# =============================================================================
# P-VALUE / FDR AUDIT
# =============================================================================

print("\n" + "=" * 80)
print("P-VALUE / FDR AUDIT")
print("=" * 80)

p_cols = [
    c for c in numeric_cols
    if any(x in c.lower() for x in ["p_value", "pvalue", "pval"])
]

fdr_cols = [
    c for c in numeric_cols
    if any(x in c.lower() for x in ["fdr", "q_value", "qvalue", "adjusted"])
]

print("P-value columns:")
print(p_cols)

print("\nFDR / adjusted columns:")
print(fdr_cols)

for col in p_cols + fdr_cols:

    s = pd.to_numeric(df[col], errors="coerce")

    valid = s.dropna()

    if len(valid) == 0:
        print(f"\n{col}: NO VALID VALUES")
        continue

    print(f"\n{col}")
    print(f"  valid: {len(valid)}")
    print(f"  min:   {valid.min():.8g}")
    print(f"  max:   {valid.max():.8g}")

    if "p" in col.lower() or "fdr" in col.lower() or "q_" in col.lower():
        print(f"  <0.05: {(valid < 0.05).sum()}")

# =============================================================================
# SIGNIFICANCE AUDIT
# =============================================================================

print("\n" + "=" * 80)
print("SIGNIFICANCE AUDIT")
print("=" * 80)

significant_summary = []

for col in p_cols + fdr_cols:

    s = pd.to_numeric(df[col], errors="coerce")

    valid = s.dropna()

    if len(valid) == 0:
        continue

    significant_summary.append(
        {
            "column": col,
            "n_valid": len(valid),
            "n_below_0_05": int((valid < 0.05).sum()),
            "min_value": float(valid.min()),
            "median_value": float(valid.median()),
        }
    )

sig_df = pd.DataFrame(significant_summary)

if len(sig_df):
    print(sig_df.to_string(index=False))
else:
    print("No valid p-value/FDR columns detected.")

# =============================================================================
# EFFECT SIZE AUDIT
# =============================================================================

effect_size_cols = [
    c for c in numeric_cols
    if any(
        x in c.lower()
        for x in [
            "effect_size",
            "cohen",
            "cliff",
            "hedges",
            "mean_difference",
            "median_difference",
        ]
    )
]

print("\n" + "=" * 80)
print("EFFECT SIZE AUDIT")
print("=" * 80)

if effect_size_cols:
    for col in effect_size_cols:

        s = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(s) == 0:
            print(f"{col}: NO VALID VALUES")
            continue

        print(
            f"{col}: "
            f"mean={s.mean():.6f}, "
            f"median={s.median():.6f}, "
            f"min={s.min():.6f}, "
            f"max={s.max():.6f}"
        )
else:
    print("No explicit effect-size columns detected.")

# =============================================================================
# FINAL STATUS
# =============================================================================

total_nan = int(df[numeric_cols].isna().sum().sum())

if total_nan == 0:
    status = "PASS"
else:
    status = "REVIEW_REQUIRED"

qc = pd.DataFrame(
    [
        {
            "input_rows": len(df),
            "input_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "candidate_statistical_columns": len(effect_cols),
            "total_nan": total_nan,
            "total_inf": int(inf_counts),
            "p_value_columns": len(p_cols),
            "fdr_columns": len(fdr_cols),
            "effect_size_columns": len(effect_size_cols),
            "status": status,
        }
    ]
)

# =============================================================================
# SAVE
# =============================================================================

audit.to_csv(AUDIT_FILE, index=False)
qc.to_csv(QC_FILE, index=False)

print("\n" + "=" * 80)
print("PERTURBATION STATISTICAL AUDIT COMPLETE")
print("=" * 80)

print(f"Statistical columns audited: {len(effect_cols)}")
print(f"Total NaN cells:             {total_nan}")
print(f"Total Inf cells:             {int(inf_counts)}")

print("\nSAVED:")
print(AUDIT_FILE)
print(QC_FILE)

print("\n" + "=" * 80)
print(f"STATUS: {status}")
print("=" * 80)