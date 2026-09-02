from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# FINAL PRODUCT DATASET V1
# EEG PERTURBATION / WHAT-IF PROJECT
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "perturbation_analysis"
    / "final_evidence_reconciliation_v1"
    / "final_evidence_reconciliation_v1.csv"
)

OUT_DIR = (
    BASE
    / "features"
    / "productization"
    / "final_product_dataset_v1"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 90)
print("FINAL PRODUCT DATASET V1")
print("=" * 90)

print(f"Project root: {BASE}")
print()

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("=" * 90)
print("LOADING FINAL EVIDENCE")
print("=" * 90)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

print(f"Rows:       {len(df)}")
print(f"Columns:    {len(df.columns)}")

# ------------------------------------------------------------
# BASIC VALIDATION
# ------------------------------------------------------------

print()
print("=" * 90)
print("BASIC VALIDATION")
print("=" * 90)

if "feature" not in df.columns:
    raise ValueError("Column 'feature' not found.")

if df["feature"].duplicated().any():
    raise ValueError("Duplicate features detected.")

numeric_cols = df.select_dtypes(include=[np.number]).columns

nan_count = int(df[numeric_cols].isna().sum().sum())
inf_count = int(
    np.isinf(df[numeric_cols].to_numpy()).sum()
)

print(f"Unique features:    {df['feature'].nunique()}")
print(f"Numeric columns:    {len(numeric_cols)}")
print(f"NaN numeric cells:  {nan_count}")
print(f"Inf numeric cells:  {inf_count}")

if nan_count != 0:
    raise ValueError("NaN numeric cells detected.")

if inf_count != 0:
    raise ValueError("Inf numeric cells detected.")

# ------------------------------------------------------------
# PRODUCT-FRIENDLY COLUMN SELECTION
# ------------------------------------------------------------

print()
print("=" * 90)
print("CREATING PRODUCT-FRIENDLY DATASET")
print("=" * 90)

preferred_columns = [
    "final_rank",
    "feature",
    "final_evidence_score",
    "final_evidence_class",

    "perturbation_max_abs_d",
    "ml_mean_importance",
    "ml_mean_sign_consistency",
    "loso_mean_effect",

    "r_fdr_significant",
    "null_significant",

    "target",
    "effect_size",
    "p_fdr",
    "direction",

    "frequency",
    "region",
]

existing = [
    c for c in preferred_columns
    if c in df.columns
]

product_df = df[existing].copy()

print(f"Selected columns: {len(product_df.columns)}")

# ------------------------------------------------------------
# ADD DERIVED PRODUCT METRICS
# ------------------------------------------------------------

print()
print("=" * 90)
print("CREATING DERIVED PRODUCT METRICS")
print("=" * 90)

# Evidence strength score
if "final_evidence_score" in product_df.columns:
    product_df["evidence_percent"] = (
        product_df["final_evidence_score"] * 100
    )

# ML stability percentage
if "ml_mean_sign_consistency" in product_df.columns:
    product_df["ml_stability_percent"] = (
        product_df["ml_mean_sign_consistency"] * 100
    )

# Absolute perturbation effect
if "perturbation_max_abs_d" in product_df.columns:
    product_df["perturbation_abs"] = (
        product_df["perturbation_max_abs_d"].abs()
    )

# ------------------------------------------------------------
# EVIDENCE CATEGORY
# ------------------------------------------------------------

def evidence_category(row):
    score = row.get("final_evidence_score", np.nan)

    if pd.isna(score):
        return "unknown"

    if score >= 0.75:
        return "high_evidence"

    if score >= 0.50:
        return "moderate_evidence"

    if score >= 0.25:
        return "weak_evidence"

    return "low_evidence"


product_df["product_evidence_category"] = (
    product_df.apply(evidence_category, axis=1)
)

# ------------------------------------------------------------
# PRODUCT PRIORITY
# ------------------------------------------------------------

def product_priority(row):
    score = row.get("final_evidence_score", np.nan)

    if pd.isna(score):
        return "low"

    if score >= 0.75:
        return "P1"

    if score >= 0.60:
        return "P2"

    if score >= 0.50:
        return "P3"

    return "P4"


product_df["product_priority"] = (
    product_df.apply(product_priority, axis=1)
)

# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

if "final_rank" in product_df.columns:
    product_df = product_df.sort_values(
        by="final_rank",
        ascending=True
    )

# ------------------------------------------------------------
# FINAL QC
# ------------------------------------------------------------

print()
print("=" * 90)
print("FINAL PRODUCT DATASET QC")
print("=" * 90)

final_numeric_cols = product_df.select_dtypes(
    include=[np.number]
).columns

final_nan = int(
    product_df[final_numeric_cols].isna().sum().sum()
)

final_inf = int(
    np.isinf(
        product_df[final_numeric_cols].to_numpy()
    ).sum()
)

duplicate_features = int(
    product_df["feature"].duplicated().sum()
)

print(f"Rows:                 {len(product_df)}")
print(f"Unique features:      {product_df['feature'].nunique()}")
print(f"Columns:              {len(product_df.columns)}")
print(f"NaN numeric cells:    {final_nan}")
print(f"Inf numeric cells:    {final_inf}")
print(f"Duplicate features:   {duplicate_features}")

if final_nan != 0:
    raise ValueError("Final dataset contains NaN values.")

if final_inf != 0:
    raise ValueError("Final dataset contains Inf values.")

if duplicate_features != 0:
    raise ValueError("Duplicate features detected.")

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

print()
print("=" * 90)
print("SAVING FINAL PRODUCT DATASET")
print("=" * 90)

output_csv = (
    OUT_DIR
    / "final_product_dataset_v1.csv"
)

output_summary = (
    OUT_DIR
    / "final_product_dataset_summary_v1.csv"
)

output_qc = (
    OUT_DIR
    / "final_product_dataset_qc_v1.csv"
)

product_df.to_csv(
    output_csv,
    index=False
)

summary_rows = []

summary_rows.append({
    "metric": "rows",
    "value": len(product_df)
})

summary_rows.append({
    "metric": "unique_features",
    "value": product_df["feature"].nunique()
})

summary_rows.append({
    "metric": "columns",
    "value": len(product_df.columns)
})

summary_rows.append({
    "metric": "nan_numeric_cells",
    "value": final_nan
})

summary_rows.append({
    "metric": "inf_numeric_cells",
    "value": final_inf
})

summary_rows.append({
    "metric": "duplicate_features",
    "value": duplicate_features
})

pd.DataFrame(summary_rows).to_csv(
    output_summary,
    index=False
)

qc_df = pd.DataFrame([{
    "rows": len(product_df),
    "unique_features": product_df["feature"].nunique(),
    "columns": len(product_df.columns),
    "nan_numeric_cells": final_nan,
    "inf_numeric_cells": final_inf,
    "duplicate_features": duplicate_features,
    "status": "PASS"
}])

qc_df.to_csv(
    output_qc,
    index=False
)

print()
print("Saved:")
print(output_csv)
print(output_summary)
print(output_qc)

print()
print("=" * 90)
print("FINAL PRODUCT DATASET V1 COMPLETE")
print("=" * 90)
print("STATUS: PASS - PRODUCTIZATION DATASET CREATED")