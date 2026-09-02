import pandas as pd
import os

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

INPUT = os.path.join(
    BASE,
    "qc",
    "trial_anomaly_inspection",
    "sub024_final_behavior_review",
    "sub024_all_trials_corrected.csv"
)

OUTDIR = os.path.join(
    BASE,
    "qc",
    "trial_anomaly_inspection",
    "sub024_final_behavior_review",
    "behavior_response_audit"
)

os.makedirs(OUTDIR, exist_ok=True)

print("=" * 90)
print("SUB-024 RESPONSE / BEHAVIOR AUDIT")
print("=" * 90)

df = pd.read_csv(INPUT)

print("\nROWS:", len(df))

print("\nCOLUMNS:")
for c in df.columns:
    print(" -", c)

print("\n" + "=" * 90)
print("1. BASIC DATA")
print("=" * 90)

for col in ["subject", "run_from_file", "trial", "memory_cond",
            "accuracy", "feedback", "event_count"]:

    if col in df.columns:
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False).sort_index())

print("\n" + "=" * 90)
print("2. POSSIBLE RESPONSE / TARGET COLUMNS")
print("=" * 90)

keywords = [
    "response",
    "answer",
    "target",
    "correct",
    "expected",
    "choice",
    "button",
    "key",
    "feedback",
    "memory"
]

for col in df.columns:
    low = col.lower()

    if any(k in low for k in keywords):
        print("\nCOLUMN:", col)
        print("dtype:", df[col].dtype)
        print("unique:", df[col].nunique(dropna=False))

        vals = df[col].value_counts(dropna=False).head(30)
        print(vals.to_string())

print("\n" + "=" * 90)
print("3. COMPLETE FIRST 10 TRIALS")
print("=" * 90)

print(df.head(10).to_string(index=False))

print("\n" + "=" * 90)
print("4. COMPLETE LAST 10 TRIALS")
print("=" * 90)

print(df.tail(10).to_string(index=False))

print("\n" + "=" * 90)
print("5. MEMORY × FEEDBACK")
print("=" * 90)

if "memory_cond" in df.columns and "feedback" in df.columns:

    tab = pd.crosstab(
        df["memory_cond"],
        df["feedback"],
        dropna=False
    )

    print(tab.to_string())

    tab.to_csv(
        os.path.join(OUTDIR, "memory_x_feedback.csv")
    )

print("\n" + "=" * 90)
print("6. RUN × FEEDBACK")
print("=" * 90)

if "run_from_file" in df.columns and "feedback" in df.columns:

    tab = pd.crosstab(
        df["run_from_file"],
        df["feedback"],
        dropna=False
    )

    print(tab.to_string())

    tab.to_csv(
        os.path.join(OUTDIR, "run_x_feedback.csv")
    )

print("\n" + "=" * 90)
print("7. SEARCH FOR POSSIBLE TARGET/RESPONSE MISMATCH")
print("=" * 90)

candidate_cols = []

for col in df.columns:
    low = col.lower()

    if any(k in low for k in [
        "response",
        "answer",
        "target",
        "choice",
        "button",
        "expected",
        "correct"
    ]):
        candidate_cols.append(col)

print("Candidate columns:")
for c in candidate_cols:
    print(" -", c)

print("\n" + "=" * 90)
print("8. SAVE FULL AUDIT TABLE")
print("=" * 90)

out_csv = os.path.join(
    OUTDIR,
    "sub024_behavior_response_audit_full.csv"
)

df.to_csv(out_csv, index=False)

print("Saved:")
print(out_csv)

print("\n" + "=" * 90)
print("AUDIT COMPLETE")
print("=" * 90)

print("\nNO EEG FILES MODIFIED.")
print("NO EPOCHS DELETED.")
print("NO SUBJECTS DELETED.")
print("NO DATA EXCLUDED.")
