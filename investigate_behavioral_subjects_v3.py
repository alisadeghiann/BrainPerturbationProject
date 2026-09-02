import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

QC_DIR = os.path.join(
    BASE,
    "qc",
    "trial_anomaly_inspection"
)

V2_DIR = os.path.join(
    QC_DIR,
    "behavioral_investigation_v2"
)

OUT_DIR = os.path.join(
    QC_DIR,
    "behavioral_investigation_v3"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# INPUT FILES
# ============================================================

FILES = {
    "accuracy": os.path.join(
        QC_DIR,
        "ACCURACY_BY_SUBJECT.csv"
    ),

    "bad_trials": os.path.join(
        QC_DIR,
        "ALL_BAD_TRIALS.csv"
    ),

    "summary": os.path.join(
        QC_DIR,
        "subject_anomaly_summary.csv"
    ),

    "sub005": os.path.join(
        QC_DIR,
        "SUB005_ALL_TRIALS.csv"
    ),

    "non14": os.path.join(
        QC_DIR,
        "all_non14_trials.csv"
    ),

    "memory_event": os.path.join(
        QC_DIR,
        "MEMORY_X_EVENTCOUNT.csv"
    ),

    "severe": os.path.join(
        QC_DIR,
        "SEVERE_TRIAL_ANOMALIES.csv"
    )
}

# ============================================================
# HELPERS
# ============================================================

def load_csv(name, path):
    print()
    print("=" * 90)
    print(f"LOADING: {name}")
    print("=" * 90)

    if not os.path.exists(path):
        print("NOT FOUND:")
        print(path)
        return None

    try:
        df = pd.read_csv(path)

        print(f"Rows:    {len(df)}")
        print(f"Columns: {len(df.columns)}")

        print("Columns:")
        print(list(df.columns))

        return df

    except Exception as e:
        print("ERROR:")
        print(e)
        return None


def find_subject_column(df):
    if df is None:
        return None

    candidates = [
        "subject",
        "sub",
        "participant",
        "participant_id"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


# ============================================================
# LOAD
# ============================================================

accuracy = load_csv(
    "ACCURACY_BY_SUBJECT",
    FILES["accuracy"]
)

bad_trials = load_csv(
    "ALL_BAD_TRIALS",
    FILES["bad_trials"]
)

summary = load_csv(
    "SUBJECT_ANOMALY_SUMMARY",
    FILES["summary"]
)

sub005 = load_csv(
    "SUB005_ALL_TRIALS",
    FILES["sub005"]
)

non14 = load_csv(
    "ALL_NON14_TRIALS",
    FILES["non14"]
)

memory_event = load_csv(
    "MEMORY_X_EVENTCOUNT",
    FILES["memory_event"]
)

severe = load_csv(
    "SEVERE_TRIAL_ANOMALIES",
    FILES["severe"]
)

# ============================================================
# 1. ACCURACY OVERVIEW
# ============================================================

if accuracy is not None:

    print()
    print("=" * 90)
    print("1. ACCURACY OVERVIEW")
    print("=" * 90)

    print(accuracy.to_string(index=False))

    accuracy.to_csv(
        os.path.join(
            OUT_DIR,
            "accuracy_overview_v3.csv"
        ),
        index=False
    )

# ============================================================
# 2. SUB-024 TARGETED ANALYSIS
# ============================================================

print()
print("=" * 90)
print("2. SUB-024 TARGETED ANALYSIS")
print("=" * 90)

def subject_filter(df, subject):
    if df is None:
        return None

    col = find_subject_column(df)

    if col is None:
        print("No subject column found.")
        return None

    s = df[col].astype(str)

    mask = (
        s.str.lower() == subject.lower()
    ) | (
        s.str.lower() == f"sub-{subject.lower().replace('sub-', '')}"
    )

    return df.loc[mask].copy()


sub024_tables = {}

for name, df in [
    ("bad_trials", bad_trials),
    ("summary", summary),
    ("non14", non14),
    ("severe", severe)
]:

    if df is None:
        continue

    result = subject_filter(
        df,
        "sub-024"
    )

    if result is not None:

        sub024_tables[name] = result

        print()
        print("-" * 90)
        print(f"SUB-024 / {name}")
        print("-" * 90)

        print(
            f"Rows: {len(result)}"
        )

        print(
            result.to_string(index=False)
        )

        result.to_csv(
            os.path.join(
                OUT_DIR,
                f"sub024_{name}_v3.csv"
            ),
            index=False
        )

# ============================================================
# 3. SUB-014 TARGETED ANALYSIS
# ============================================================

print()
print("=" * 90)
print("3. SUB-014 TARGETED ANALYSIS")
print("=" * 90)

for name, df in [
    ("bad_trials", bad_trials),
    ("summary", summary),
    ("non14", non14),
    ("severe", severe)
]:

    if df is None:
        continue

    result = subject_filter(
        df,
        "sub-014"
    )

    if result is not None:

        print()
        print("-" * 90)
        print(f"SUB-014 / {name}")
        print("-" * 90)

        print(
            f"Rows: {len(result)}"
        )

        print(
            result.to_string(index=False)
        )

        result.to_csv(
            os.path.join(
                OUT_DIR,
                f"sub014_{name}_v3.csv"
            ),
            index=False
        )

# ============================================================
# 4. SUB-005 CHECK
# ============================================================

print()
print("=" * 90)
print("4. SUB-005 CHECK")
print("=" * 90)

if sub005 is not None:

    print(
        sub005.to_string(index=False)
    )

    sub005.to_csv(
        os.path.join(
            OUT_DIR,
            "sub005_trials_v3.csv"
        ),
        index=False
    )

# ============================================================
# 5. MEMORY × EVENT COUNT
# ============================================================

print()
print("=" * 90)
print("5. MEMORY × EVENT COUNT")
print("=" * 90)

if memory_event is not None:

    print(
        memory_event.to_string(index=False)
    )

    memory_event.to_csv(
        os.path.join(
            OUT_DIR,
            "memory_event_count_v3.csv"
        ),
        index=False
    )

# ============================================================
# 6. GLOBAL ANOMALY SUMMARY
# ============================================================

print()
print("=" * 90)
print("6. GLOBAL SUBJECT RISK")
print("=" * 90)

if accuracy is not None:

    acc_col = None

    for c in [
        "accuracy_percent",
        "accuracy"
    ]:
        if c in accuracy.columns:
            acc_col = c
            break

    subject_col = find_subject_column(
        accuracy
    )

    if (
        acc_col is not None
        and subject_col is not None
    ):

        tmp = accuracy.copy()

        if acc_col == "accuracy":
            tmp["accuracy_percent_calc"] = (
                pd.to_numeric(
                    tmp[acc_col],
                    errors="coerce"
                ) * 100
            )
            value_col = "accuracy_percent_calc"
        else:
            tmp[acc_col] = pd.to_numeric(
                tmp[acc_col],
                errors="coerce"
            )
            value_col = acc_col

        tmp = tmp.sort_values(
            value_col
        )

        print(
            tmp[
                [
                    subject_col,
                    "trials",
                    value_col
                ]
            ].to_string(index=False)
        )

        tmp.to_csv(
            os.path.join(
                OUT_DIR,
                "subject_accuracy_sorted_v3.csv"
            ),
            index=False
        )

# ============================================================
# 7. FINAL INTERPRETATION
# ============================================================

print()
print("=" * 90)
print("7. AUTOMATIC INTERPRETATION")
print("=" * 90)

print()
print("SUB-024:")
print("CRITICAL REVIEW ONLY")
print(
    "Do NOT exclude yet."
)

print()
print("SUB-014:")
print("MODERATE REVIEW ONLY")
print(
    "Do NOT exclude yet."
)

print()
print("SUB-005:")
print(
    "Behavioral accuracy is high; "
    "do NOT exclude based on anomaly tables alone."
)

print()
print("GENERAL RULE:")
print(
    "Behavioral anomalies must be resolved before "
    "subject-level exclusion."
)

print()
print("=" * 90)
print("INVESTIGATION V3 COMPLETE")
print("=" * 90)

print()
print("OUTPUT:")
print(OUT_DIR)

print()
print("NO EEG FILES MODIFIED.")
print("NO EPOCHS DELETED.")
print("NO SUBJECTS DELETED.")
print("NO DATA EXCLUDED.")