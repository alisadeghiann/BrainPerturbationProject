# ============================================================
# INVESTIGATE BEHAVIORAL / TRIAL ANOMALIES V2
# Brain Perturbation Project
#
# PURPOSE:
#   Investigate suspicious behavioral results before making
#   ANY exclusion decision.
#
# IMPORTANT:
#   READ-ONLY
#   NO EEG FILES ARE MODIFIED
#   NO EPOCHS ARE DELETED
#   NO SUBJECTS ARE DELETED
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

INPUT_DIR = (
    BASE
    / "qc"
    / "trial_anomaly_inspection"
)

OUTPUT_DIR = (
    BASE
    / "qc"
    / "trial_anomaly_inspection"
    / "behavioral_investigation_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# INPUT FILES
# ============================================================

FILES = {
    "accuracy_subject":
        INPUT_DIR / "ACCURACY_BY_SUBJECT.csv",

    "all_bad_trials":
        INPUT_DIR / "ALL_BAD_TRIALS.csv",

    "memory_eventcount":
        INPUT_DIR / "MEMORY_X_EVENTCOUNT.csv",

    "severe_anomalies":
        INPUT_DIR / "SEVERE_TRIAL_ANOMALIES.csv",

    "sub005":
        INPUT_DIR / "SUB005_ALL_TRIALS.csv",

    "non14":
        INPUT_DIR / "all_non14_trials.csv",

    "subject_summary":
        INPUT_DIR / "subject_anomaly_summary.csv",
}

# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("BEHAVIORAL / TRIAL ANOMALY INVESTIGATION V2")
print("=" * 80)
print()

data = {}

for name, path in FILES.items():

    print(f"Loading: {name}")

    if not path.exists():

        print(
            f"  WARNING - FILE NOT FOUND:\n"
            f"  {path}"
        )

        continue

    try:

        df = pd.read_csv(path)

        data[name] = df

        print(
            f"  Rows: {len(df):,}"
        )

        print(
            f"  Columns: {list(df.columns)}"
        )

    except Exception as e:

        print(
            f"  ERROR: {e}"
        )

    print()

# ============================================================
# 1. SUBJECT ACCURACY
# ============================================================

print("=" * 80)
print("1. SUBJECT ACCURACY")
print("=" * 80)

if "accuracy_subject" in data:

    df = data["accuracy_subject"].copy()

    print(df.to_string(index=False))

    out = OUTPUT_DIR / "accuracy_subject_v2.csv"

    df.to_csv(
        out,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("Saved:")
    print(out)

# ============================================================
# 2. FLAG LOW ACCURACY SUBJECTS
# ============================================================

print()
print("=" * 80)
print("2. LOW ACCURACY SUBJECT SCREEN")
print("=" * 80)

if "accuracy_subject" in data:

    df = data["accuracy_subject"].copy()

    if "accuracy" in df.columns:

        df["accuracy_percent_calc"] = (
            df["accuracy"] * 100
        )

        low = df[
            df["accuracy_percent_calc"] < 80
        ].copy()

        print()
        print("Subjects below 80%:")

        if len(low) == 0:

            print("NONE")

        else:

            print(
                low.to_string(index=False)
            )

        low.to_csv(
            OUTPUT_DIR / "low_accuracy_subjects.csv",
            index=False,
            encoding="utf-8-sig"
        )

# ============================================================
# 3. SUBJECT ANOMALY SUMMARY
# ============================================================

print()
print("=" * 80)
print("3. SUBJECT ANOMALY SUMMARY")
print("=" * 80)

if "subject_summary" in data:

    df = data["subject_summary"].copy()

    print(
        df.to_string(index=False)
    )

    df.to_csv(
        OUTPUT_DIR / "subject_anomaly_summary_v2.csv",
        index=False,
        encoding="utf-8-sig"
    )

# ============================================================
# 4. SUB-005
# ============================================================

print()
print("=" * 80)
print("4. SUB-005 DETAILED INSPECTION")
print("=" * 80)

if "sub005" in data:

    df = data["sub005"].copy()

    print()
    print(
        f"Total SUB-005 records: {len(df)}"
    )

    print()

    print(
        df.to_string(index=False)
    )

    df.to_csv(
        OUTPUT_DIR / "SUB005_ALL_TRIALS_v2.csv",
        index=False,
        encoding="utf-8-sig"
    )

else:

    print(
        "SUB005_ALL_TRIALS.csv not found."
    )

# ============================================================
# 5. BAD TRIALS
# ============================================================

print()
print("=" * 80)
print("5. BAD TRIAL ANALYSIS")
print("=" * 80)

if "all_bad_trials" in data:

    df = data["all_bad_trials"].copy()

    print(
        f"Bad-trial records: {len(df):,}"
    )

    print()

    print(
        "Columns:"
    )

    print(
        list(df.columns)
    )

    print()

    print(
        df.head(100).to_string(index=False)
    )

    df.to_csv(
        OUTPUT_DIR / "ALL_BAD_TRIALS_v2.csv",
        index=False,
        encoding="utf-8-sig"
    )

# ============================================================
# 6. SEVERE ANOMALIES
# ============================================================

print()
print("=" * 80)
print("6. SEVERE TRIAL ANOMALIES")
print("=" * 80)

if "severe_anomalies" in data:

    df = data["severe_anomalies"].copy()

    print(
        f"Severe anomaly records: {len(df):,}"
    )

    print()

    print(
        df.head(100).to_string(index=False)
    )

    df.to_csv(
        OUTPUT_DIR / "SEVERE_TRIAL_ANOMALIES_v2.csv",
        index=False,
        encoding="utf-8-sig"
    )

# ============================================================
# 7. NON-14 TRIALS
# ============================================================

print()
print("=" * 80)
print("7. NON-14 TRIALS")
print("=" * 80)

if "non14" in data:

    df = data["non14"].copy()

    print(
        f"Non-14 records: {len(df):,}"
    )

    print()

    print(
        df.head(100).to_string(index=False)
    )

    df.to_csv(
        OUTPUT_DIR / "all_non14_trials_v2.csv",
        index=False,
        encoding="utf-8-sig"
    )

# ============================================================
# 8. MEMORY × EVENT COUNT
# ============================================================

print()
print("=" * 80)
print("8. MEMORY × EVENT COUNT")
print("=" * 80)

if "memory_eventcount" in data:

    df = data["memory_eventcount"].copy()

    print(
        df.to_string(index=False)
    )

    df.to_csv(
        OUTPUT_DIR / "MEMORY_X_EVENTCOUNT_v2.csv",
        index=False,
        encoding="utf-8-sig"
    )

# ============================================================
# 9. AUTOMATIC SUBJECT SUMMARY
# ============================================================

print()
print("=" * 80)
print("9. AUTOMATIC SUBJECT RISK SCREEN")
print("=" * 80)

if "accuracy_subject" in data:

    df = data["accuracy_subject"].copy()

    # Standardize subject column

    subject_col = None

    for col in df.columns:

        if col.lower() == "subject":

            subject_col = col
            break

    if subject_col is not None:

        records = []

        for _, row in df.iterrows():

            subject = row[subject_col]

            accuracy = np.nan

            if "accuracy" in df.columns:

                accuracy = float(
                    row["accuracy"]
                )

            elif "accuracy_percent" in df.columns:

                accuracy = (
                    float(row["accuracy_percent"])
                    / 100.0
                )

            if np.isnan(accuracy):

                classification = "UNKNOWN"

            elif accuracy < 0.70:

                classification = "CRITICAL_REVIEW"

            elif accuracy < 0.80:

                classification = "HIGH_REVIEW"

            elif accuracy < 0.90:

                classification = "MODERATE_REVIEW"

            else:

                classification = "NORMAL"

            records.append(
                {
                    "subject": subject,
                    "accuracy": accuracy,
                    "accuracy_percent":
                        accuracy * 100
                        if not np.isnan(accuracy)
                        else np.nan,
                    "behavioral_flag":
                        classification
                }
            )

        risk = pd.DataFrame(records)

        print(
            risk.to_string(index=False)
        )

        risk.to_csv(
            OUTPUT_DIR / "subject_behavioral_risk_v2.csv",
            index=False,
            encoding="utf-8-sig"
        )

# ============================================================
# 10. SPECIFIC SUB-024 CHECK
# ============================================================

print()
print("=" * 80)
print("10. SUB-024 TARGETED REVIEW")
print("=" * 80)

if "accuracy_subject" in data:

    df = data["accuracy_subject"].copy()

    if "subject" in df.columns:

        sub024 = df[
            df["subject"].astype(str).str.lower()
            == "sub-024"
        ]

        if len(sub024) > 0:

            print(
                sub024.to_string(index=False)
            )

        else:

            print(
                "SUB-024 not found in accuracy table."
            )

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("INVESTIGATION V2 COMPLETE")
print("=" * 80)

print()
print("OUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print()
print("IMPORTANT:")
print("NO EEG FILES WERE MODIFIED.")
print("NO EPOCHS WERE DELETED.")
print("NO SUBJECTS WERE DELETED.")
print("NO DATA WAS EXCLUDED.")
print()
print(
    "NEXT DECISION MUST BE MADE ONLY AFTER "
    "REVIEWING THE GENERATED CSV FILES."
)