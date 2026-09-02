import os
import pandas as pd

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

TRIAL_FILE = os.path.join(
    BASE,
    "qc",
    "trial_table",
    "TRIAL_LEVEL_TABLE.csv"
)

OUT_DIR = os.path.join(
    BASE,
    "qc",
    "trial_anomaly_inspection"
)

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 90)
print("TRIAL ANOMALY INVESTIGATION")
print("=" * 90)

print("\nLoading:")
print(TRIAL_FILE)

df = pd.read_csv(TRIAL_FILE)

print("\nRows:", len(df))
print("Columns:")
for c in df.columns:
    print("  ", c)

# ---------------------------------------------------------
# BASIC NORMALIZATION
# ---------------------------------------------------------

if "subject" in df.columns:
    df["subject"] = df["subject"].astype(str)

if "run" in df.columns:
    df["run"] = df["run"].astype(str)

# ---------------------------------------------------------
# 1. NON-14 TRIALS
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("1. NON-14 EVENT TRIALS")
print("=" * 90)

non14 = df[df["event_count"] != 14].copy()

print("Count:", len(non14))

if len(non14) > 0:

    print("\nBy subject:")
    print(
        non14.groupby("subject")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nBy event count:")
    print(
        non14["event_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nBy memory condition:")
    if "memory_cond" in non14.columns:
        print(
            non14["memory_cond"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

non14.to_csv(
    os.path.join(OUT_DIR, "all_non14_trials.csv"),
    index=False
)

# ---------------------------------------------------------
# 2. SUBJECT-LEVEL ANOMALIES
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("2. SUBJECT-LEVEL ANOMALY SUMMARY")
print("=" * 90)

subject_summary = (
    df.groupby("subject")
    .agg(
        total_trials=("event_count", "size"),
        mean_events=("event_count", "mean"),
        min_events=("event_count", "min"),
        max_events=("event_count", "max"),
        non14_trials=("event_count", lambda x: (x != 14).sum()),
        bad_trials=("bad_trial", "sum"),
    )
    .reset_index()
)

subject_summary["non14_percent"] = (
    subject_summary["non14_trials"]
    / subject_summary["total_trials"]
    * 100
)

subject_summary = subject_summary.sort_values(
    "non14_percent",
    ascending=False
)

print(subject_summary.to_string(index=False))

subject_summary.to_csv(
    os.path.join(OUT_DIR, "subject_anomaly_summary.csv"),
    index=False
)

# ---------------------------------------------------------
# 3. SUB-005 DEEP INSPECTION
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("3. SUB-005 DEEP INSPECTION")
print("=" * 90)

sub005 = df[df["subject"] == "sub-005"].copy()

print("Sub-005 trials:", len(sub005))

if len(sub005) > 0:

    print("\nColumns available:")
    print(list(sub005.columns))

    print("\nEvent-count distribution:")
    print(
        sub005["event_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFirst 30 trials:")
    cols = [
        c for c in [
            "subject",
            "run",
            "trial",
            "event_count",
            "memory_cond",
            "accuracy",
            "bad_trial"
        ]
        if c in sub005.columns
    ]

    print(
        sub005[cols]
        .head(30)
        .to_string(index=False)
    )

    sub005.to_csv(
        os.path.join(OUT_DIR, "SUB005_ALL_TRIALS.csv"),
        index=False
    )

# ---------------------------------------------------------
# 4. SEVERE ANOMALIES
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("4. SEVERE TRIAL ANOMALIES")
print("=" * 90)

severe = df[
    (df["event_count"] <= 4) |
    (df["event_count"] >= 15)
].copy()

print("Severe anomalies:", len(severe))

if len(severe) > 0:
    print(
        severe[
            [
                c for c in [
                    "subject",
                    "run",
                    "trial",
                    "event_count",
                    "memory_cond",
                    "accuracy",
                    "bad_trial"
                ]
                if c in severe.columns
            ]
        ].to_string(index=False)
    )

severe.to_csv(
    os.path.join(OUT_DIR, "SEVERE_TRIAL_ANOMALIES.csv"),
    index=False
)

# ---------------------------------------------------------
# 5. BAD TRIAL INSPECTION
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("5. BAD TRIALS")
print("=" * 90)

bad = df[df["bad_trial"] == True].copy()

print("Bad trials:", len(bad))

if len(bad) > 0:

    cols = [
        c for c in [
            "subject",
            "run",
            "trial",
            "event_count",
            "memory_cond",
            "accuracy",
            "bad_trial"
        ]
        if c in bad.columns
    ]

    print(
        bad[cols]
        .sort_values(["subject", "trial"])
        .to_string(index=False)
    )

bad.to_csv(
    os.path.join(OUT_DIR, "ALL_BAD_TRIALS.csv"),
    index=False
)

# ---------------------------------------------------------
# 6. MEMORY CONDITION DISTRIBUTION
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("6. MEMORY CONDITION × EVENT COUNT")
print("=" * 90)

if "memory_cond" in df.columns:

    mem = pd.crosstab(
        df["event_count"],
        df["memory_cond"]
    )

    print(mem.to_string())

    mem.to_csv(
        os.path.join(
            OUT_DIR,
            "MEMORY_X_EVENTCOUNT.csv"
        )
    )

# ---------------------------------------------------------
# 7. ACCURACY
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("7. ACCURACY BY SUBJECT")
print("=" * 90)

if "accuracy" in df.columns:

    acc = (
        df.groupby("subject")["accuracy"]
        .agg(
            trials="size",
            accuracy="mean"
        )
        .reset_index()
    )

    acc["accuracy_percent"] = acc["accuracy"] * 100

    acc = acc.sort_values(
        "accuracy_percent"
    )

    print(acc.to_string(index=False))

    acc.to_csv(
        os.path.join(
            OUT_DIR,
            "ACCURACY_BY_SUBJECT.csv"
        ),
        index=False
    )

# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n" + "=" * 90)
print("INVESTIGATION COMPLETE")
print("=" * 90)

print("\nOutput:")
print(OUT_DIR)

print("\nCreated files:")

for f in sorted(os.listdir(OUT_DIR)):
    print("  ", f)

print("\nIMPORTANT:")
print("No EEG files were modified.")
print("No trials were deleted.")
print("No subjects were deleted.")

print("\nNEXT STEP:")
print("Review SUB005_ALL_TRIALS.csv and subject_anomaly_summary.csv")