import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "behavior_aligned"
    / "trial_level_behavior_full.csv"
)

LOG_DIR = BASE / "features" / "behavior_aligned" / "qc"
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("BEHAVIORAL ALIGNMENT QC")
print("=" * 90)

df = pd.read_csv(INPUT)

print(f"Trials:    {len(df):,}")
print(f"Subjects:  {df['subject'].nunique()}")
print(
    f"Runs:      "
    f"{df[['subject','run']].drop_duplicates().shape[0]:,}"
)

# =========================================================
# 1. Alignment status
# =========================================================

print()
print("=" * 90)
print("1. ALIGNMENT STATUS")
print("=" * 90)

print(
    df["alignment_status"]
    .value_counts(dropna=False)
)

# =========================================================
# 2. Inspect partial trials
# =========================================================

partial = df[
    df["alignment_status"] != "ALIGNED"
].copy()

print()
print("=" * 90)
print("2. PARTIAL / PROBLEM TRIALS")
print("=" * 90)

if len(partial) == 0:
    print("NONE")

else:
    cols = [
        "subject",
        "run",
        "trial",
        "memory_cond",
        "remember_count",
        "ignore_count",
        "remember_letters",
        "ignore_letters",
        "probe_type",
        "probe_letter",
        "behavior_label",
        "alignment_status"
    ]

    available = [c for c in cols if c in partial.columns]

    print(
        partial[available]
        .to_string(index=False)
    )

partial.to_csv(
    LOG_DIR / "partial_trials.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 3. Memory condition
# =========================================================

print()
print("=" * 90)
print("3. MEMORY CONDITION")
print("=" * 90)

print(
    df["memory_cond"]
    .value_counts(dropna=False)
    .sort_index()
)

# =========================================================
# 4. Probe type
# =========================================================

print()
print("=" * 90)
print("4. PROBE TYPE")
print("=" * 90)

print(
    df["probe_type"]
    .value_counts(dropna=False)
)

# =========================================================
# 5. Behavior labels
# =========================================================

print()
print("=" * 90)
print("5. BEHAVIOR LABEL")
print("=" * 90)

print(
    df["behavior_label"]
    .value_counts(dropna=False)
)

# =========================================================
# 6. Remember vs Ignore
# =========================================================

print()
print("=" * 90)
print("6. REMEMBER / IGNORE COUNTS")
print("=" * 90)

print(
    "Trials with remember items:",
    (df["remember_count"] > 0).sum()
)

print(
    "Trials with ignore items:",
    (df["ignore_count"] > 0).sum()
)

print(
    "Trials with BOTH:",
    (
        (df["remember_count"] > 0)
        &
        (df["ignore_count"] > 0)
    ).sum()
)

# =========================================================
# 7. Correctness consistency
# =========================================================

print()
print("=" * 90)
print("7. BEHAVIOR CONSISTENCY")
print("=" * 90)

valid_labels = [
    "remembered_correct",
    "ignored_correct",
    "remembered_incorrect",
    "ignored_incorrect"
]

label_mask = df["behavior_label"].isin(valid_labels)

print(
    "Valid behavioral labels:",
    label_mask.sum()
)

print(
    "Missing behavioral labels:",
    (~label_mask).sum()
)

# =========================================================
# 8. Target consistency
# =========================================================

print()
print("=" * 90)
print("8. TARGET CONSISTENCY")
print("=" * 90)

target = df["probe_type"] == "target"
not_shown = df["probe_type"] == "not_shown"

print("Target:", target.sum())
print("Not shown:", not_shown.sum())
print("Missing:", df["probe_type"].isna().sum())

# =========================================================
# 9. Subject coverage
# =========================================================

print()
print("=" * 90)
print("9. SUBJECT COVERAGE")
print("=" * 90)

subject_summary = (
    df.groupby("subject")
      .agg(
          trials=("trial", "count"),
          aligned=("alignment_status",
                   lambda x: (x == "ALIGNED").sum()),
          partial=("alignment_status",
                   lambda x: (x != "ALIGNED").sum()),
          valid_labels=("behavior_label",
                        lambda x: x.notna().sum())
      )
      .reset_index()
)

print(subject_summary.to_string(index=False))

subject_summary.to_csv(
    LOG_DIR / "subject_behavior_qc.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 10. Run coverage
# =========================================================

print()
print("=" * 90)
print("10. RUN COVERAGE")
print("=" * 90)

run_summary = (
    df.groupby(["subject", "run"])
      .agg(
          trials=("trial", "count"),
          aligned=("alignment_status",
                   lambda x: (x == "ALIGNED").sum()),
          partial=("alignment_status",
                   lambda x: (x != "ALIGNED").sum()),
          valid_labels=("behavior_label",
                        lambda x: x.notna().sum())
      )
      .reset_index()
)

print(
    run_summary[
        run_summary["partial"] > 0
    ].to_string(index=False)
)

run_summary.to_csv(
    LOG_DIR / "run_behavior_qc.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 11. Duplicate trial check
# =========================================================

print()
print("=" * 90)
print("11. DUPLICATE TRIAL CHECK")
print("=" * 90)

duplicates = df[
    df.duplicated(
        subset=["subject", "run", "trial"],
        keep=False
    )
]

print(
    "Duplicate subject/run/trial rows:",
    len(duplicates)
)

# =========================================================
# 12. Final QC decision
# =========================================================

print()
print("=" * 90)
print("12. FINAL QC DECISION")
print("=" * 90)

issues = []

if duplicates.shape[0] > 0:
    issues.append("DUPLICATE_TRIALS")

if df["behavior_label"].isna().sum() > 0:
    issues.append("MISSING_BEHAVIOR_LABELS")

if df["probe_type"].isna().sum() > 0:
    issues.append("MISSING_PROBE_TYPE")

if len(partial) > 0:
    issues.append("PARTIAL_TRIALS")

if issues:
    print("STATUS: REVIEW REQUIRED")
    print()
    print("Issues:")
    for x in issues:
        print("-", x)

else:
    print("STATUS: PASS")
    print("All behavioral alignment checks passed.")

print()
print("=" * 90)
print("QC COMPLETE - READ ONLY")
print("=" * 90)

print()
print("Saved:")
print(LOG_DIR / "partial_trials.csv")
print(LOG_DIR / "subject_behavior_qc.csv")
print(LOG_DIR / "run_behavior_qc.csv")

