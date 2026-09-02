import pandas as pd
from pathlib import Path

# ============================================================
# GROUP-LEVEL BEHAVIORAL QC - READ ONLY
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

EVENT_DIR = BASE / "qc" / "events"

OUT_DIR = (
    BASE
    / "qc"
    / "group_behavioral_qc"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("GROUP-LEVEL BEHAVIORAL QC - READ ONLY")
print("=" * 90)

event_files = sorted(EVENT_DIR.glob("*_eeg_events.csv"))

if not event_files:
    print("NO EVENT FILES FOUND")
    raise SystemExit(1)

print(f"\nEvent files found: {len(event_files)}")

all_trials = []
run_summaries = []

# ============================================================
# PROCESS EVERY EVENT FILE
# ============================================================

for event_file in event_files:

    name = event_file.name

    # --------------------------------------------------------
    # Parse subject
    # --------------------------------------------------------

    parts = name.split("_")

    subject = parts[0]
    run_part = [p for p in parts if p.startswith("run-")]

    if not run_part:
        print(f"SKIPPING - RUN NOT FOUND: {name}")
        continue

    run = int(run_part[0].replace("run-", ""))

    print(f"\nProcessing: {subject} | Run {run}")

    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    try:
        df = pd.read_csv(event_file)
    except Exception as e:
        print(f"READ ERROR: {e}")
        continue

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    for col in ["type", "task_role", "value"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "trial" not in df.columns:
        print("NO TRIAL COLUMN - SKIPPING")
        continue

    trial_numeric = pd.to_numeric(
        df["trial"],
        errors="coerce"
    )

    trial_ids = sorted(
        trial_numeric.dropna().unique()
    )

    # --------------------------------------------------------
    # Trial reconstruction
    # --------------------------------------------------------

    for trial in trial_ids:

        t = df[trial_numeric == trial].copy()

        # ----------------------------------------------------
        # Memory condition
        # ----------------------------------------------------

        if (
            "memory_cond" in t.columns
            and t["memory_cond"].notna().any()
        ):
            memory_cond = t["memory_cond"].dropna().iloc[0]
        else:
            memory_cond = None

        # ----------------------------------------------------
        # Probe
        # ----------------------------------------------------

        probe = t[
            t["task_role"].isin(
                ["probe_target", "probe_not_shown"]
            )
        ]

        if len(probe) > 0:
            probe_role = probe.iloc[-1]["task_role"]

            probe_letter = (
                probe.iloc[-1]["letter"]
                if "letter" in probe.columns
                else None
            )

            probe_latency = probe.iloc[-1]["latency"]
        else:
            probe_role = None
            probe_letter = None
            probe_latency = None

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        response = t[
            t["task_role"].isin(
                ["ignored_correct", "ignored_incorrect"]
            )
        ]

        if len(response) > 0:
            response_role = response.iloc[0]["task_role"]
            response_latency = response.iloc[0]["latency"]
        else:
            response_role = None
            response_latency = None

        # ----------------------------------------------------
        # Feedback
        # ----------------------------------------------------

        feedback = t[
            t["task_role"].isin(
                ["feedback_correct", "feedback_incorrect"]
            )
        ]

        if len(feedback) > 0:
            feedback_role = feedback.iloc[0]["task_role"]
            feedback_value = feedback.iloc[0]["value"]
        else:
            feedback_role = None
            feedback_value = None

        # ----------------------------------------------------
        # RT
        # ----------------------------------------------------

        if (
            probe_latency is not None
            and response_latency is not None
        ):
            try:
                rt_ms = (
                    float(response_latency)
                    - float(probe_latency)
                )
            except Exception:
                rt_ms = None
        else:
            rt_ms = None

        # ----------------------------------------------------
        # Behavioral classification
        # ----------------------------------------------------

        if response_role == "ignored_correct":
            behavior = "correct"

        elif response_role == "ignored_incorrect":
            behavior = "wrong"

        elif feedback_role == "feedback_correct":
            behavior = "correct"

        elif feedback_role == "feedback_incorrect":
            behavior = "wrong"

        else:
            behavior = "unresolved"

        # ----------------------------------------------------
        # Response / feedback consistency
        # ----------------------------------------------------

        if (
            response_role == "ignored_correct"
            and feedback_role == "feedback_correct"
        ):
            consistency = True

        elif (
            response_role == "ignored_incorrect"
            and feedback_role == "feedback_incorrect"
        ):
            consistency = True

        elif (
            response_role is not None
            and feedback_role is not None
        ):
            consistency = False

        else:
            consistency = None

        # ----------------------------------------------------
        # Click count
        # ----------------------------------------------------

        click_count = len(
            t[t["type"] == "left_click"]
        )

        # ----------------------------------------------------
        # Issues
        # ----------------------------------------------------

        issues = []

        if len(probe) == 0:
            issues.append("missing_probe")

        if len(response) == 0:
            issues.append("missing_response")

        if len(feedback) == 0:
            issues.append("missing_feedback")

        if click_count != 2:
            issues.append(
                f"unexpected_click_count_{click_count}"
            )

        if consistency is False:
            issues.append(
                "response_feedback_mismatch"
            )

        if behavior == "unresolved":
            issues.append(
                "unresolved_behavior"
            )

        status = (
            "OK"
            if len(issues) == 0
            else "CHECK"
        )

        # ----------------------------------------------------
        # Save trial
        # ----------------------------------------------------

        all_trials.append(
            {
                "subject": subject,
                "run": run,
                "trial": int(trial),
                "memory_cond": memory_cond,
                "probe_role": probe_role,
                "probe_letter": probe_letter,
                "response_role": response_role,
                "feedback_role": feedback_role,
                "feedback_value": feedback_value,
                "rt_ms": rt_ms,
                "click_count": click_count,
                "response_feedback_consistent": consistency,
                "behavior": behavior,
                "status": status,
                "issues": ";".join(issues),
                "source_file": name
            }
        )

    # --------------------------------------------------------
    # Run summary
    # --------------------------------------------------------

    run_trials = [
        x for x in all_trials
        if x["subject"] == subject
        and x["run"] == run
    ]

    run_df = pd.DataFrame(run_trials)

    total = len(run_df)

    correct = int(
        (run_df["behavior"] == "correct").sum()
    )

    wrong = int(
        (run_df["behavior"] == "wrong").sum()
    )

    unresolved = int(
        (run_df["behavior"] == "unresolved").sum()
    )

    checks = int(
        (run_df["status"] == "CHECK").sum()
    )

    accuracy = (
        correct / (correct + wrong) * 100
        if (correct + wrong) > 0
        else None
    )

    consistency_true = int(
        (run_df["response_feedback_consistent"] == True).sum()
    )

    consistency_false = int(
        (run_df["response_feedback_consistent"] == False).sum()
    )

    run_summaries.append(
        {
            "subject": subject,
            "run": run,
            "trials": total,
            "correct": correct,
            "wrong": wrong,
            "unresolved": unresolved,
            "accuracy_percent": accuracy,
            "consistent": consistency_true,
            "inconsistent": consistency_false,
            "trials_needing_check": checks,
            "source_file": name
        }
    )


# ============================================================
# DATAFRAMES
# ============================================================

trial_df = pd.DataFrame(all_trials)

run_df = pd.DataFrame(run_summaries)


# ============================================================
# SUBJECT-LEVEL SUMMARY
# ============================================================

subject_rows = []

for subject, g in trial_df.groupby("subject"):

    total = len(g)

    correct = int(
        (g["behavior"] == "correct").sum()
    )

    wrong = int(
        (g["behavior"] == "wrong").sum()
    )

    unresolved = int(
        (g["behavior"] == "unresolved").sum()
    )

    checks = int(
        (g["status"] == "CHECK").sum()
    )

    accuracy = (
        correct / (correct + wrong) * 100
        if (correct + wrong) > 0
        else None
    )

    consistent = int(
        (g["response_feedback_consistent"] == True).sum()
    )

    inconsistent = int(
        (g["response_feedback_consistent"] == False).sum()
    )

    n_runs = g["run"].nunique()

    subject_rows.append(
        {
            "subject": subject,
            "runs": n_runs,
            "trials": total,
            "correct": correct,
            "wrong": wrong,
            "unresolved": unresolved,
            "accuracy_percent": accuracy,
            "consistent": consistent,
            "inconsistent": inconsistent,
            "trials_needing_check": checks
        }
    )

subject_df = pd.DataFrame(subject_rows)


# ============================================================
# GROUP STATISTICS
# ============================================================

valid_accuracy = subject_df[
    subject_df["accuracy_percent"].notna()
]["accuracy_percent"]

if len(valid_accuracy) > 0:

    group_mean = valid_accuracy.mean()
    group_median = valid_accuracy.median()
    group_sd = valid_accuracy.std(ddof=1)

    subject_df["accuracy_zscore"] = (
        (subject_df["accuracy_percent"] - group_mean)
        / group_sd
        if group_sd > 0
        else 0
    )

    subject_df["behavioral_outlier_flag"] = (
        subject_df["accuracy_zscore"].abs() >= 2
    )

else:

    group_mean = None
    group_median = None
    group_sd = None

    subject_df["accuracy_zscore"] = None
    subject_df["behavioral_outlier_flag"] = False


# ============================================================
# SAVE FILES
# ============================================================

trial_output = (
    OUT_DIR
    / "group_trial_level_behavioral_qc.csv"
)

run_output = (
    OUT_DIR
    / "group_run_level_behavioral_qc.csv"
)

subject_output = (
    OUT_DIR
    / "group_subject_level_behavioral_qc.csv"
)

trial_df.to_csv(
    trial_output,
    index=False,
    encoding="utf-8-sig"
)

run_df.to_csv(
    run_output,
    index=False,
    encoding="utf-8-sig"
)

subject_df.to_csv(
    subject_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("SUBJECT-LEVEL BEHAVIORAL QC")
print("=" * 90)

display_cols = [
    "subject",
    "runs",
    "trials",
    "correct",
    "wrong",
    "accuracy_percent",
    "unresolved",
    "inconsistent",
    "trials_needing_check"
]

print(
    subject_df[
        display_cols
    ].sort_values("accuracy_percent").to_string(index=False)
)

print("\n")
print("=" * 90)
print("GROUP ACCURACY STATISTICS")
print("=" * 90)

if group_mean is not None:
    print(f"Subjects: {len(valid_accuracy)}")
    print(f"Mean accuracy:   {group_mean:.2f}%")
    print(f"Median accuracy: {group_median:.2f}%")
    print(f"SD:              {group_sd:.2f}%")
else:
    print("No valid accuracy values.")

print("\n")
print("=" * 90)
print("BEHAVIORAL OUTLIERS | |Z| >= 2")
print("=" * 90)

outliers = subject_df[
    subject_df["behavioral_outlier_flag"] == True
]

if len(outliers) == 0:
    print("NO BEHAVIORAL ACCURACY OUTLIERS FOUND.")
else:
    print(
        outliers[
            [
                "subject",
                "accuracy_percent",
                "accuracy_zscore"
            ]
        ].to_string(index=False)
    )

print("\n")
print("=" * 90)
print("SUB-024 CHECK")
print("=" * 90)

sub024 = subject_df[
    subject_df["subject"].astype(str).str.lower() == "sub-024"
]

if len(sub024) > 0:
    print(
        sub024[
            display_cols + [
                "accuracy_zscore",
                "behavioral_outlier_flag"
            ]
        ].to_string(index=False)
    )
else:
    print("SUB-024 NOT FOUND IN GROUP EVENT FILES.")

print("\n")
print("=" * 90)
print("SAVED FILES")
print("=" * 90)

print(trial_output)
print(run_output)
print(subject_output)

print("\n")
print("=" * 90)
print("SAFETY CHECK")
print("=" * 90)

print("READ-ONLY QC")
print("NO EEG FILES MODIFIED.")
print("NO EPOCHS DELETED.")
print("NO SUBJECTS DELETED.")
print("NO DATA EXCLUDED.")

print("=" * 90)

