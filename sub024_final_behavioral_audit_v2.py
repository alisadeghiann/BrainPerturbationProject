import pandas as pd
from pathlib import Path

# ============================================================
# SUB-024 FINAL BEHAVIORAL AUDIT - READ ONLY
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

EVENT_DIR = BASE / "qc" / "events"

OUT_DIR = (
    BASE
    / "qc"
    / "trial_anomaly_inspection"
    / "sub024_final_behavior_review"
    / "behavior_response_audit"
    / "final_behavioral_audit_v2"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT = "sub-024"

all_trials = []
run_summaries = []

print("=" * 80)
print("SUB-024 FINAL BEHAVIORAL AUDIT - READ ONLY")
print("=" * 80)

for run in [1, 2, 3]:

    event_file = (
        EVENT_DIR
        / f"sub-024_ses-01_task-WorkingMemory_run-{run}_eeg_events.csv"
    )

    print(f"\n{'=' * 80}")
    print(f"PROCESSING RUN {run}")
    print(f"{'=' * 80}")

    if not event_file.exists():
        print("EVENT FILE NOT FOUND:")
        print(event_file)
        continue

    df = pd.read_csv(event_file)

    # --------------------------------------------------------
    # Normalize string columns
    # --------------------------------------------------------

    for col in ["type", "task_role", "value"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # --------------------------------------------------------
    # Identify actual task trials
    # --------------------------------------------------------

    trial_ids = sorted(
        pd.to_numeric(df["trial"], errors="coerce")
        .dropna()
        .unique()
    )

    print(f"Total trial IDs found: {len(trial_ids)}")

    # --------------------------------------------------------
    # Trial-level reconstruction
    # --------------------------------------------------------

    for trial in trial_ids:

        t = df[
            pd.to_numeric(df["trial"], errors="coerce") == trial
        ].copy()

        # ----------------------------------------------------
        # Memory condition
        # ----------------------------------------------------

        memory_cond = (
            t["memory_cond"].dropna().iloc[0]
            if "memory_cond" in t.columns and t["memory_cond"].notna().any()
            else None
        )

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
            probe_letter = probe.iloc[-1].get("letter", None)
            probe_latency = probe.iloc[-1].get("latency", None)
        else:
            probe_role = None
            probe_letter = None
            probe_latency = None

        # ----------------------------------------------------
        # Behavioral response
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
            feedback_latency = feedback.iloc[0]["latency"]
        else:
            feedback_role = None
            feedback_value = None
            feedback_latency = None

        # ----------------------------------------------------
        # RT
        # ----------------------------------------------------

        if (
            probe_latency is not None
            and response_latency is not None
        ):
            try:
                rt_ms = float(response_latency) - float(probe_latency)
            except:
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
        # Consistency check
        # ----------------------------------------------------

        response_feedback_consistent = None

        if (
            response_role == "ignored_correct"
            and feedback_role == "feedback_correct"
        ):
            response_feedback_consistent = True

        elif (
            response_role == "ignored_incorrect"
            and feedback_role == "feedback_incorrect"
        ):
            response_feedback_consistent = True

        elif response_role is not None and feedback_role is not None:
            response_feedback_consistent = False

        # ----------------------------------------------------
        # Click count
        # ----------------------------------------------------

        click_count = len(
            t[t["type"] == "left_click"]
        )

        # ----------------------------------------------------
        # Trial completeness
        # ----------------------------------------------------

        issues = []

        if len(probe) == 0:
            issues.append("missing_probe")

        if len(response) == 0:
            issues.append("missing_response")

        if len(feedback) == 0:
            issues.append("missing_feedback")

        if click_count != 2:
            issues.append(f"unexpected_click_count_{click_count}")

        if response_feedback_consistent is False:
            issues.append("response_feedback_mismatch")

        if behavior == "unresolved":
            issues.append("unresolved_behavior")

        status = "OK" if len(issues) == 0 else "CHECK"

        # ----------------------------------------------------
        # Save trial record
        # ----------------------------------------------------

        all_trials.append(
            {
                "subject": SUBJECT,
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
                "response_feedback_consistent":
                    response_feedback_consistent,
                "behavior": behavior,
                "status": status,
                "issues": ";".join(issues)
            }
        )

    # --------------------------------------------------------
    # Run summary
    # --------------------------------------------------------

    run_df = pd.DataFrame(
        [
            x for x in all_trials
            if x["run"] == run
        ]
    )

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

    checked = int(
        (run_df["status"] == "CHECK").sum()
    )

    accuracy = (
        correct / (correct + wrong) * 100
        if (correct + wrong) > 0
        else None
    )

    consistent = int(
        (run_df["response_feedback_consistent"] == True).sum()
    )

    inconsistent = int(
        (run_df["response_feedback_consistent"] == False).sum()
    )

    run_summaries.append(
        {
            "subject": SUBJECT,
            "run": run,
            "trials": total,
            "correct": correct,
            "wrong": wrong,
            "unresolved": unresolved,
            "accuracy_percent": accuracy,
            "response_feedback_consistent": consistent,
            "response_feedback_inconsistent": inconsistent,
            "trials_needing_check": checked
        }
    )

    print(f"Trials:       {total}")
    print(f"Correct:      {correct}")
    print(f"Wrong:        {wrong}")
    print(f"Unresolved:   {unresolved}")

    if accuracy is not None:
        print(f"Accuracy:     {accuracy:.2f}%")
    else:
        print("Accuracy:     N/A")

    print(f"Consistent:   {consistent}")
    print(f"Inconsistent: {inconsistent}")
    print(f"Need CHECK:   {checked}")


# ============================================================
# SAVE TRIAL-LEVEL OUTPUT
# ============================================================

trial_df = pd.DataFrame(all_trials)

trial_output = OUT_DIR / "sub024_trial_level_behavioral_audit_v2.csv"

trial_df.to_csv(
    trial_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE RUN SUMMARY
# ============================================================

summary_df = pd.DataFrame(run_summaries)

summary_output = OUT_DIR / "sub024_run_behavioral_summary_v2.csv"

summary_df.to_csv(
    summary_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

total_trials = len(trial_df)

total_correct = int(
    (trial_df["behavior"] == "correct").sum()
)

total_wrong = int(
    (trial_df["behavior"] == "wrong").sum()
)

total_unresolved = int(
    (trial_df["behavior"] == "unresolved").sum()
)

total_check = int(
    (trial_df["status"] == "CHECK").sum()
)

overall_accuracy = (
    total_correct / (total_correct + total_wrong) * 100
    if (total_correct + total_wrong) > 0
    else None
)

print("\n")
print("=" * 80)
print("FINAL SUB-024 BEHAVIORAL SUMMARY")
print("=" * 80)

print(f"Trials:       {total_trials}")
print(f"Correct:      {total_correct}")
print(f"Wrong:        {total_wrong}")
print(f"Unresolved:   {total_unresolved}")

if overall_accuracy is not None:
    print(f"Accuracy:     {overall_accuracy:.2f}%")
else:
    print("Accuracy:     N/A")

print(f"Need CHECK:   {total_check}")

print("\nSaved:")
print(trial_output)

print("\nSaved:")
print(summary_output)

print("\n")
print("=" * 80)
print("SAFETY CHECK")
print("=" * 80)
print("NO EEG FILES MODIFIED.")
print("NO EPOCHS DELETED.")
print("NO SUBJECTS DELETED.")
print("NO DATA EXCLUDED.")
print("=" * 80)

