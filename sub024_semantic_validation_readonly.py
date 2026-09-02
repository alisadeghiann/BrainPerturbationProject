import pandas as pd
from pathlib import Path
import numpy as np

# ============================================================
# SUB-024 FINAL BEHAVIORAL SEMANTIC VALIDATION
# READ ONLY
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

EVENT_DIR = BASE / "qc" / "events"

OUT_DIR = (
    BASE
    / "qc"
    / "trial_anomaly_inspection"
    / "sub024_final_behavior_review"
    / "behavior_response_audit"
    / "semantic_validation"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("SUB-024 FINAL BEHAVIORAL SEMANTIC VALIDATION - READ ONLY")
print("=" * 90)

# ============================================================
# FIND SUB-024 EVENT FILES
# ============================================================

event_files = sorted(
    EVENT_DIR.glob(
        "sub-024_*_task-WorkingMemory_run-*_eeg_events.csv"
    )
)

if not event_files:
    print("\nERROR: No Sub-024 event files found.")
    raise SystemExit(1)

print(f"\nEvent files found: {len(event_files)}")

# ============================================================
# STORAGE
# ============================================================

all_trials = []
run_summaries = []

# ============================================================
# PROCESS RUNS
# ============================================================

for event_file in event_files:

    name = event_file.name

    print("\n" + "=" * 90)
    print(f"PROCESSING: {name}")
    print("=" * 90)

    # --------------------------------------------------------
    # RUN NUMBER
    # --------------------------------------------------------

    run_text = [
        x for x in name.split("_")
        if x.startswith("run-")
    ]

    if not run_text:
        print("RUN NUMBER NOT FOUND - SKIPPING")
        continue

    run = int(
        run_text[0].replace("run-", "")
    )

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    df = pd.read_csv(event_file)

    print(f"Events: {len(df)}")

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    for col in [
        "type",
        "task_role",
        "value"
    ]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )

    df["trial_num"] = pd.to_numeric(
        df["trial"],
        errors="coerce"
    )

    trial_ids = sorted(
        df["trial_num"]
        .dropna()
        .unique()
    )

    print(f"Trials found: {len(trial_ids)}")

    # --------------------------------------------------------
    # PROCESS TRIALS
    # --------------------------------------------------------

    for trial in trial_ids:

        t = df[
            df["trial_num"] == trial
        ].copy()

        # ====================================================
        # MEMORY CONDITION
        # ====================================================

        if (
            "memory_cond" in t.columns
            and t["memory_cond"].notna().any()
        ):
            memory_cond = (
                t["memory_cond"]
                .dropna()
                .iloc[0]
            )
        else:
            memory_cond = None

        # ====================================================
        # PRESENTED LETTERS
        # ====================================================

        remember_rows = t[
            t["task_role"] == "to_remember"
        ]

        ignore_rows = t[
            t["task_role"] == "to_ignore"
        ]

        remembered_letters = []

        for x in remember_rows["letter"].tolist():
            if pd.notna(x):
                remembered_letters.append(
                    str(x).strip().upper()
                )

        ignored_letters = []

        for x in ignore_rows["letter"].tolist():
            if pd.notna(x):
                ignored_letters.append(
                    str(x).strip().upper()
                )

        # ====================================================
        # PROBE
        # ====================================================

        target_rows = t[
            t["task_role"] == "probe_target"
        ]

        notshown_rows = t[
            t["task_role"] == "probe_not_shown"
        ]

        if len(target_rows) > 0:

            probe_role = "probe_target"

            probe_letter = (
                target_rows.iloc[-1]["letter"]
            )

        elif len(notshown_rows) > 0:

            probe_role = "probe_not_shown"

            probe_letter = (
                notshown_rows.iloc[-1]["letter"]
            )

        else:

            probe_role = None
            probe_letter = None

        if pd.notna(probe_letter):
            probe_letter = (
                str(probe_letter)
                .strip()
                .upper()
            )
        else:
            probe_letter = None

        # ====================================================
        # TRUE GROUND TRUTH
        # ====================================================

        in_remembered = (
            probe_letter in remembered_letters
            if probe_letter is not None
            else None
        )

        in_ignored = (
            probe_letter in ignored_letters
            if probe_letter is not None
            else None
        )

        # ----------------------------------------------------
        # Semantic ground truth based on event labels
        # ----------------------------------------------------

        if probe_role == "probe_target":
            semantic_truth = "target"

        elif probe_role == "probe_not_shown":
            semantic_truth = "not_target"

        else:
            semantic_truth = "unresolved"

        # ----------------------------------------------------
        # Cross-check target against remembered letters
        # ----------------------------------------------------

        if probe_role == "probe_target":

            if in_remembered is True:
                probe_memory_consistency = "consistent"

            elif in_remembered is False:
                probe_memory_consistency = "MISMATCH"

            else:
                probe_memory_consistency = "unknown"

        elif probe_role == "probe_not_shown":

            if in_remembered is False:
                probe_memory_consistency = "consistent"

            elif in_remembered is True:
                probe_memory_consistency = "MISMATCH"

            else:
                probe_memory_consistency = "unknown"

        else:
            probe_memory_consistency = "unknown"

        # ====================================================
        # RESPONSE
        # ====================================================

        response_rows = t[
            t["task_role"].isin(
                [
                    "ignored_correct",
                    "ignored_incorrect"
                ]
            )
        ]

        if len(response_rows) > 0:

            response_role = (
                response_rows.iloc[0]["task_role"]
            )

            response_latency = (
                response_rows.iloc[0]["latency"]
            )

        else:

            response_role = None
            response_latency = None

        # ====================================================
        # FEEDBACK
        # ====================================================

        feedback_correct_rows = t[
            t["task_role"] == "feedback_correct"
        ]

        feedback_wrong_rows = t[
            t["task_role"] == "feedback_incorrect"
        ]

        if len(feedback_correct_rows) > 0:

            feedback = "correct"

        elif len(feedback_wrong_rows) > 0:

            feedback = "wrong"

        else:

            feedback = None

        # ====================================================
        # RESPONSE SEMANTICS
        # ====================================================

        if response_role == "ignored_correct":

            response_result = "correct"

        elif response_role == "ignored_incorrect":

            response_result = "wrong"

        else:

            response_result = None

        # ====================================================
        # RESPONSE-FEEDBACK CONSISTENCY
        # ====================================================

        if (
            response_result is not None
            and feedback is not None
        ):

            response_feedback_consistent = (
                response_result == feedback
            )

        else:

            response_feedback_consistent = None

        # ====================================================
        # TRUE BEHAVIOR
        # ====================================================

        # We first trust the probe's explicit task role.
        # Then compare participant result against that truth.
        #
        # Because the original response labels are
        # "ignored_correct"/"ignored_incorrect", we do NOT
        # assume that their literal word "ignored" means
        # anything about the participant's answer.

        if semantic_truth in [
            "target",
            "not_target"
        ]:

            if response_result == "correct":
                semantic_behavior = "correct"

            elif response_result == "wrong":
                semantic_behavior = "wrong"

            else:
                semantic_behavior = "unresolved"

        else:

            semantic_behavior = "unresolved"

        # ====================================================
        # RESPONSE / PROBE LOGICAL CROSS-CHECK
        # ====================================================

        logical_status = "OK"
        logical_issue = ""

        if probe_role is None:
            logical_status = "CHECK"
            logical_issue = "missing_probe"

        elif probe_memory_consistency == "MISMATCH":
            logical_status = "CHECK"
            logical_issue = "probe_memory_mismatch"

        elif response_result is None:
            logical_status = "CHECK"
            logical_issue = "missing_response"

        elif feedback is None:
            logical_status = "CHECK"
            logical_issue = "missing_feedback"

        elif response_feedback_consistent is False:
            logical_status = "CHECK"
            logical_issue = "response_feedback_mismatch"

        # ====================================================
        # RT
        # ====================================================

        probe_latency = None

        probe_rows = t[
            t["task_role"].isin(
                [
                    "probe_target",
                    "probe_not_shown"
                ]
            )
        ]

        if len(probe_rows) > 0:

            probe_latency = (
                probe_rows.iloc[-1]["latency"]
            )

        try:

            if (
                probe_latency is not None
                and response_latency is not None
            ):

                rt_ms = (
                    float(response_latency)
                    - float(probe_latency)
                )

            else:

                rt_ms = np.nan

        except Exception:

            rt_ms = np.nan

        # ====================================================
        # CLICK COUNT
        # ====================================================

        click_count = int(
            (
                t["type"] == "left_click"
            ).sum()
        )

        # ====================================================
        # SAVE TRIAL
        # ====================================================

        all_trials.append(
            {
                "subject": "sub-024",
                "run": run,
                "trial": int(trial),

                "memory_cond": memory_cond,

                "remembered_letters":
                    ",".join(remembered_letters),

                "ignored_letters":
                    ",".join(ignored_letters),

                "probe_role":
                    probe_role,

                "probe_letter":
                    probe_letter,

                "probe_in_remembered":
                    in_remembered,

                "probe_in_ignored":
                    in_ignored,

                "probe_memory_consistency":
                    probe_memory_consistency,

                "semantic_truth":
                    semantic_truth,

                "response_role":
                    response_role,

                "response_result":
                    response_result,

                "feedback":
                    feedback,

                "response_feedback_consistent":
                    response_feedback_consistent,

                "semantic_behavior":
                    semantic_behavior,

                "probe_latency":
                    probe_latency,

                "response_latency":
                    response_latency,

                "rt_ms":
                    rt_ms,

                "click_count":
                    click_count,

                "logical_status":
                    logical_status,

                "logical_issue":
                    logical_issue,

                "source_file":
                    name
            }
        )

    # ========================================================
    # RUN SUMMARY
    # ========================================================

    run_trials = [
        x for x in all_trials
        if x["run"] == run
    ]

    rdf = pd.DataFrame(run_trials)

    total = len(rdf)

    correct = int(
        (
            rdf["semantic_behavior"]
            == "correct"
        ).sum()
    )

    wrong = int(
        (
            rdf["semantic_behavior"]
            == "wrong"
        ).sum()
    )

    unresolved = int(
        (
            rdf["semantic_behavior"]
            == "unresolved"
        ).sum()
    )

    checks = int(
        (
            rdf["logical_status"]
            == "CHECK"
        ).sum()
    )

    probe_mismatches = int(
        (
            rdf["probe_memory_consistency"]
            == "MISMATCH"
        ).sum()
    )

    feedback_mismatches = int(
        (
            rdf["response_feedback_consistent"]
            == False
        ).sum()
    )

    accuracy = (
        correct / (correct + wrong) * 100
        if (correct + wrong) > 0
        else np.nan
    )

    run_summaries.append(
        {
            "subject": "sub-024",
            "run": run,
            "trials": total,
            "correct": correct,
            "wrong": wrong,
            "unresolved": unresolved,
            "accuracy_percent": accuracy,
            "probe_memory_mismatches":
                probe_mismatches,
            "response_feedback_mismatches":
                feedback_mismatches,
            "trials_needing_check":
                checks
        }
    )


# ============================================================
# FINAL DATAFRAMES
# ============================================================

trial_df = pd.DataFrame(all_trials)

run_df = pd.DataFrame(run_summaries)


# ============================================================
# FINAL SUMMARY
# ============================================================

total = len(trial_df)

correct = int(
    (
        trial_df["semantic_behavior"]
        == "correct"
    ).sum()
)

wrong = int(
    (
        trial_df["semantic_behavior"]
        == "wrong"
    ).sum()
)

unresolved = int(
    (
        trial_df["semantic_behavior"]
        == "unresolved"
    ).sum()
)

checks = int(
    (
        trial_df["logical_status"]
        == "CHECK"
    ).sum()
)

probe_mismatches = int(
    (
        trial_df["probe_memory_consistency"]
        == "MISMATCH"
    ).sum()
)

feedback_mismatches = int(
    (
        trial_df["response_feedback_consistent"]
        == False
    ).sum()
)

accuracy = (
    correct / (correct + wrong) * 100
    if (correct + wrong) > 0
    else np.nan
)


# ============================================================
# SAVE
# ============================================================

trial_output = (
    OUT_DIR
    / "sub024_semantic_validation_trial_level.csv"
)

run_output = (
    OUT_DIR
    / "sub024_semantic_validation_run_level.csv"
)

summary_output = (
    OUT_DIR
    / "sub024_semantic_validation_summary.csv"
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

summary_df = pd.DataFrame(
    [
        {
            "subject": "sub-024",
            "trials": total,
            "correct": correct,
            "wrong": wrong,
            "unresolved": unresolved,
            "accuracy_percent": accuracy,
            "probe_memory_mismatches":
                probe_mismatches,
            "response_feedback_mismatches":
                feedback_mismatches,
            "trials_needing_check":
                checks
        }
    ]
)

summary_df.to_csv(
    summary_output,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("FINAL SEMANTIC VALIDATION - SUB-024")
print("=" * 90)

print(f"Trials:                    {total}")
print(f"Semantic correct:          {correct}")
print(f"Semantic wrong:            {wrong}")
print(f"Unresolved:                {unresolved}")

if not np.isnan(accuracy):
    print(f"Accuracy:                  {accuracy:.2f}%")
else:
    print("Accuracy:                  NA")

print(f"Probe-memory mismatches:   {probe_mismatches}")
print(f"Response-feedback mismatch:{feedback_mismatches}")
print(f"Trials needing CHECK:      {checks}")


print("\n")
print("=" * 90)
print("RUN-LEVEL SEMANTIC VALIDATION")
print("=" * 90)

print(
    run_df.to_string(index=False)
)


print("\n")
print("=" * 90)
print("PROBE MEMORY CONSISTENCY")
print("=" * 90)

print(
    trial_df[
        "probe_memory_consistency"
    ].value_counts(dropna=False)
)


print("\n")
print("=" * 90)
print("RESPONSE / FEEDBACK CONSISTENCY")
print("=" * 90)

print(
    trial_df[
        "response_feedback_consistent"
    ].value_counts(dropna=False)
)


print("\n")
print("=" * 90)
print("SEMANTIC BEHAVIOR")
print("=" * 90)

print(
    trial_df[
        "semantic_behavior"
    ].value_counts(dropna=False)
)


# ============================================================
# SHOW POTENTIAL PROBLEMS
# ============================================================

problem_df = trial_df[
    (
        trial_df["logical_status"] == "CHECK"
    )
    |
    (
        trial_df["probe_memory_consistency"]
        == "MISMATCH"
    )
]

print("\n")
print("=" * 90)
print("POTENTIAL SEMANTIC PROBLEMS")
print("=" * 90)

if len(problem_df) == 0:

    print("NO SEMANTIC PROBLEMS FOUND.")

else:

    print(
        problem_df[
            [
                "run",
                "trial",
                "remembered_letters",
                "ignored_letters",
                "probe_role",
                "probe_letter",
                "probe_memory_consistency",
                "semantic_truth",
                "response_role",
                "response_result",
                "feedback",
                "semantic_behavior",
                "logical_status",
                "logical_issue"
            ]
        ].to_string(index=False)
    )


# ============================================================
# COMPARISON WITH PREVIOUS AUDIT
# ============================================================

print("\n")
print("=" * 90)
print("COMPARISON WITH PREVIOUS BEHAVIORAL AUDIT")
print("=" * 90)

previous_correct = 31
previous_wrong = 44
previous_accuracy = 41.333333

print(f"Previous audit:  {previous_correct} correct / {previous_wrong} wrong / {previous_accuracy:.2f}%")
print(f"Semantic audit:  {correct} correct / {wrong} wrong / {accuracy:.2f}%")

if (
    correct == previous_correct
    and wrong == previous_wrong
):

    print("\nRESULT: SEMANTIC AUDIT MATCHES PREVIOUS AUDIT.")

else:

    print("\nRESULT: SEMANTIC AUDIT DIFFERS FROM PREVIOUS AUDIT.")
    print("THIS REQUIRES REVIEW BEFORE ANY SUBJECT DECISION.")


# ============================================================
# SAFETY
# ============================================================

print("\n")
print("=" * 90)
print("SAFETY CHECK")
print("=" * 90)

print("READ-ONLY.")
print("NO EEG FILES MODIFIED.")
print("NO EPOCHS DELETED.")
print("NO SUBJECTS DELETED.")
print("NO DATA EXCLUDED.")

print("\nSaved:")
print(trial_output)
print(run_output)
print(summary_output)

print("=" * 90)

