import os
import glob
import numpy as np
import pandas as pd
import mne
from collections import Counter

# ============================================================
# CONDITION HARMONIZATION QC - 82 RUNS
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

EPOCH_DIR = os.path.join(BASE_DIR, "epochs_clean")
LOG_DIR = os.path.join(EPOCH_DIR, "logs", "condition_harmonization")

os.makedirs(LOG_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    LOG_DIR, "condition_harmonization_qc_82runs.csv"
)

SUBJECT_CSV = os.path.join(
    LOG_DIR, "subject_condition_summary_82runs.csv"
)

RUN_CONDITION_CSV = os.path.join(
    LOG_DIR, "run_condition_counts_82runs.csv"
)

SUMMARY_TXT = os.path.join(
    LOG_DIR, "condition_harmonization_qc_82runs_summary.txt"
)

# ------------------------------------------------------------
# EXPECTED CONDITIONS
# ------------------------------------------------------------

EXPECTED_CONDITIONS = [
    "left_click",
    "right_click",
    "show_cross",
    "show_dash",
    "show_letter",
    "sound_beep",
    "sound_buzz",
]

# ------------------------------------------------------------
# FIND FILES
# ------------------------------------------------------------

files = sorted(
    glob.glob(
        os.path.join(
            EPOCH_DIR,
            "*_clean_epo.fif"
        )
    )
)

print("=" * 80)
print("CONDITION HARMONIZATION QC - 82 RUNS")
print("=" * 80)

print()
print(f"Epoch files found: {len(files)}")

if len(files) != 82:
    print()
    print("WARNING:")
    print(f"Expected 82 files, found {len(files)}")

# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

run_records = []
subject_condition_counter = {}
global_counter = Counter()

# ------------------------------------------------------------
# PROCESS
# ------------------------------------------------------------

for idx, filepath in enumerate(files, 1):

    fname = os.path.basename(filepath)

    print()
    print("=" * 80)
    print(f"[{idx}/{len(files)}] {fname}")
    print("=" * 80)

    try:

        epochs = mne.read_epochs(
            filepath,
            preload=False,
            verbose=False
        )

        n_epochs = len(epochs)

        # ----------------------------------------------------
        # SUBJECT / RUN
        # ----------------------------------------------------

        parts = fname.split("_")

        subject = parts[0]

        run = None

        for p in parts:
            if p.startswith("run-"):
                try:
                    run = int(p.replace("run-", ""))
                except:
                    pass

        print(f"Subject:          {subject}")
        print(f"Run:              {run}")
        print(f"Epochs:           {n_epochs}")
        print(f"Channels:         {len(epochs.ch_names)}")
        print(f"Sampling rate:    {epochs.info['sfreq']}")

        # ----------------------------------------------------
        # EVENT EXTRACTION
        # ----------------------------------------------------

        event_names = []

        if hasattr(epochs, "events") and epochs.events is not None:

            event_ids = epochs.event_id

            if event_ids:

                inverse_event_id = {
                    int(v): str(k)
                    for k, v in event_ids.items()
                }

                for code in epochs.events[:, 2]:

                    if int(code) in inverse_event_id:
                        event_names.append(
                            inverse_event_id[int(code)]
                        )

        counts = Counter(event_names)

        # ----------------------------------------------------
        # FALLBACK: METADATA
        # ----------------------------------------------------

        if len(event_names) == 0:

            try:

                if epochs.metadata is not None:

                    for col in [
                        "event_type",
                        "condition",
                        "type",
                        "event"
                    ]:

                        if col in epochs.metadata.columns:

                            event_names = [
                                str(x)
                                for x in epochs.metadata[col]
                            ]

                            counts = Counter(event_names)
                            break

            except Exception:
                pass

        # ----------------------------------------------------
        # PRINT CONDITIONS
        # ----------------------------------------------------

        print()
        print("CONDITIONS")

        for condition in EXPECTED_CONDITIONS:

            count = counts.get(condition, 0)

            print(
                f"{condition:<25} {count:>6}"
            )

            global_counter[condition] += count

        # ----------------------------------------------------
        # UNKNOWN CONDITIONS
        # ----------------------------------------------------

        unknown = {
            k: v
            for k, v in counts.items()
            if k not in EXPECTED_CONDITIONS
        }

        if unknown:

            print()
            print("UNKNOWN CONDITIONS")

            for k, v in unknown.items():
                print(f"{k:<25} {v}")

        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        counted_total = sum(
            counts.get(c, 0)
            for c in EXPECTED_CONDITIONS
        )

        missing_conditions = [
            c
            for c in EXPECTED_CONDITIONS
            if counts.get(c, 0) == 0
        ]

        # ----------------------------------------------------
        # MAJOR CONDITION BALANCE
        #
        # Do NOT classify a run as BAD merely because
        # sound_beep/buzz are smaller.
        # Primary task conditions are:
        # show_letter
        # show_cross
        # show_dash
        # left/right click
        # ----------------------------------------------------

        primary = [
            "show_letter",
            "show_cross",
            "show_dash",
            "left_click",
            "right_click",
        ]

        primary_counts = [
            counts.get(c, 0)
            for c in primary
        ]

        positive_primary = [
            x for x in primary_counts
            if x > 0
        ]

        if positive_primary:

            primary_min = min(positive_primary)
            primary_max = max(positive_primary)

            primary_ratio = (
                primary_min / primary_max
                if primary_max > 0
                else 0
            )

        else:

            primary_ratio = 0

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        reasons = []

        if len(event_names) == 0:
            reasons.append("NO_EVENT_LABELS")

        if missing_conditions:
            reasons.append(
                "MISSING:" +
                ",".join(missing_conditions)
            )

        if counted_total != n_epochs:
            reasons.append(
                f"COUNT_MISMATCH_{counted_total}_vs_{n_epochs}"
            )

        if unknown:
            reasons.append(
                "UNKNOWN_CONDITIONS"
            )

        # Primary balance is informational.
        # It should NOT reject the run because
        # experimental designs can intentionally be unequal.

        if not reasons:

            status = "PASS"

        else:

            status = "REVIEW"

        # ----------------------------------------------------
        # SUBJECT COUNTS
        # ----------------------------------------------------

        if subject not in subject_condition_counter:
            subject_condition_counter[subject] = Counter()

        for condition in EXPECTED_CONDITIONS:
            subject_condition_counter[subject][condition] += (
                counts.get(condition, 0)
            )

        # ----------------------------------------------------
        # RUN RECORD
        # ----------------------------------------------------

        record = {
            "subject": subject,
            "run": run,
            "file": fname,
            "n_epochs": n_epochs,
            "counted_events": counted_total,
            "primary_balance_ratio": primary_ratio,
            "status": status,
            "reasons": ";".join(reasons),
        }

        for condition in EXPECTED_CONDITIONS:
            record[condition] = counts.get(condition, 0)

        run_records.append(record)

        print()
        print(
            f"Primary balance ratio: {primary_ratio:.3f}"
        )

        print(
            f"Counted events: {counted_total}/{n_epochs}"
        )

        print(
            f"STATUS: {status}"
        )

        if reasons:
            print(
                "REASONS: " +
                ";".join(reasons)
            )

    except Exception as e:

        print()
        print("ERROR:")
        print(str(e))

        run_records.append({
            "subject": fname.split("_")[0],
            "run": run,
            "file": fname,
            "n_epochs": 0,
            "counted_events": 0,
            "primary_balance_ratio": np.nan,
            "status": "ERROR",
            "reasons": str(e),
        })

# ============================================================
# DATAFRAMES
# ============================================================

run_df = pd.DataFrame(run_records)

subject_records = []

for subject in sorted(subject_condition_counter):

    counter = subject_condition_counter[subject]

    total = sum(counter.values())

    row = {
        "subject": subject,
        "total_epochs": total,
    }

    for condition in EXPECTED_CONDITIONS:
        row[condition] = counter.get(condition, 0)

    subject_records.append(row)

subject_df = pd.DataFrame(subject_records)

# ============================================================
# GLOBAL CONDITION COUNTS
# ============================================================

global_df = pd.DataFrame({
    "condition": EXPECTED_CONDITIONS,
    "total_epochs": [
        global_counter[c]
        for c in EXPECTED_CONDITIONS
    ]
})

# ============================================================
# RUN CONDITION TABLE
# ============================================================

run_condition_records = []

for record in run_records:

    row = {
        "subject": record.get("subject"),
        "run": record.get("run"),
        "file": record.get("file"),
    }

    for condition in EXPECTED_CONDITIONS:
        row[condition] = record.get(condition, 0)

    run_condition_records.append(row)

run_condition_df = pd.DataFrame(
    run_condition_records
)

# ============================================================
# SAVE
# ============================================================

run_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

subject_df.to_csv(
    SUBJECT_CSV,
    index=False,
    encoding="utf-8-sig"
)

run_condition_df.to_csv(
    RUN_CONDITION_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

status_counts = (
    run_df["status"]
    .value_counts()
    .to_dict()
)

summary_lines = []

summary_lines.append(
    "CONDITION HARMONIZATION QC - 82 RUNS"
)

summary_lines.append("=" * 80)

summary_lines.append(
    f"Epoch files found: {len(files)}"
)

summary_lines.append(
    f"Run records: {len(run_df)}"
)

summary_lines.append("")

summary_lines.append(
    "RUN STATUS COUNTS"
)

for k, v in status_counts.items():

    summary_lines.append(
        f"{k}: {v}"
    )

summary_lines.append("")

summary_lines.append(
    "GLOBAL CONDITION COUNTS"
)

for condition in EXPECTED_CONDITIONS:

    summary_lines.append(
        f"{condition}: {global_counter[condition]}"
    )

summary_lines.append("")

summary_lines.append(
    "IMPORTANT"
)

summary_lines.append(
    "Condition imbalance was NOT used as an automatic exclusion rule."
)

summary_lines.append(
    "Low-frequency conditions such as sound_beep/sound_buzz "
    "are preserved."
)

summary_lines.append(
    "No epoch files were modified."
)

summary_lines.append(
    "No raw/preprocessed data were modified."
)

with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary_lines)
    )

# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("CONDITION HARMONIZATION QC COMPLETE")
print("=" * 80)

print()

print("RUN STATUS COUNTS")

for k, v in status_counts.items():

    print(
        f"{k:<10} {v}"
    )

print()

print("GLOBAL CONDITION COUNTS")

for condition in EXPECTED_CONDITIONS:

    print(
        f"{condition:<25} {global_counter[condition]}"
    )

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_CSV)
print(SUBJECT_CSV)
print(RUN_CONDITION_CSV)
print(SUMMARY_TXT)

print()
print("=" * 80)
print("NO DATA WAS MODIFIED.")
print("NO EPOCH FILE WAS MODIFIED.")
print("=" * 80)