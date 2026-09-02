import pandas as pd
from pathlib import Path
import re

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

TRIAL_FILE = (
    BASE
    / "features"
    / "behavior_aligned"
    / "trial_level_behavior.csv"
)

QC_EVENTS = BASE / "qc" / "events"
BIDS_DATA = BASE / "data"

OUTPUT_DIR = BASE / "features" / "behavior_aligned"
OUTPUT = OUTPUT_DIR / "trial_level_behavior_full.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("TRIAL-LEVEL BEHAVIORAL RECONSTRUCTION - FULL")
print("=" * 90)

trials = pd.read_csv(TRIAL_FILE)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def find_event_file(subject, run):
    """
    Find event source for subject/run.
    Priority:
    1. BIDS events.tsv
    2. QC events.csv
    """

    sub_num = subject.replace("sub-", "")
    run_num = int(str(run).replace("run-", ""))

    bids_dir = (
        BIDS_DATA
        / subject
        / "ses-01"
        / "eeg"
    )

    bids_candidates = list(
        bids_dir.glob(
            f"{subject}_ses-01_task-WorkingMemory_run-{run_num}_events.tsv"
        )
    )

    if bids_candidates:
        return bids_candidates[0], "BIDS"

    qc_candidates = list(
        QC_EVENTS.glob(
            f"{subject}_ses-01_task-WorkingMemory_run-{run_num}_eeg_events.csv"
        )
    )

    if qc_candidates:
        return qc_candidates[0], "QC"

    return None, None


def load_events(path):
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")

    return pd.read_csv(path)


def clean_value(x):
    if pd.isna(x):
        return ""

    return str(x).strip()


# ---------------------------------------------------------
# Process each trial
# ---------------------------------------------------------

records = []

for i, row in trials.iterrows():

    subject = row["subject"]
    run = row["run"]
    trial_num = int(row["trial"])

    event_file, source = find_event_file(subject, run)

    if event_file is None:
        records.append({
            **row.to_dict(),
            "event_source": "MISSING",
            "memory_cond": pd.NA,
            "remember_count": pd.NA,
            "ignore_count": pd.NA,
            "remember_letters": pd.NA,
            "ignore_letters": pd.NA,
            "probe_type": pd.NA,
            "probe_letter": pd.NA,
            "behavior_outcome": row.get("feedback", "unknown"),
            "behavior_label": pd.NA,
            "alignment_status": "NO_EVENT_SOURCE"
        })
        continue

    try:
        ev = load_events(event_file)
    except Exception as e:
        records.append({
            **row.to_dict(),
            "event_source": source,
            "memory_cond": pd.NA,
            "remember_count": pd.NA,
            "ignore_count": pd.NA,
            "remember_letters": pd.NA,
            "ignore_letters": pd.NA,
            "probe_type": pd.NA,
            "probe_letter": pd.NA,
            "behavior_outcome": row.get("feedback", "unknown"),
            "behavior_label": pd.NA,
            "alignment_status": "EVENT_READ_ERROR"
        })
        continue

    # -----------------------------------------------------
    # Normalize columns
    # -----------------------------------------------------

    for c in ["trial", "task_role", "letter", "memory_cond", "type", "value"]:
        if c not in ev.columns:
            ev[c] = pd.NA

    ev["trial_num_internal"] = pd.to_numeric(
        ev["trial"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Match trial
    # -----------------------------------------------------

    g = ev[
        ev["trial_num_internal"] == trial_num
    ].copy()

    # Some BIDS files may not contain explicit trial numbers.
    # In that case alignment cannot safely be inferred.
    if len(g) == 0:

        records.append({
            **row.to_dict(),
            "event_source": source,
            "memory_cond": pd.NA,
            "remember_count": pd.NA,
            "ignore_count": pd.NA,
            "remember_letters": pd.NA,
            "ignore_letters": pd.NA,
            "probe_type": pd.NA,
            "probe_letter": pd.NA,
            "behavior_outcome": row.get("feedback", "unknown"),
            "behavior_label": pd.NA,
            "alignment_status": "TRIAL_NOT_FOUND"
        })

        continue

    # -----------------------------------------------------
    # memory condition
    # -----------------------------------------------------

    mem_values = (
        pd.to_numeric(
            g["memory_cond"],
            errors="coerce"
        )
        .dropna()
        .unique()
    )

    if len(mem_values) == 1:
        memory_cond = int(mem_values[0])
    elif len(mem_values) > 1:
        memory_cond = "|".join(
            str(int(x)) for x in sorted(mem_values)
        )
    else:
        memory_cond = pd.NA

    # -----------------------------------------------------
    # Remember / Ignore
    # -----------------------------------------------------

    task_roles = g["task_role"].fillna("").astype(str)

    remember = g[
        task_roles.str.contains(
            "to_remember",
            case=False,
            na=False
        )
    ]

    ignore = g[
        task_roles.str.contains(
            "to_ignore",
            case=False,
            na=False
        )
    ]

    remember_letters = [
        clean_value(x)
        for x in remember["letter"]
        if clean_value(x)
    ]

    ignore_letters = [
        clean_value(x)
        for x in ignore["letter"]
        if clean_value(x)
    ]

    # -----------------------------------------------------
    # Probe
    # -----------------------------------------------------

    probe_target = g[
        task_roles.str.contains(
            "probe_target",
            case=False,
            na=False
        )
    ]

    probe_not_shown = g[
        task_roles.str.contains(
            "probe_not_shown",
            case=False,
            na=False
        )
    ]

    if len(probe_target) > 0:
        probe_type = "target"

        probe_letters = [
            clean_value(x)
            for x in probe_target["letter"]
            if clean_value(x)
        ]

    elif len(probe_not_shown) > 0:
        probe_type = "not_shown"

        probe_letters = [
            clean_value(x)
            for x in probe_not_shown["letter"]
            if clean_value(x)
        ]

    else:
        probe_type = pd.NA
        probe_letters = []

    probe_letter = (
        probe_letters[0]
        if probe_letters
        else pd.NA
    )

    # -----------------------------------------------------
    # Behavioral response
    # -----------------------------------------------------

    roles = task_roles.tolist()

    if "remembered_correct" in roles:
        behavior_label = "remembered_correct"

    elif "ignored_correct" in roles:
        behavior_label = "ignored_correct"

    elif "remembered_incorrect" in roles:
        behavior_label = "remembered_incorrect"

    elif "ignored_incorrect" in roles:
        behavior_label = "ignored_incorrect"

    elif "sound_beep" in g["type"].fillna("").astype(str).tolist():
        behavior_label = "correct"

    elif "sound_buzz" in g["type"].fillna("").astype(str).tolist():
        behavior_label = "incorrect"

    else:
        behavior_label = pd.NA

    # -----------------------------------------------------
    # Outcome
    # -----------------------------------------------------

    if "sound_beep" in g["type"].fillna("").astype(str).tolist():
        outcome = "correct"

    elif "sound_buzz" in g["type"].fillna("").astype(str).tolist():
        outcome = "incorrect"

    else:
        outcome = "unknown"

    # -----------------------------------------------------
    # Alignment status
    # -----------------------------------------------------

    alignment_ok = (
        len(remember_letters) > 0
        or len(ignore_letters) > 0
    ) and (
        probe_type in ["target", "not_shown"]
    )

    alignment_status = (
        "ALIGNED"
        if alignment_ok
        else "PARTIAL"
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    records.append({
        **row.to_dict(),

        "event_source": source,

        "memory_cond": memory_cond,

        "remember_count": len(remember_letters),
        "ignore_count": len(ignore_letters),

        "remember_letters": "|".join(
            remember_letters
        ),

        "ignore_letters": "|".join(
            ignore_letters
        ),

        "probe_type": probe_type,

        "probe_letter": probe_letter,

        "behavior_outcome": outcome,

        "behavior_label": behavior_label,

        "alignment_status": alignment_status
    })


# ---------------------------------------------------------
# DataFrame
# ---------------------------------------------------------

out = pd.DataFrame(records)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

out.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print()
print("=" * 90)
print("FULL BEHAVIORAL RECONSTRUCTION COMPLETE")
print("=" * 90)

print(f"Trials:       {len(out):,}")
print(f"Subjects:     {out['subject'].nunique()}")
print(
    f"Runs:         "
    f"{out[['subject','run']].drop_duplicates().shape[0]:,}"
)

print()
print("EVENT SOURCES:")
print(
    out["event_source"]
    .value_counts(dropna=False)
)

print()
print("MEMORY CONDITIONS:")
print(
    out["memory_cond"]
    .value_counts(dropna=False)
    .sort_index()
)

print()
print("PROBE TYPE:")
print(
    out["probe_type"]
    .value_counts(dropna=False)
)

print()
print("BEHAVIOR OUTCOME:")
print(
    out["behavior_outcome"]
    .value_counts(dropna=False)
)

print()
print("BEHAVIOR LABEL:")
print(
    out["behavior_label"]
    .value_counts(dropna=False)
)

print()
print("ALIGNMENT STATUS:")
print(
    out["alignment_status"]
    .value_counts(dropna=False)
)

print()
print("Saved:")
print(OUTPUT)

print("=" * 90)
