from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

EVENT_MAP = (
    BASE / "features" / "behavior_aligned" / "epoch_event_map.csv"
)

BEHAVIOR = (
    BASE / "features" / "behavior_aligned" / "final"
    / "final_behavioral_trials.csv"
)

OUTDIR = (
    BASE / "features" / "scientific_v1" / "merged"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "deterministic_trial_epoch_map.csv"
QC = OUTDIR / "deterministic_trial_epoch_map_qc.csv"

print("=" * 90)
print("DETERMINISTIC TRIAL -> EEG EPOCH MAPPING V2")
print("=" * 90)

events = pd.read_csv(EVENT_MAP)
behavior = pd.read_csv(BEHAVIOR)

print(f"Event-map rows: {len(events):,}")
print(f"Behavior trials: {len(behavior):,}")

required_event = {"file", "subject", "run", "epoch", "event_name"}
required_behavior = {
    "subject", "run", "trial",
    "memory_cond", "probe_type",
    "probe_letter", "behavior_label"
}

missing_event = required_event - set(events.columns)
missing_behavior = required_behavior - set(behavior.columns)

if missing_event:
    raise RuntimeError(f"Missing event columns: {missing_event}")

if missing_behavior:
    raise RuntimeError(f"Missing behavior columns: {missing_behavior}")

# ------------------------------------------------------------------
# Normalize identifiers
# ------------------------------------------------------------------

events["subject"] = events["subject"].astype(str)
events["run"] = events["run"].astype(str)

behavior["subject"] = behavior["subject"].astype(str)
behavior["run"] = behavior["run"].astype(str)

events["epoch"] = pd.to_numeric(events["epoch"], errors="coerce")
behavior["trial"] = pd.to_numeric(behavior["trial"], errors="coerce")

events = events.dropna(subset=["epoch"])
behavior = behavior.dropna(subset=["trial"])

events["epoch"] = events["epoch"].astype(int)
behavior["trial"] = behavior["trial"].astype(int)

# ------------------------------------------------------------------
# Important:
# We reconstruct trial boundaries from show_cross events.
#
# Every show_cross starts a new trial.
# All events until the next show_cross belong to that trial.
# ------------------------------------------------------------------

events = events.sort_values(
    ["subject", "run", "epoch"]
).reset_index(drop=True)

events["trial"] = (
    events.groupby(["subject", "run"])["event_name"]
    .transform(lambda x: (x == "show_cross").cumsum())
)

# Remove events before first trial boundary
events = events[events["trial"] > 0].copy()

print()
print("Reconstructed trial IDs from show_cross.")
print(f"Usable event rows: {len(events):,}")
print(f"Reconstructed trials: {events[['subject','run','trial']].drop_duplicates().shape[0]:,}")

# ------------------------------------------------------------------
# Build trial -> epoch mapping
# ------------------------------------------------------------------

rows = []

for _, b in behavior.iterrows():

    subject = b["subject"]
    run = b["run"]
    trial = int(b["trial"])

    e = events[
        (events["subject"] == subject) &
        (events["run"] == run) &
        (events["trial"] == trial)
    ].copy()

    if len(e) == 0:
        continue

    # Keep all EEG epochs belonging to this trial.
    # This preserves the complete temporal structure.
    for _, er in e.iterrows():

        row = b.to_dict()

        row["file"] = er["file"]
        row["epoch"] = int(er["epoch"])
        row["event_code"] = er["event_code"]
        row["event_name"] = er["event_name"]

        rows.append(row)

mapping = pd.DataFrame(rows)

if len(mapping) == 0:
    raise RuntimeError(
        "No trial-to-epoch mappings were created. "
        "Stop here; do not continue to ML."
    )

# ------------------------------------------------------------------
# QC
# ------------------------------------------------------------------

behavior_keys = behavior[
    ["subject", "run", "trial"]
].drop_duplicates()

mapped_keys = mapping[
    ["subject", "run", "trial"]
].drop_duplicates()

merged_keys = behavior_keys.merge(
    mapped_keys,
    on=["subject", "run", "trial"],
    how="left",
    indicator=True
)

mapped_trials = (merged_keys["_merge"] == "both").sum()
unmapped_trials = (merged_keys["_merge"] == "left_only").sum()

qc = pd.DataFrame({
    "metric": [
        "behavior_trials",
        "mapped_trials",
        "unmapped_trials",
        "mapping_rows",
        "subjects",
        "runs",
    ],
    "value": [
        len(behavior_keys),
        mapped_trials,
        unmapped_trials,
        len(mapping),
        mapping["subject"].nunique(),
        mapping[["subject", "run"]].drop_duplicates().shape[0],
    ]
})

mapping.to_csv(OUT, index=False)
qc.to_csv(QC, index=False)

print()
print("=" * 90)
print("MAPPING COMPLETE")
print("=" * 90)

print(f"Behavior trials:       {len(behavior_keys):,}")
print(f"Mapped trials:         {mapped_trials:,}")
print(f"Unmapped trials:       {unmapped_trials:,}")
print(f"Mapping rows:          {len(mapping):,}")
print(f"Subjects:              {mapping['subject'].nunique()}")
print(
    f"Subject-runs:          "
    f"{mapping[['subject','run']].drop_duplicates().shape[0]}"
)

print()
print("EVENT TYPES IN MAPPING:")
print(mapping["event_name"].value_counts())

print()
print("SAVED:")
print(OUT)
print(QC)

print()
print("=" * 90)

if unmapped_trials == 0:
    print("STATUS: PASS - ALL BEHAVIOR TRIALS HAVE EEG EPOCH MAPPINGS")
else:
    print("STATUS: REVIEW REQUIRED - SOME BEHAVIOR TRIALS ARE UNMAPPED")

print("=" * 90)