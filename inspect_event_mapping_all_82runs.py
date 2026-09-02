import os
import glob
import mne
import pandas as pd
from collections import Counter, defaultdict

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = (
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
    r"\\epochs_clean\\logs\\perturbation_dataset_v4\\ELIGIBLE"
)

OUTPUT_DIR = (
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
    r"\\epochs_clean\\logs\\perturbation_dataset_v4"
    r"\\event_mapping_investigation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_OUT = os.path.join(
    OUTPUT_DIR,
    "event_mapping_all_82runs.csv"
)

SUMMARY_OUT = os.path.join(
    OUTPUT_DIR,
    "event_mapping_all_82runs_summary.txt"
)

# ============================================================
# EXPECTED
# ============================================================

EXPECTED_FILES = 82

# ============================================================
# FILES
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "*_standardized_epo.fif"
        )
    )
)

print("=" * 80)
print("EVENT MAPPING INVESTIGATION - ALL 82 RUNS")
print("=" * 80)

print()
print(f"Files found: {len(files)}")
print(f"Expected:    {EXPECTED_FILES}")
print()

# ============================================================
# RECORDS
# ============================================================

records = []

global_event_id = defaultdict(set)
global_event_counts = Counter()

# ============================================================
# PROCESS
# ============================================================

for idx, path in enumerate(files, 1):

    fname = os.path.basename(path)

    print("=" * 80)
    print(f"[{idx}/{len(files)}] {fname}")
    print("=" * 80)

    try:

        epochs = mne.read_epochs(
            path,
            preload=False,
            verbose=False
        )

        event_id = epochs.event_id
        events = epochs.events

        inverse = {
            int(v): str(k)
            for k, v in event_id.items()
        }

        unique_ids = sorted(
            set(events[:, 2])
        )

        print()
        print("EVENT_ID:")
        print(event_id)

        print()
        print("EVENTS:")

        for eid in unique_ids:

            count = int(
                (events[:, 2] == eid).sum()
            )

            condition = inverse.get(
                int(eid),
                "UNKNOWN"
            )

            print(
                f"ID {eid:2d} | "
                f"{condition:30s} | "
                f"{count}"
            )

            records.append({
                "file": fname,
                "event_id": int(eid),
                "condition": condition,
                "count": count,
                "n_epochs": len(epochs),
                "sfreq": float(
                    epochs.info["sfreq"]
                ),
                "n_channels": len(
                    epochs.ch_names
                )
            })

            global_event_id[
                int(eid)
            ].add(condition)

            global_event_counts[
                (int(eid), condition)
            ] += count

        print()

    except Exception as e:

        print("ERROR:")
        print(e)

        records.append({
            "file": fname,
            "event_id": -1,
            "condition": "ERROR",
            "count": 0,
            "n_epochs": 0,
            "sfreq": None,
            "n_channels": 0
        })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)

df.to_csv(
    CSV_OUT,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# GLOBAL MAPPING
# ============================================================

print()
print("=" * 80)
print("GLOBAL EVENT ID -> CONDITION MAPPING")
print("=" * 80)
print()

mapping_rows = []

for eid in sorted(global_event_id):

    conditions = sorted(
        global_event_id[eid]
    )

    mapping_rows.append({
        "event_id": eid,
        "conditions": " | ".join(conditions),
        "n_conditions": len(conditions)
    })

    print(
        f"ID {eid:2d} -> "
        f"{' | '.join(conditions)}"
    )

# ============================================================
# AMBIGUOUS IDS
# ============================================================

ambiguous = {
    eid: sorted(conditions)
    for eid, conditions
    in global_event_id.items()
    if len(conditions) > 1
}

print()
print("=" * 80)
print("AMBIGUOUS EVENT IDs")
print("=" * 80)
print()

if len(ambiguous) == 0:

    print("NONE")

else:

    for eid, conditions in ambiguous.items():

        print(
            f"ID {eid}: "
            f"{' | '.join(conditions)}"
        )

# ============================================================
# GLOBAL CONDITION COUNTS
# ============================================================

print()
print("=" * 80)
print("GLOBAL CONDITION COUNTS")
print("=" * 80)
print()

condition_counts = (
    df[
        df["condition"] != "ERROR"
    ]
    .groupby("condition")["count"]
    .sum()
    .sort_values(ascending=False)
)

for condition, count in condition_counts.items():

    print(
        f"{condition:30s} "
        f"{int(count)}"
    )

# ============================================================
# FILE-SPECIFIC MAPPINGS
# ============================================================

print()
print("=" * 80)
print("FILE-SPECIFIC MAPPING SUMMARY")
print("=" * 80)
print()

file_mapping_count = {}

for fname, group in df.groupby("file"):

    mapping = []

    for _, row in group.iterrows():

        mapping.append(
            f"{int(row['event_id'])}="
            f"{row['condition']}"
        )

    mapping_str = "; ".join(mapping)

    file_mapping_count[fname] = mapping_str

    print(
        f"{fname}:"
    )
    print(
        f"  {mapping_str}"
    )

# ============================================================
# UNIQUE MAPPINGS
# ============================================================

unique_mappings = Counter(
    file_mapping_count.values()
)

print()
print("=" * 80)
print("UNIQUE FILE MAPPINGS")
print("=" * 80)
print()

for mapping, count in unique_mappings.items():

    print(
        f"{count} files:"
    )
    print(
        f"  {mapping}"
    )
    print()

# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.append("=" * 80)
summary.append(
    "EVENT MAPPING INVESTIGATION - ALL 82 RUNS"
)
summary.append("=" * 80)
summary.append("")

summary.append(
    f"Files found:    {len(files)}"
)

summary.append(
    f"Expected files: {EXPECTED_FILES}"
)

summary.append("")

summary.append("=" * 80)
summary.append("GLOBAL EVENT ID -> CONDITION")
summary.append("=" * 80)
summary.append("")

for eid in sorted(global_event_id):

    conditions = sorted(
        global_event_id[eid]
    )

    summary.append(
        f"ID {eid} -> "
        f"{' | '.join(conditions)}"
    )

summary.append("")

summary.append("=" * 80)
summary.append("AMBIGUOUS EVENT IDs")
summary.append("=" * 80)
summary.append("")

if len(ambiguous) == 0:

    summary.append("NONE")

else:

    for eid, conditions in ambiguous.items():

        summary.append(
            f"ID {eid}: "
            f"{' | '.join(conditions)}"
        )

summary.append("")

summary.append("=" * 80)
summary.append("GLOBAL CONDITION COUNTS")
summary.append("=" * 80)
summary.append("")

for condition, count in condition_counts.items():

    summary.append(
        f"{condition}: {int(count)}"
    )

summary.append("")

summary.append("=" * 80)
summary.append("UNIQUE FILE MAPPINGS")
summary.append("=" * 80)
summary.append("")

for mapping, count in unique_mappings.items():

    summary.append(
        f"{count} files:"
    )

    summary.append(
        f"  {mapping}"
    )

    summary.append("")

summary.append("=" * 80)
summary.append("FINAL ASSESSMENT")
summary.append("=" * 80)
summary.append("")

if len(files) != EXPECTED_FILES:

    final_status = (
        "REVIEW - EXPECTED 82 FILES "
        "NOT FOUND"
    )

elif len(ambiguous) > 0:

    final_status = (
        "REVIEW - EVENT IDs ARE "
        "NOT GLOBALLY UNIQUE"
    )

else:

    final_status = (
        "PASS - GLOBAL EVENT MAPPING "
        "IS CONSISTENT"
    )

summary.append(final_status)

summary.append("")

summary.append("=" * 80)
summary.append("IMPORTANT")
summary.append("=" * 80)
summary.append("")

summary.append(
    "NO DATA WAS MODIFIED."
)

summary.append(
    "NO EPOCH FILE WAS MODIFIED."
)

summary.append(
    "READ-ONLY INVESTIGATION."
)

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("EVENT MAPPING INVESTIGATION COMPLETE")
print("=" * 80)

print()
print(
    f"Files: {len(files)} / {EXPECTED_FILES}"
)

print(
    f"Unique event IDs: "
    f"{len(global_event_id)}"
)

print(
    f"Ambiguous IDs: "
    f"{len(ambiguous)}"
)

print()
print("CSV:")
print(CSV_OUT)

print()
print("SUMMARY:")
print(SUMMARY_OUT)

print()
print("NO DATA WAS MODIFIED.")