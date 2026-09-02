from pathlib import Path
import numpy as np
import pandas as pd
import mne

BASE_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

INPUT_DIR = (
    BASE_DIR
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v4"
    / "ELIGIBLE"
)

OUT_DIR = INPUT_DIR / "event_condition_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUT_DIR / "event_condition_audit_82runs.csv"
SUMMARY_OUT = OUT_DIR / "event_condition_audit_82runs_summary.txt"

files = sorted(INPUT_DIR.glob("*_standardized_epo.fif"))

print("=" * 80)
print("FINAL EVENT / CONDITION AUDIT")
print("=" * 80)
print(f"Files found: {len(files)}")

records = []

for i, filepath in enumerate(files, 1):

    print()
    print("=" * 80)
    print(f"[{i}/{len(files)}] {filepath.name}")
    print("=" * 80)

    try:

        epochs = mne.read_epochs(
            filepath,
            preload=False,
            verbose=False
        )

        event_id = epochs.event_id
        events = epochs.events

        print("Event ID mapping:")
        print(event_id)

        print(f"Epochs: {len(epochs)}")

        inverse = {
            int(v): k
            for k, v in event_id.items()
        }

        unique_ids, counts = np.unique(
            events[:, 2],
            return_counts=True
        )

        for eid, count in zip(
            unique_ids,
            counts
        ):

            condition = inverse.get(
                int(eid),
                "UNKNOWN"
            )

            records.append({
                "file": filepath.name,
                "event_id": int(eid),
                "condition": condition,
                "count": int(count),
                "total_epochs": len(epochs)
            })

            print(
                f"  ID {eid:>3} | "
                f"{condition:<20} | "
                f"{count}"
            )

    except Exception as e:

        print("ERROR:", e)

        records.append({
            "file": filepath.name,
            "event_id": np.nan,
            "condition": "ERROR",
            "count": np.nan,
            "total_epochs": np.nan
        })


df = pd.DataFrame(records)

df.to_csv(
    CSV_OUT,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# GLOBAL MAPPING
# ============================================================

mapping = (
    df[
        df["condition"] != "ERROR"
    ]
    [["event_id", "condition"]]
    .drop_duplicates()
    .sort_values("event_id")
)

# ============================================================
# CONDITION TOTALS
# ============================================================

condition_totals = (
    df[
        df["condition"] != "ERROR"
    ]
    .groupby(
        "condition",
        as_index=False
    )["count"]
    .sum()
    .sort_values(
        "count",
        ascending=False
    )
)

# ============================================================
# MAPPING CONSISTENCY
# ============================================================

mapping_per_file = (
    df[
        df["condition"] != "ERROR"
    ]
    .groupby("file")
    .apply(
        lambda x: set(
            zip(
                x["event_id"],
                x["condition"]
            )
        ),
        include_groups=False
    )
)

unique_mappings = set(
    tuple(sorted(x))
    for x in mapping_per_file
)

mapping_consistent = (
    len(unique_mappings) == 1
)

# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.append("=" * 80)
summary.append("FINAL EVENT / CONDITION AUDIT SUMMARY")
summary.append("=" * 80)
summary.append("")

summary.append(
    f"Files found: {len(files)}"
)

summary.append(
    f"Files successfully audited: "
    f"{df['file'].nunique()}"
)

summary.append("")

summary.append("=" * 80)
summary.append("GLOBAL EVENT MAPPING")
summary.append("=" * 80)
summary.append("")

for _, row in mapping.iterrows():

    summary.append(
        f"ID {int(row['event_id'])}: "
        f"{row['condition']}"
    )

summary.append("")

summary.append("=" * 80)
summary.append("GLOBAL CONDITION COUNTS")
summary.append("=" * 80)
summary.append("")

for _, row in condition_totals.iterrows():

    summary.append(
        f"{row['condition']}: "
        f"{int(row['count'])}"
    )

summary.append("")

summary.append("=" * 80)
summary.append("MAPPING CONSISTENCY")
summary.append("=" * 80)
summary.append("")

if mapping_consistent:

    summary.append(
        "PASS - Event/condition mapping "
        "is identical across all files."
    )

else:

    summary.append(
        "FAIL - Event/condition mapping "
        "differs between files."
    )

summary.append("")

summary.append("=" * 80)
summary.append("FINAL DECISION")
summary.append("=" * 80)
summary.append("")

if (
    len(files) == 82
    and mapping_consistent
    and "ERROR" not in df["condition"].values
):

    final_status = (
        "PASS - EVENT MAPPING READY "
        "FOR PERTURBATION ANALYSIS"
    )

else:

    final_status = (
        "REVIEW - EVENT MAPPING "
        "REQUIRES INVESTIGATION"
    )

summary.append(final_status)

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )

print()
print("=" * 80)
print("FINAL EVENT / CONDITION AUDIT COMPLETE")
print("=" * 80)

print()
print("GLOBAL MAPPING")
print(mapping.to_string(index=False))

print()
print("GLOBAL CONDITION COUNTS")
print(condition_totals.to_string(index=False))

print()
print(
    "Mapping consistent:",
    mapping_consistent
)

print()
print("FINAL STATUS:")
print(final_status)

print()
print("Saved:")
print(CSV_OUT)
print(SUMMARY_OUT)

print()
print("NO DATA WAS MODIFIED.")