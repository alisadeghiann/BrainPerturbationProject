import pandas as pd
from pathlib import Path


# ============================================================
# PATH
# ============================================================

eeg_dir = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-002\ses-01\eeg"
)

events_file = eeg_dir / (
    "sub-002_ses-01_task-WorkingMemory_run-1_events.tsv"
)


# ============================================================
# LOAD EVENTS
# ============================================================

events = pd.read_csv(
    events_file,
    sep="\t"
)


# ============================================================
# TRIAL SUMMARY
# ============================================================

print("=" * 80)
print("TRIAL SUMMARY")
print("=" * 80)

for trial in sorted(events["trial"].dropna().unique()):

    t = events[events["trial"] == trial]

    memory_cond = t["memory_cond"].iloc[0]

    print("\n" + "-" * 80)
    print(f"TRIAL {int(trial)}")
    print(f"Memory condition: {memory_cond}")

    print("\nEvents:")

    for _, row in t.iterrows():

        print(
            f"{row['onset']:8.3f}s | "
            f"{row['event_type']:15s} | "
            f"{str(row['task_role']):20s} | "
            f"value={row['value']}"
        )


# ============================================================
# CONDITION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MEMORY CONDITION SUMMARY")
print("=" * 80)

summary = (
    events
    .groupby(["trial", "memory_cond"])
    .size()
    .reset_index(name="event_count")
)

print(summary.to_string(index=False))


# ============================================================
# NUMBER OF TRIALS PER CONDITION
# ============================================================

print("\n" + "=" * 80)
print("TRIAL COUNT BY MEMORY CONDITION")
print("=" * 80)

trial_conditions = (
    events[["trial", "memory_cond"]]
    .drop_duplicates()
    .sort_values("trial")
)

print(
    trial_conditions
    .groupby("memory_cond")
    .size()
)


# ============================================================
# REMEMBER / IGNORE COUNTS
# ============================================================

print("\n" + "=" * 80)
print("REMEMBER / IGNORE COUNTS")
print("=" * 80)

print(
    events["task_role"]
    .value_counts()
)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)