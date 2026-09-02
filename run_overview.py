import pandas as pd
from pathlib import Path


# ============================================================
# PATH
# ============================================================

eeg_dir = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-002\ses-01\eeg"
)


# ============================================================
# CHECK ALL 4 RUNS
# ============================================================

print("=" * 80)
print("CHECKING ALL WORKING MEMORY RUNS")
print("=" * 80)


for run_number in range(1, 5):

    events_file = eeg_dir / (
        f"sub-002_ses-01_task-WorkingMemory_run-{run_number}_events.tsv"
    )

    print("\n" + "-" * 80)
    print(f"RUN {run_number}")
    print("-" * 80)

    if not events_file.exists():

        print("⚠ Events file not found")
        continue

    events = pd.read_csv(
        events_file,
        sep="\t"
    )

    print("Number of events:", len(events))

    print(
        "Number of trials:",
        events["trial"].nunique()
    )

    print("\nMemory conditions:")

    print(
        events[["trial", "memory_cond"]]
        .drop_duplicates()
        ["memory_cond"]
        .value_counts()
        .sort_index()
    )

    print("\nTask roles:")

    print(
        events["task_role"]
        .value_counts()
        .to_string()
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("ALL RUNS CHECK COMPLETED")
print("=" * 80)