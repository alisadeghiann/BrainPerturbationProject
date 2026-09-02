from pathlib import Path
import pandas as pd

# ============================================================
# PROJECT PATH
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

# ============================================================
# INPUT
# ============================================================

EVENTS_FILE = (
    BASE
    / "data"
    / "sub-024"
    / "ses-01"
    / "eeg"
    / "sub-024_ses-01_task-WorkingMemory_run-1_events.tsv"
)

# ============================================================
# CHECK FILE
# ============================================================

print("=" * 90)
print("SUB-024 RUN-01 EVENTS INSPECTION")
print("=" * 90)

print(f"\nFile:")
print(EVENTS_FILE)

print(f"\nExists: {EVENTS_FILE.exists()}")

if not EVENTS_FILE.exists():
    raise FileNotFoundError(
        f"\nERROR: File not found:\n{EVENTS_FILE}"
    )

# ============================================================
# READ EVENTS
# ============================================================

df = pd.read_csv(EVENTS_FILE, sep="\t")

print("\n" + "=" * 90)
print("DATASET SHAPE")
print("=" * 90)

print(f"Rows:    {len(df)}")
print(f"Columns: {len(df.columns)}")

# ============================================================
# COLUMNS
# ============================================================

print("\n" + "=" * 90)
print("COLUMNS")
print("=" * 90)

for i, col in enumerate(df.columns, start=1):
    print(f"{i:02d}. {col}")

# ============================================================
# FIRST 20 ROWS
# ============================================================

print("\n" + "=" * 90)
print("FIRST 20 ROWS")
print("=" * 90)

print(df.head(20).to_string(index=False))

# ============================================================
# UNIQUE VALUES FOR CATEGORICAL COLUMNS
# ============================================================

print("\n" + "=" * 90)
print("UNIQUE / VALUE COUNTS FOR CATEGORICAL COLUMNS")
print("=" * 90)

for col in df.columns:

    if df[col].dtype == "object":

        print("\n" + "-" * 80)
        print(f"COLUMN: {col}")
        print("-" * 80)

        print(
            df[col]
            .value_counts(dropna=False)
            .to_string()
        )

# ============================================================
# SEARCH FOR REMEMBER / IGNORE
# ============================================================

print("\n" + "=" * 90)
print("SEARCH FOR REMEMBER / IGNORE")
print("=" * 90)

terms = [
    "remember",
    "ignore",
    "recall",
    "forget",
    "memory",
]

found = False

for col in df.columns:

    values = df[col].astype(str).str.lower()

    for term in terms:

        mask = values.str.contains(term, na=False)

        if mask.any():

            found = True

            print(
                f"\nFOUND '{term}' in column: {col}"
            )

            print(
                df.loc[mask, [col]]
                .value_counts(dropna=False)
                .to_string()
            )

if not found:
    print("\nNo Remember/Ignore-related text found.")

# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 90)
print("INSPECTION FINISHED")
print("=" * 90)