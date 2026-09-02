from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

FEATURES = BASE / "features" / "scientific_v1" / "scientific_features_v1.csv"
MAPPING = BASE / "features" / "scientific_v1" / "merged" / "deterministic_trial_epoch_map.csv"

f = pd.read_csv(FEATURES)
m = pd.read_csv(MAPPING)

print("=" * 90)
print("FEATURE ↔ MAPPING KEY DIAGNOSIS")
print("=" * 90)

for name, df in [("FEATURES", f), ("MAPPING", m)]:

    print()
    print(name)

    print("columns:", list(df.columns))

    print("\nfirst 15 rows:")
    print(
        df[
            [c for c in ["file", "subject", "run", "epoch", "trial", "event_name"]
             if c in df.columns]
        ].head(15).to_string(index=False)
    )

    print("\nsubject:", df["subject"].iloc[0])

    print(
        "runs:",
        df["run"].drop_duplicates().tolist()[:10]
    )

    print(
        "epoch min:",
        df["epoch"].min(),
        "max:",
        df["epoch"].max()
    )

# ------------------------------------------------------------
# SAME SUBJECT/RUN COMPARISON
# ------------------------------------------------------------

subject = f["subject"].iloc[0]
run = f["run"].iloc[0]

ff = f[
    (f["subject"] == subject) &
    (f["run"] == run)
].copy()

mm = m[
    (m["subject"] == subject) &
    (m["run"] == run)
].copy()

print()
print("=" * 90)
print(f"SAMPLE COMPARISON: {subject} {run}")
print("=" * 90)

print("\nFEATURE EPOCHS:")
print(
    ff["epoch"].head(30).tolist()
)

print("\nMAPPING EPOCHS:")
print(
    mm["epoch"].head(30).tolist()
)

print()
print("FEATURE epoch dtype:", f["epoch"].dtype)
print("MAPPING epoch dtype:", m["epoch"].dtype)

print()
print("FEATURE rows for sample:", len(ff))
print("MAPPING rows for sample:", len(mm))

# ------------------------------------------------------------
# CHECK POSSIBLE OFFSETS
# ------------------------------------------------------------

F = set(ff["epoch"].astype(int))
M = set(mm["epoch"].astype(int))

print()
print("=" * 90)
print("POSSIBLE EPOCH OFFSETS")
print("=" * 90)

for offset in range(-20, 21):

    overlap = len(
        F.intersection(
            {x + offset for x in M}
        )
    )

    if overlap > 0:

        print(
            f"offset {offset:+d} -> "
            f"{overlap} overlapping epochs"
        )

# ------------------------------------------------------------
# FILE NAMING
# ------------------------------------------------------------

print()
print("=" * 90)
print("FILE NAMES")
print("=" * 90)

print("\nFEATURE files:")
print(
    f["file"].drop_duplicates().head(10).to_string(index=False)
)

print("\nMAPPING files:")
print(
    m["file"].drop_duplicates().head(10).to_string(index=False)
)

print()
print("=" * 90)
print("DIAGNOSIS COMPLETE")
print("=" * 90)