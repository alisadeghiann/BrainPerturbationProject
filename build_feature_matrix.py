from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = BASE / "features" / "basic" / "eeg_band_power_features.csv"
OUTPUT = BASE / "features" / "basic"

df = pd.read_csv(INPUT)

print("=" * 80)
print("FEATURE MATRIX CONSTRUCTION")
print("=" * 80)

# One row = one epoch
# Columns = channel + frequency-band features

matrix = df.pivot_table(
    index=["file", "subject", "run", "epoch"],
    columns=["band", "channel"],
    values="power",
    aggfunc="mean"
)

matrix.columns = [
    f"{band}_{channel}"
    for band, channel in matrix.columns
]

matrix = matrix.reset_index()

output = OUTPUT / "eeg_feature_matrix.csv"

matrix.to_csv(output, index=False)

print(f"Epochs:       {len(matrix):,}")
print(f"Feature cols: {len(matrix.columns) - 4:,}")
print(f"Subjects:     {matrix['subject'].nunique()}")
print(f"Runs:         {matrix['file'].nunique()}")

print()
print("Missing values:")
print(matrix.isna().sum().sum())

print()
print("Saved:")
print(output)

print("=" * 80)
