import h5py
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

eeg_dir = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-002\ses-01\eeg"
)

set_file = eeg_dir / "sub-002_ses-01_task-WorkingMemory_run-1_eeg.set"
fdt_file = eeg_dir / "sub-002_ses-01_task-WorkingMemory_run-1_eeg.fdt"


# ============================================================
# 1. CHECK FILES
# ============================================================

print("=" * 70)
print("CHECKING FILES")
print("=" * 70)

print("SET exists:", set_file.exists())
print("FDT exists:", fdt_file.exists())

if not set_file.exists():
    raise FileNotFoundError("SET file not found!")

if not fdt_file.exists():
    raise FileNotFoundError("FDT file not found!")


# ============================================================
# 2. READ BASIC INFORMATION FROM HDF5 SET
# ============================================================

print("\n" + "=" * 70)
print("READING HDF5 SET FILE")
print("=" * 70)

with h5py.File(set_file, "r") as f:

    nbchan = int(f["nbchan"][()][0][0])
    pnts = int(f["pnts"][()][0][0])
    trials = int(f["trials"][()][0][0])
    srate = float(f["srate"][()][0][0])
    xmin = float(f["xmin"][()][0][0])
    xmax = float(f["xmax"][()][0][0])

    print("Channels:", nbchan)
    print("Samples:", pnts)
    print("Trials:", trials)
    print("Sampling rate:", srate, "Hz")
    print("Xmin:", xmin)
    print("Xmax:", xmax)

    print("\nData dataset:")
    print("Shape:", f["data"].shape)
    print("Dtype:", f["data"].dtype)


# ============================================================
# 3. CHECK FDT FILE SIZE
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FDT FILE")
print("=" * 70)

fdt_size = fdt_file.stat().st_size

print("FDT size:", fdt_size, "bytes")
print("FDT size:", round(fdt_size / (1024 * 1024), 2), "MB")


# ============================================================
# 4. CALCULATE EXPECTED SIZE
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING EXPECTED DATA SIZE")
print("=" * 70)

expected_float32 = nbchan * pnts * 4
expected_float64 = nbchan * pnts * 8

print("Expected if float32:",
      expected_float32,
      "bytes =",
      round(expected_float32 / (1024 * 1024), 2),
      "MB")

print("Expected if float64:",
      expected_float64,
      "bytes =",
      round(expected_float64 / (1024 * 1024), 2),
      "MB")


# ============================================================
# 5. READ FDT AS FLOAT32
# ============================================================

print("\n" + "=" * 70)
print("READING FDT DATA")
print("=" * 70)

data = np.fromfile(fdt_file, dtype=np.float32)

print("Raw number of float32 values:", data.size)

expected_values = nbchan * pnts * trials

print("Expected number of values:", expected_values)


# ============================================================
# 6. VERIFY SIZE
# ============================================================

print("\n" + "=" * 70)
print("SIZE VERIFICATION")
print("=" * 70)

if data.size == expected_values:
    print("✓ FDT size matches EEG dimensions exactly!")

    data = data.reshape(
        (nbchan, pnts * trials),
        order="F"
    )

    print("Reshaped data:", data.shape)

else:
    print("⚠ FDT size does NOT match expected dimensions.")

    print("Difference:",
          data.size - expected_values)


# ============================================================
# 7. BASIC DATA CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)

print("Data dtype:", data.dtype)
print("Minimum:", np.min(data))
print("Maximum:", np.max(data))
print("Mean:", np.mean(data))
print("Standard deviation:", np.std(data))

nan_count = np.isnan(data).sum()

print("NaN values:", nan_count)


# ============================================================
# 8. FIRST FEW VALUES
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 VALUES FROM FIRST CHANNEL")
print("=" * 70)

print(data[0, :10])


# ============================================================
# 9. FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

if data.shape == (nbchan, pnts * trials):
    print("✓ EEG DATA SUCCESSFULLY EXTRACTED")
    print("✓ Channels:", data.shape[0])
    print("✓ Samples:", data.shape[1])
    print("✓ Sampling rate:", srate, "Hz")
else:
    print("⚠ Something needs further investigation.")

print("\nTEST COMPLETED")
print("=" * 70)