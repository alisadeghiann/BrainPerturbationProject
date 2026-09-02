import os
import glob
import h5py
import numpy as np

PROJECT_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# پیدا کردن اولین فایل EEG
files = glob.glob(
    os.path.join(DATA_DIR, "**", "*_eeg.set"),
    recursive=True
)

if not files:
    raise FileNotFoundError("No *_eeg.set files found.")

set_file = sorted(files)[0]

print("=" * 80)
print("HDF5 DATA RESOLVER")
print("=" * 80)
print("FILE:")
print(set_file)

with h5py.File(set_file, "r") as f:

    print("\nTOP LEVEL:")
    for key in f.keys():
        print(" ", key)

    print("\nDATA OBJECT")
    print("-" * 80)

    data = f["data"]

    print("Path:", data.name)
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)

    raw = data[()]

    print("\nRaw /data:")
    print(raw)

    # --------------------------------------------------------
    # Check references
    # --------------------------------------------------------

    if data.dtype == h5py.ref_dtype or data.dtype.kind == "O":

        print("\nDATA CONTAINS REFERENCES")
        print("-" * 80)

        refs = np.asarray(raw).flatten()

        for i, ref in enumerate(refs[:20]):

            print(f"\nReference {i}")

            if not ref:
                print("  NULL REFERENCE")
                continue

            try:
                obj = f[ref]

                print("  Target:", obj.name)
                print("  Type:", type(obj).__name__)

                if isinstance(obj, h5py.Dataset):

                    print("  Shape:", obj.shape)
                    print("  Dtype:", obj.dtype)

                    # اگر Dataset کوچک باشد، محتوایش را هم نشان بده
                    if obj.size < 100:
                        print("  Data:", obj[()])

            except Exception as e:
                print("  ERROR:", e)

    # --------------------------------------------------------
    # Search ALL datasets
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("ALL HDF5 DATASETS")
    print("=" * 80)

    datasets = []

    def visitor(name, obj):

        if isinstance(obj, h5py.Dataset):

            try:
                size = int(np.prod(obj.shape))
            except Exception:
                size = 0

            datasets.append(
                (size, name, obj.shape, str(obj.dtype))
            )

    f.visititems(visitor)

    datasets.sort(reverse=True)

    print(f"\nTotal datasets: {len(datasets)}")

    print("\nLARGEST DATASETS:")
    print("-" * 80)

    for size, name, shape, dtype in datasets[:50]:

        print(
            f"SIZE={size:>12} | "
            f"SHAPE={str(shape):<25} | "
            f"DTYPE={dtype:<15} | "
            f"{name}"
        )

    # --------------------------------------------------------
    # Check datfile
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("DATFILE")
    print("=" * 80)

    if "datfile" in f:

        obj = f["datfile"]

        print("Shape:", obj.shape)
        print("Dtype:", obj.dtype)

        try:
            print("Raw:", obj[()])
        except Exception as e:
            print("Could not read:", e)

    else:

        print("No datfile field found.")

print("\n")
print("=" * 80)
print("RESOLUTION COMPLETE")
print("=" * 80)