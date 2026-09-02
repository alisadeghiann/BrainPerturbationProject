import h5py
import os

set_file = r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-002\ses-01\eeg\sub-002_ses-01_task-WorkingMemory_run-1_eeg.set"

print("=" * 70)
print("CHECKING EEGLAB HDF5 FILE")
print("=" * 70)

print("\nFile:")
print(set_file)

print("\nExists:", os.path.exists(set_file))

with h5py.File(set_file, "r") as f:

    print("\nTOP-LEVEL KEYS:")
    for key in f.keys():
        print(" ", key)

    print("\nIMPORTANT DATASET INFORMATION:")

    for key in ["data", "nbchan", "pnts", "trials", "srate", "xmin", "xmax"]:

        if key in f:

            obj = f[key]

            print(f"\n{key}")
            print("  type:", type(obj))

            if hasattr(obj, "shape"):
                print("  shape:", obj.shape)

            if hasattr(obj, "dtype"):
                print("  dtype:", obj.dtype)

            try:
                if key != "data":
                    print("  value:", obj[()])
            except Exception:
                pass

print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)