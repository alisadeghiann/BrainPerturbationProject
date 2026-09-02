import h5py
import numpy as np

# =========================
# SET FILE
# =========================

set_path = r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-009\ses-01\eeg\sub-009_ses-01_task-WorkingMemory_run-1_eeg.set"

# =========================
# HELPER FUNCTION
# =========================

def decode_hdf5_string(f, ref):
    obj = f[ref]
    values = np.array(obj).flatten()

    return "".join(chr(int(x)) for x in values)


def decode_hdf5_numeric(f, ref):
    obj = f[ref]
    values = np.array(obj).flatten()

    return float(values[0])


# =========================
# OPEN FILE
# =========================

with h5py.File(set_path, "r") as f:

    chanlocs = f["chanlocs"]

    labels = chanlocs["labels"]
    X = chanlocs["X"]
    Y = chanlocs["Y"]
    Z = chanlocs["Z"]

    print("========== CHANNEL + COORDINATES ==========")

    print(
        f"{'INDEX':<6}"
        f"{'LABEL':<8}"
        f"{'X':>12}"
        f"{'Y':>12}"
        f"{'Z':>12}"
    )

    print("-" * 50)

    for i in range(len(labels)):

        label = decode_hdf5_string(
            f,
            labels[i, 0]
        )

        x = decode_hdf5_numeric(
            f,
            X[i, 0]
        )

        y = decode_hdf5_numeric(
            f,
            Y[i, 0]
        )

        z = decode_hdf5_numeric(
            f,
            Z[i, 0]
        )

        print(
            f"{i:<6}"
            f"{label:<8}"
            f"{x:>12.4f}"
            f"{y:>12.4f}"
            f"{z:>12.4f}"
        )

    # =========================
    # SPECIAL CHANNELS
    # =========================

    print("\n========== SPECIAL CHANNELS ==========")

    print("Last channels:")

    for i in range(69, 71):

        label = decode_hdf5_string(
            f,
            labels[i, 0]
        )

        x = decode_hdf5_numeric(
            f,
            X[i, 0]
        )

        y = decode_hdf5_numeric(
            f,
            Y[i, 0]
        )

        z = decode_hdf5_numeric(
            f,
            Z[i, 0]
        )

        print(
            i,
            label,
            "X=", x,
            "Y=", y,
            "Z=", z
        )