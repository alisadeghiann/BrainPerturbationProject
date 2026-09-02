from pathlib import Path
import h5py
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# MATLAB HDF5 STRING DECODER
# ============================================================

def decode_uint16_string(x):
    """
    Decode MATLAB v7.3 char array stored as uint16.
    """
    arr = np.asarray(x).squeeze()

    if arr.dtype.kind in ("u", "i"):
        return "".join(chr(int(v)) for v in arr if int(v) != 0)

    return str(arr)


def read_scalar(h5, name):
    """
    Read a MATLAB scalar dataset.
    """
    if name not in h5:
        return None

    x = np.asarray(h5[name][()]).squeeze()

    if x.size == 0:
        return None

    return float(x.flat[0])


# ============================================================
# READ SET HEADER
# ============================================================

def inspect_set(set_path):

    print("=" * 80)
    print("INSPECTING")
    print(set_path.name)
    print("=" * 80)

    with h5py.File(set_path, "r") as f:

        # ----------------------------------------------------
        # DATFILE
        # ----------------------------------------------------

        if "datfile" not in f:
            raise RuntimeError("datfile not found in SET file")

        datfile_raw = f["datfile"][()]
        datfile = decode_uint16_string(datfile_raw)

        print("DATFILE:")
        print(datfile)

        # ----------------------------------------------------
        # BASIC EEGLAB PARAMETERS
        # ----------------------------------------------------

        nbchan = read_scalar(f, "nbchan")
        pnts = read_scalar(f, "pnts")
        trials = read_scalar(f, "trials")
        srate = read_scalar(f, "srate")
        xmin = read_scalar(f, "xmin")
        xmax = read_scalar(f, "xmax")

        print()
        print("EEGLAB HEADER")
        print("-" * 60)
        print("nbchan :", nbchan)
        print("pnts   :", pnts)
        print("trials :", trials)
        print("srate  :", srate)
        print("xmin   :", xmin)
        print("xmax   :", xmax)

    # --------------------------------------------------------
    # FIND FDT
    # --------------------------------------------------------

    fdt_path = set_path.parent / datfile

    print()
    print("EXPECTED FDT:")
    print(fdt_path)

    if not fdt_path.exists():

        # Case-insensitive fallback
        candidates = list(set_path.parent.glob("*.fdt"))

        matches = [
            p for p in candidates
            if p.name.lower() == Path(datfile).name.lower()
        ]

        if matches:
            fdt_path = matches[0]

    if not fdt_path.exists():

        raise FileNotFoundError(
            f"FDT file not found:\n{fdt_path}"
        )

    print("FDT EXISTS: YES")

    return {
        "set_path": set_path,
        "fdt_path": fdt_path,
        "nbchan": int(nbchan),
        "pnts": int(pnts),
        "trials": int(trials),
        "srate": float(srate),
        "xmin": xmin,
        "xmax": xmax,
    }


# ============================================================
# LOAD FDT
# ============================================================

def load_fdt(info):

    fdt_path = info["fdt_path"]
    nbchan = info["nbchan"]
    pnts = info["pnts"]
    trials = info["trials"]

    print()
    print("READING FDT")
    print("-" * 60)
    print(fdt_path)

    # EEGLAB .fdt files are normally float32
    raw = np.fromfile(fdt_path, dtype=np.float32)

    print("Raw float32 values:", raw.size)

    expected = nbchan * pnts * trials

    print("Expected values   :", expected)

    if raw.size != expected:

        print()
        print("WARNING:")
        print("Raw size does not match nbchan*pnts*trials")
        print()

        # Try float64 as diagnostic
        raw64 = np.fromfile(fdt_path, dtype=np.float64)

        print("Float64 values    :", raw64.size)

        if raw64.size == expected:
            print("FDT appears to be FLOAT64")
            raw = raw64

        else:
            raise RuntimeError(
                "FDT size does not match EEGLAB header."
            )

    # --------------------------------------------------------
    # RESHAPE
    # --------------------------------------------------------

    if trials == 1:

        data = raw.reshape(nbchan, pnts)

    else:

        data = raw.reshape(nbchan, pnts, trials)

    print()
    print("REAL EEG DATA")
    print("-" * 60)
    print("Shape:", data.shape)
    print("dtype:", data.dtype)
    print("Min  :", np.min(data))
    print("Max  :", np.max(data))
    print("Mean :", np.mean(data))
    print("STD  :", np.std(data))

    return data


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    set_files = sorted(DATA_DIR.rglob("*_eeg.set"))

    print()
    print("=" * 80)
    print("EEGLAB SET/FDT RESOLVER")
    print("=" * 80)
    print("SET files found:", len(set_files))
    print()

    # Start with ONE file
    test_file = set_files[0]

    print("TEST FILE:")
    print(test_file)
    print()

    info = inspect_set(test_file)

    data = load_fdt(info)

    print()
    print("=" * 80)
    print("SUCCESS")
    print("=" * 80)

    print("SET :", info["set_path"].name)
    print("FDT :", info["fdt_path"].name)
    print("Channels :", info["nbchan"])
    print("Samples  :", info["pnts"])
    print("Trials   :", info["trials"])
    print("Sampling :", info["srate"])
    print("Data shape:", data.shape)

    print()
    print("First 5 channels:")
    print(data[:5, :10])