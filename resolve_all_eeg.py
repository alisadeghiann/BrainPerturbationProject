from pathlib import Path
import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "qc" / "resolved_eeg"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def decode_uint16_string(x):
    arr = np.asarray(x).squeeze()

    if arr.dtype.kind in ("u", "i"):
        return "".join(chr(int(v)) for v in arr if int(v) != 0)

    return str(arr)


def read_scalar(h5, name):
    if name not in h5:
        return None

    x = np.asarray(h5[name][()]).squeeze()

    if x.size == 0:
        return None

    return float(x.flat[0])


def inspect_file(set_path):

    result = {
        "subject": set_path.parts[-4] if len(set_path.parts) >= 4 else "",
        "file": set_path.name,
        "set_path": str(set_path),
        "status": "ERROR",
        "fdt_path": "",
        "channels": None,
        "samples": None,
        "trials": None,
        "sampling_rate": None,
        "file_size_mb": None,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "error": ""
    }

    try:

        # --------------------------------------------------
        # READ SET
        # --------------------------------------------------

        with h5py.File(set_path, "r") as f:

            if "datfile" not in f:
                raise RuntimeError("datfile not found")

            datfile = decode_uint16_string(f["datfile"][()])

            nbchan = read_scalar(f, "nbchan")
            pnts = read_scalar(f, "pnts")
            trials = read_scalar(f, "trials")
            srate = read_scalar(f, "srate")

        if nbchan is None:
            raise RuntimeError("nbchan not found")

        if pnts is None:
            raise RuntimeError("pnts not found")

        if trials is None:
            trials = 1

        if srate is None:
            raise RuntimeError("srate not found")

        # --------------------------------------------------
        # FIND FDT
        # --------------------------------------------------

        fdt_path = set_path.parent / datfile

        if not fdt_path.exists():

            candidates = list(set_path.parent.glob("*.fdt"))

            matches = [
                p for p in candidates
                if p.name.lower() == Path(datfile).name.lower()
            ]

            if matches:
                fdt_path = matches[0]

        if not fdt_path.exists():
            raise FileNotFoundError(
                f"FDT not found: {datfile}"
            )

        # --------------------------------------------------
        # READ FDT
        # --------------------------------------------------

        raw = np.fromfile(fdt_path, dtype=np.float32)

        expected = int(nbchan * pnts * trials)

        dtype_used = "float32"

        if raw.size != expected:

            raw64 = np.fromfile(fdt_path, dtype=np.float64)

            if raw64.size == expected:
                raw = raw64
                dtype_used = "float64"

            else:
                raise RuntimeError(
                    f"FDT size mismatch: "
                    f"expected={expected}, "
                    f"float32={raw.size}, "
                    f"float64={raw64.size}"
                )

        # --------------------------------------------------
        # RESHAPE
        # --------------------------------------------------

        if int(trials) == 1:

            data = raw.reshape(
                int(nbchan),
                int(pnts)
            )

        else:

            data = raw.reshape(
                int(nbchan),
                int(pnts),
                int(trials)
            )

        # --------------------------------------------------
        # STATISTICS
        # --------------------------------------------------

        result["status"] = "OK"
        result["fdt_path"] = str(fdt_path)
        result["channels"] = int(nbchan)
        result["samples"] = int(pnts)
        result["trials"] = int(trials)
        result["sampling_rate"] = float(srate)
        result["file_size_mb"] = round(
            fdt_path.stat().st_size / (1024 ** 2), 2
        )

        result["min"] = float(np.min(data))
        result["max"] = float(np.max(data))
        result["mean"] = float(np.mean(data))
        result["std"] = float(np.std(data))

        return result

    except Exception as e:

        result["error"] = str(e)

        return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    set_files = sorted(DATA_DIR.rglob("*_eeg.set"))

    print("=" * 80)
    print("FULL EEG SET/FDT RESOLUTION")
    print("=" * 80)

    print(f"SET files found: {len(set_files)}")
    print()

    results = []

    for i, set_path in enumerate(set_files, 1):

        print("=" * 80)
        print(f"PROCESSING {i}/{len(set_files)}")
        print("=" * 80)
        print(set_path.name)

        result = inspect_file(set_path)

        results.append(result)

        if result["status"] == "OK":

            print("STATUS: OK")
            print("FDT:", Path(result["fdt_path"]).name)
            print("Channels:", result["channels"])
            print("Samples:", result["samples"])
            print("Sampling rate:", result["sampling_rate"])
            print("Min:", result["min"])
            print("Max:", result["max"])
            print("STD:", result["std"])

        else:

            print("STATUS: ERROR")
            print(result["error"])

    # ------------------------------------------------------
    # SAVE REPORT
    # ------------------------------------------------------

    df = pd.DataFrame(results)

    report = OUT_DIR / "resolved_eeg_report.csv"

    df.to_csv(
        report,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 80)
    print("FINAL RESOLUTION SUMMARY")
    print("=" * 80)

    print()
    print("STATUS:")
    print(df["status"].value_counts())

    print()
    print("CHANNEL COUNTS:")
    print(df["channels"].value_counts(dropna=False).sort_index())

    print()
    print("SAMPLING RATES:")
    print(
        df["sampling_rate"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print()
    print("SUBJECTS:")
    print(df["subject"].nunique())

    print()
    print("ERROR FILES:")

    errors = df[df["status"] != "OK"]

    if len(errors) == 0:
        print("None")
    else:
        print(errors[
            ["subject", "file", "error"]
        ].to_string(index=False))

    print()
    print("REPORT:")
    print(report)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)