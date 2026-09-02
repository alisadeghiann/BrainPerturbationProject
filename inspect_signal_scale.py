import os
import glob
import numpy as np
import pandas as pd
import h5py

BASE = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

OUT_DIR = os.path.join(
    BASE,
    "qc",
    "signal_scale_inspection"
)

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(
    glob.glob(
        os.path.join(
            BASE,
            "data",
            "**",
            "*_eeg.set"
        ),
        recursive=True
    )
)

print("=" * 90)
print("EEG SIGNAL SCALE + REFERENCE INSPECTION")
print("=" * 90)

print("Files:", len(files))

records = []

for i, set_file in enumerate(files, 1):

    name = os.path.basename(set_file)

    print("\n" + "=" * 80)
    print(f"{i}/{len(files)}")
    print(name)

    try:

        with h5py.File(set_file, "r") as f:

            nbchan = int(np.array(f["nbchan"]).squeeze())
            pnts = int(np.array(f["pnts"]).squeeze())
            srate = float(np.array(f["srate"]).squeeze())

            ref = None

            if "ref" in f:

                try:
                    ref_raw = np.array(f["ref"]).squeeze()

                    if ref_raw.dtype.kind in ["U", "S", "O"]:
                        ref = str(ref_raw)

                    else:
                        ref = str(ref_raw)

                except Exception:
                    ref = "unreadable"

            # --------------------------------------------
            # Channel labels
            # --------------------------------------------

            labels = []

            try:

                chanlocs = f["chanlocs"]

                if "labels" in chanlocs:

                    refs = np.array(
                        chanlocs["labels"]
                    ).flatten()

                    for r in refs:

                        try:

                            obj = f[r]

                            arr = np.array(obj).flatten()

                            text = "".join(
                                chr(int(x))
                                for x in arr
                            )

                            labels.append(text)

                        except Exception:

                            labels.append("?")

            except Exception:

                labels = []

            # --------------------------------------------
            # FDT
            # --------------------------------------------

            fdt = set_file.replace(
                "_eeg.set",
                "_eeg.fdt"
            )

            raw = np.fromfile(
                fdt,
                dtype=np.float32
            )

            expected = nbchan * pnts

            if len(raw) != expected:

                print("SIZE ERROR")

                continue

            data = raw.reshape(
                nbchan,
                pnts
            )

            # --------------------------------------------
            # Global statistics
            # --------------------------------------------

            global_std = float(np.std(data))
            global_mean = float(np.mean(data))
            minimum = float(np.min(data))
            maximum = float(np.max(data))

            # --------------------------------------------
            # Channel statistics
            # --------------------------------------------

            ch_mean = np.mean(data, axis=1)
            ch_std = np.std(data, axis=1)
            ch_min = np.min(data, axis=1)
            ch_max = np.max(data, axis=1)

            median_std = float(
                np.median(ch_std)
            )

            max_std = float(
                np.max(ch_std)
            )

            max_std_idx = int(
                np.argmax(ch_std)
            )

            max_std_channel = (
                labels[max_std_idx]
                if max_std_idx < len(labels)
                else f"CH_{max_std_idx+1}"
            )

            # --------------------------------------------
            # Robust percentiles
            # --------------------------------------------

            p01 = float(
                np.percentile(data, 1)
            )

            p99 = float(
                np.percentile(data, 99)
            )

            # --------------------------------------------
            # Save
            # --------------------------------------------

            records.append({

                "subject": name.split("_")[0],

                "file": name,

                "sampling_rate": srate,

                "channels": nbchan,

                "samples": pnts,

                "reference": ref,

                "global_mean": global_mean,

                "global_std": global_std,

                "min": minimum,

                "max": maximum,

                "p01": p01,

                "p99": p99,

                "median_channel_std": median_std,

                "max_channel_std": max_std,

                "max_std_channel": max_std_channel

            })

            print("Sampling:", srate)
            print("Reference:", ref)
            print("Global STD:", global_std)
            print("Median channel STD:", median_std)
            print("Max channel STD:", max_std)
            print("Max STD channel:", max_std_channel)
            print("P01:", p01)
            print("P99:", p99)

    except Exception as e:

        print("ERROR:", e)

# =========================================================
# SUMMARY
# =========================================================

df = pd.DataFrame(records)

print("\n")
print("=" * 90)
print("SIGNAL SCALE SUMMARY")
print("=" * 90)

print(
    df[
        [
            "subject",
            "file",
            "sampling_rate",
            "reference",
            "global_std",
            "median_channel_std",
            "max_channel_std",
            "max_std_channel",
            "p01",
            "p99"
        ]
    ]
    .sort_values("global_std")
    .to_string(index=False)
)

# =========================================================
# SAVE
# =========================================================

output = os.path.join(
    OUT_DIR,
    "SIGNAL_SCALE_REFERENCE_REPORT.csv"
)

df.to_csv(
    output,
    index=False
)

print("\n")
print("=" * 90)
print("SAVED")
print("=" * 90)

print(output)

# =========================================================
# EXTREME FILES
# =========================================================

print("\n")
print("=" * 90)
print("LOWEST SCALE FILES")
print("=" * 90)

print(
    df.nsmallest(
        10,
        "global_std"
    )[
        [
            "subject",
            "file",
            "global_std",
            "median_channel_std",
            "max_channel_std"
        ]
    ].to_string(index=False)
)

print("\n")
print("=" * 90)
print("HIGHEST SCALE FILES")
print("=" * 90)

print(
    df.nlargest(
        10,
        "global_std"
    )[
        [
            "subject",
            "file",
            "global_std",
            "median_channel_std",
            "max_channel_std"
        ]
    ].to_string(index=False)
)

print("\n")
print("=" * 90)
print("DONE")
print("=" * 90)

print("No EEG data modified.")
print("No files deleted.")