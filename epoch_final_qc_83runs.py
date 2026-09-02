import os
import glob
import numpy as np
import pandas as pd
import mne

# ============================================================
# FINAL EPOCH QC - 83 RUNS
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

EPOCH_DIR = os.path.join(BASE_DIR, "epochs_v3")
QC_DIR = os.path.join(EPOCH_DIR, "logs")

os.makedirs(QC_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(QC_DIR, "epoch_final_qc_83runs.csv")
OUTPUT_SUMMARY = os.path.join(QC_DIR, "epoch_final_qc_83runs_summary.txt")

# ============================================================
# SETTINGS
# ============================================================

EXPECTED_FILES = 83
EXPECTED_CHANNELS = 71
EXPECTED_SFREQ = 500.0
EXPECTED_SAMPLES = 501

# Amplitude thresholds in uV
REVIEW_ABS_AMPLITUDE = 300.0
BAD_ABS_AMPLITUDE = 500.0

# Percentage thresholds
REVIEW_HIGH_AMP_PERCENT = 5.0
BAD_HIGH_AMP_PERCENT = 20.0

# NaN / Inf
MAX_INVALID_PERCENT = 0.0

# ============================================================
# FIND FILES
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            EPOCH_DIR,
            "*_epo.fif"
        )
    )
)

print("=" * 80)
print("FINAL EPOCH QC - 83 RUNS")
print("=" * 80)

print()
print(f"Epoch directory: {EPOCH_DIR}")
print(f"Epoch files found: {len(files)}")
print(f"Expected files:    {EXPECTED_FILES}")

if len(files) == 0:
    raise FileNotFoundError(
        "No epoch FIF files were found."
    )

# ============================================================
# STORAGE
# ============================================================

results = []

total_epochs = 0
total_channels = 0

pass_count = 0
review_count = 0
bad_count = 0

# ============================================================
# PROCESS FILES
# ============================================================

for idx, filepath in enumerate(files, start=1):

    filename = os.path.basename(filepath)

    print()
    print("=" * 80)
    print(f"[{idx}/{len(files)}] {filename}")
    print("=" * 80)

    status = "PASS"
    reasons = []

    try:

        # ----------------------------------------------------
        # READ EPOCHS
        # ----------------------------------------------------

        print("Reading epochs...")

        epochs = mne.read_epochs(
            filepath,
            preload=True,
            verbose=False
        )

        # ----------------------------------------------------
        # BASIC INFORMATION
        # ----------------------------------------------------

        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        n_times = len(epochs.times)
        sfreq = float(epochs.info["sfreq"])

        duration = float(
            epochs.times[-1] - epochs.times[0]
        ) if n_times > 1 else 0.0

        total_epochs += n_epochs
        total_channels += n_channels

        print(f"Epochs:       {n_epochs}")
        print(f"Channels:     {n_channels}")
        print(f"Samples:      {n_times}")
        print(f"Sampling:     {sfreq} Hz")
        print(f"Epoch length: {duration:.6f} sec")

        # ----------------------------------------------------
        # ZERO EPOCH CHECK
        # ----------------------------------------------------

        if n_epochs == 0:
            status = "BAD"
            reasons.append("ZERO_EPOCH")

            results.append({
                "file": filename,
                "n_epochs": n_epochs,
                "n_channels": n_channels,
                "n_times": n_times,
                "sfreq": sfreq,
                "epoch_duration_sec": duration,
                "nan_percent": np.nan,
                "inf_percent": np.nan,
                "global_min": np.nan,
                "global_max": np.nan,
                "global_std": np.nan,
                "max_abs": np.nan,
                "high_amplitude_percent": np.nan,
                "bad_amplitude_percent": np.nan,
                "suspicious_epochs": np.nan,
                "suspicious_channels": np.nan,
                "event_count": 0,
                "event_types": "",
                "status": status,
                "reasons": ";".join(reasons)
            })

            print("STATUS: BAD - ZERO EPOCHS")
            continue

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        data = epochs.get_data()

        print(f"Data shape: {data.shape}")
        print(f"Data dtype: {data.dtype}")

        # ----------------------------------------------------
        # NAN / INF
        # ----------------------------------------------------

        total_values = data.size

        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())

        nan_percent = (
            100.0 * nan_count / total_values
        )

        inf_percent = (
            100.0 * inf_count / total_values
        )

        print()
        print("INVALID VALUES")
        print("-" * 80)
        print(f"NaN count:     {nan_count}")
        print(f"NaN percent:   {nan_percent:.6f}%")
        print(f"Inf count:     {inf_count}")
        print(f"Inf percent:   {inf_percent:.6f}%")

        if nan_percent > MAX_INVALID_PERCENT:
            status = "BAD"
            reasons.append("NAN")

        if inf_percent > MAX_INVALID_PERCENT:
            status = "BAD"
            reasons.append("INF")

        # ----------------------------------------------------
        # GLOBAL AMPLITUDE
        # ----------------------------------------------------

        global_min = float(np.nanmin(data))
        global_max = float(np.nanmax(data))
        global_std = float(np.nanstd(data))
        max_abs = float(np.nanmax(np.abs(data)))

        high_amp_mask = np.abs(data) > REVIEW_ABS_AMPLITUDE
        bad_amp_mask = np.abs(data) > BAD_ABS_AMPLITUDE

        high_amplitude_percent = (
            100.0 * high_amp_mask.sum() / total_values
        )

        bad_amplitude_percent = (
            100.0 * bad_amp_mask.sum() / total_values
        )

        print()
        print("AMPLITUDE")
        print("-" * 80)
        print(f"Min:                  {global_min:.3f}")
        print(f"Max:                  {global_max:.3f}")
        print(f"STD:                  {global_std:.3f}")
        print(f"Max absolute:         {max_abs:.3f}")
        print(
            f"|x| > {REVIEW_ABS_AMPLITUDE:.0f}:          "
            f"{high_amplitude_percent:.4f}%"
        )
        print(
            f"|x| > {BAD_ABS_AMPLITUDE:.0f}:          "
            f"{bad_amplitude_percent:.4f}%"
        )

        # ----------------------------------------------------
        # EPOCH-LEVEL AMPLITUDE
        # ----------------------------------------------------

        epoch_max_abs = np.nanmax(
            np.abs(data),
            axis=(1, 2)
        )

        suspicious_epoch_mask = (
            epoch_max_abs > REVIEW_ABS_AMPLITUDE
        )

        bad_epoch_mask = (
            epoch_max_abs > BAD_ABS_AMPLITUDE
        )

        suspicious_epochs = int(
            suspicious_epoch_mask.sum()
        )

        bad_epochs = int(
            bad_epoch_mask.sum()
        )

        print()
        print("EPOCH-LEVEL AMPLITUDE")
        print("-" * 80)
        print(
            f"Epochs > {REVIEW_ABS_AMPLITUDE:.0f} uV: "
            f"{suspicious_epochs}"
        )
        print(
            f"Epochs > {BAD_ABS_AMPLITUDE:.0f} uV: "
            f"{bad_epochs}"
        )

        # ----------------------------------------------------
        # CHANNEL-LEVEL AMPLITUDE
        # ----------------------------------------------------

        channel_max_abs = np.nanmax(
            np.abs(data),
            axis=(0, 2)
        )

        channel_high_percent = (
            100.0 *
            np.mean(
                np.abs(data) > REVIEW_ABS_AMPLITUDE,
                axis=(0, 2)
            )
        )

        suspicious_channel_mask = (
            (channel_max_abs > REVIEW_ABS_AMPLITUDE)
            |
            (channel_high_percent > REVIEW_HIGH_AMP_PERCENT)
        )

        bad_channel_mask = (
            (channel_max_abs > BAD_ABS_AMPLITUDE)
            |
            (channel_high_percent > BAD_HIGH_AMP_PERCENT)
        )

        suspicious_channels = int(
            suspicious_channel_mask.sum()
        )

        bad_channels = int(
            bad_channel_mask.sum()
        )

        print()
        print("CHANNEL-LEVEL QC")
        print("-" * 80)
        print(
            f"Suspicious channels: {suspicious_channels}"
        )
        print(
            f"Bad channels:        {bad_channels}"
        )

        # ----------------------------------------------------
        # PRINT SUSPICIOUS CHANNELS
        # ----------------------------------------------------

        if suspicious_channels > 0:

            print()
            print("SUSPICIOUS CHANNELS")
            print("-" * 80)

            for ch_idx, ch_name in enumerate(
                epochs.ch_names
            ):

                if suspicious_channel_mask[ch_idx]:

                    print(
                        f"{ch_name:8s} "
                        f"max_abs={channel_max_abs[ch_idx]:10.3f} "
                        f"high_amp={channel_high_percent[ch_idx]:8.3f}%"
                    )

        # ----------------------------------------------------
        # EVENT INFORMATION
        # ----------------------------------------------------

        event_count = 0
        event_types = []

        try:

            if epochs.events is not None:

                event_count = len(
                    epochs.events
                )

                if epochs.event_id:

                    event_types = list(
                        epochs.event_id.keys()
                    )

        except Exception:

            event_count = 0
            event_types = []

        print()
        print("EVENT INFORMATION")
        print("-" * 80)
        print(f"Events: {event_count}")
        print(
            "Event types:",
            ", ".join(event_types)
            if event_types
            else "NONE"
        )

        # ----------------------------------------------------
        # STRUCTURAL QC
        # ----------------------------------------------------

        if n_channels != EXPECTED_CHANNELS:

            status = "REVIEW"
            reasons.append(
                f"CHANNEL_COUNT_{n_channels}"
            )

        if abs(sfreq - EXPECTED_SFREQ) > 0.01:

            status = "REVIEW"
            reasons.append(
                f"SFREQ_{sfreq}"
            )

        if n_times != EXPECTED_SAMPLES:

            status = "REVIEW"
            reasons.append(
                f"SAMPLES_{n_times}"
            )

        # ----------------------------------------------------
        # AMPLITUDE DECISION
        # ----------------------------------------------------

        if bad_amplitude_percent > BAD_HIGH_AMP_PERCENT:

            status = "BAD"
            reasons.append(
                "HIGH_GLOBAL_AMPLITUDE"
            )

        elif high_amplitude_percent > REVIEW_HIGH_AMP_PERCENT:

            if status != "BAD":
                status = "REVIEW"

            reasons.append(
                "ELEVATED_GLOBAL_AMPLITUDE"
            )

        if bad_channels > 0:

            status = "BAD"
            reasons.append(
                f"BAD_CHANNELS_{bad_channels}"
            )

        elif suspicious_channels > 0:

            if status != "BAD":
                status = "REVIEW"

            reasons.append(
                f"SUSPICIOUS_CHANNELS_{suspicious_channels}"
            )

        # ----------------------------------------------------
        # EVENT CHECK
        # ----------------------------------------------------

        if event_count == 0:

            status = "REVIEW"
            reasons.append(
                "NO_EVENTS"
            )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append({

            "file": filename,

            "n_epochs": n_epochs,

            "n_channels": n_channels,

            "n_times": n_times,

            "sfreq": sfreq,

            "epoch_duration_sec": duration,

            "nan_percent": nan_percent,

            "inf_percent": inf_percent,

            "global_min": global_min,

            "global_max": global_max,

            "global_std": global_std,

            "max_abs": max_abs,

            "high_amplitude_percent":
                high_amplitude_percent,

            "bad_amplitude_percent":
                bad_amplitude_percent,

            "suspicious_epochs":
                suspicious_epochs,

            "bad_epochs":
                bad_epochs,

            "suspicious_channels":
                suspicious_channels,

            "bad_channels":
                bad_channels,

            "event_count":
                event_count,

            "event_types":
                ";".join(event_types),

            "status":
                status,

            "reasons":
                ";".join(reasons)
        })

        print()
        print(f"STATUS: {status}")

        if reasons:
            print(
                "REASONS:",
                ";".join(reasons)
            )

    except Exception as e:

        print()
        print("ERROR:")
        print(str(e))

        results.append({

            "file": filename,

            "n_epochs": np.nan,

            "n_channels": np.nan,

            "n_times": np.nan,

            "sfreq": np.nan,

            "epoch_duration_sec": np.nan,

            "nan_percent": np.nan,

            "inf_percent": np.nan,

            "global_min": np.nan,

            "global_max": np.nan,

            "global_std": np.nan,

            "max_abs": np.nan,

            "high_amplitude_percent": np.nan,

            "bad_amplitude_percent": np.nan,

            "suspicious_epochs": np.nan,

            "bad_epochs": np.nan,

            "suspicious_channels": np.nan,

            "bad_channels": np.nan,

            "event_count": np.nan,

            "event_types": "",

            "status": "BAD",

            "reasons": "READ_ERROR"
        })

# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(results)

# ============================================================
# STATUS COUNTS
# ============================================================

status_counts = (
    df["status"]
    .value_counts()
)

pass_count = int(
    status_counts.get("PASS", 0)
)

review_count = int(
    status_counts.get("REVIEW", 0)
)

bad_count = int(
    status_counts.get("BAD", 0)
)

# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 80 + "\n")
    f.write("FINAL EPOCH QC SUMMARY - 83 RUNS\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        f"Expected epoch files: {EXPECTED_FILES}\n"
    )

    f.write(
        f"Found epoch files:    {len(files)}\n"
    )

    f.write(
        f"QC records:           {len(df)}\n\n"
    )

    f.write("=" * 80 + "\n")
    f.write("STATUS COUNTS\n")
    f.write("=" * 80 + "\n")

    f.write(
        status_counts.to_string()
    )

    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("TOTALS\n")
    f.write("=" * 80 + "\n")

    f.write(
        f"Total epochs: {int(df['n_epochs'].fillna(0).sum())}\n"
    )

    f.write(
        f"PASS files:   {pass_count}\n"
    )

    f.write(
        f"REVIEW files: {review_count}\n"
    )

    f.write(
        f"BAD files:    {bad_count}\n"
    )

    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("FILES REQUIRING REVIEW OR EXCLUSION\n")
    f.write("=" * 80 + "\n")

    problematic = df[
        df["status"].isin(
            ["REVIEW", "BAD"]
        )
    ]

    if len(problematic) == 0:

        f.write("NONE\n")

    else:

        cols = [
            "file",
            "n_epochs",
            "n_channels",
            "sfreq",
            "high_amplitude_percent",
            "bad_amplitude_percent",
            "suspicious_channels",
            "bad_channels",
            "status",
            "reasons"
        ]

        f.write(
            problematic[cols].to_string(
                index=False
            )
        )

    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("EPOCH COUNT DISTRIBUTION\n")
    f.write("=" * 80 + "\n")

    if df["n_epochs"].notna().any():

        f.write(
            df["n_epochs"]
            .describe()
            .to_string()
        )

    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("IMPORTANT\n")
    f.write("=" * 80 + "\n")

    f.write(
        "This script ONLY reads epochs_v3 FIF files.\n"
    )

    f.write(
        "NO RAW DATA WAS MODIFIED.\n"
    )

    f.write(
        "NO SET FILE WAS MODIFIED.\n"
    )

    f.write(
        "NO FDT FILE WAS MODIFIED.\n"
    )

    f.write(
        "NO EPOCH FILE WAS MODIFIED.\n"
    )

# ============================================================
# FINAL TERMINAL REPORT
# ============================================================

print()
print("=" * 80)
print("FINAL EPOCH QC COMPLETE")
print("=" * 80)

print()
print("STATUS COUNTS")
print(status_counts)

print()
print("=" * 80)
print("TOTAL")
print("=" * 80)

print(
    f"Epoch files found: {len(files)}"
)

print(
    f"Expected files:    {EXPECTED_FILES}"
)

print(
    f"Total epochs:      "
    f"{int(df['n_epochs'].fillna(0).sum())}"
)

print(
    f"PASS:              {pass_count}"
)

print(
    f"REVIEW:            {review_count}"
)

print(
    f"BAD:               {bad_count}"
)

print()
print("=" * 80)
print("FILES REQUIRING REVIEW")
print("=" * 80)

problematic = df[
    df["status"].isin(
        ["REVIEW", "BAD"]
    )
]

if len(problematic) == 0:

    print("NONE")

else:

    print(
        problematic[
            [
                "file",
                "n_epochs",
                "suspicious_channels",
                "bad_channels",
                "status",
                "reasons"
            ]
        ].to_string(index=False)
    )

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(
    OUTPUT_CSV
)

print(
    OUTPUT_SUMMARY
)

print()
print("=" * 80)
print("RAW DATA WAS NOT MODIFIED.")
print("NO EPOCH FILE WAS MODIFIED.")
print("=" * 80)