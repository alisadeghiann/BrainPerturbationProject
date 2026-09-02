import os
import glob
import numpy as np
import pandas as pd
import mne

# ============================================================
# FINAL CLEAN EPOCH QC - 82 RUNS
# READ-ONLY QC
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"
INPUT_DIR = os.path.join(BASE_DIR, "epochs_clean")
LOG_DIR = os.path.join(INPUT_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

CSV_OUT = os.path.join(LOG_DIR, "final_clean_epoch_qc_82runs.csv")
TXT_OUT = os.path.join(LOG_DIR, "final_clean_epoch_qc_82runs_summary.txt")

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_clean_epo.fif")))

print("=" * 80)
print("FINAL CLEAN EPOCH QC - 82 RUNS")
print("=" * 80)
print(f"Files found: {len(files)}")
print()

records = []

for i, path in enumerate(files, 1):

    fname = os.path.basename(path)

    print("=" * 80)
    print(f"[{i}/{len(files)}] {fname}")
    print("=" * 80)

    status = "PASS"
    reasons = []

    try:
        epochs = mne.read_epochs(path, preload=True, verbose=False)

        data = epochs.get_data(copy=False)

        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        n_times = len(epochs.times)
        sfreq = float(epochs.info["sfreq"])

        duration = float(epochs.times[-1] - epochs.times[0])

        nan_percent = float(np.isnan(data).mean() * 100)
        inf_percent = float(np.isinf(data).mean() * 100)

        finite = np.isfinite(data)

        if finite.any():
            finite_data = data[finite]

            global_min = float(np.min(finite_data))
            global_max = float(np.max(finite_data))
            global_std = float(np.std(finite_data))
            max_abs = float(np.max(np.abs(finite_data)))

            high_amp_percent = float(
                (np.abs(data[finite]) > 200).mean() * 100
            )

            bad_amp_percent = float(
                (np.abs(data[finite]) > 300).mean() * 100
            )
        else:
            global_min = np.nan
            global_max = np.nan
            global_std = np.nan
            max_abs = np.nan
            high_amp_percent = np.nan
            bad_amp_percent = np.nan

        # ----------------------------------------------------
        # Epoch-level amplitude screening
        # ----------------------------------------------------

        epoch_max_abs = np.nanmax(np.abs(data), axis=(1, 2))

        suspicious_epochs = int(np.sum(epoch_max_abs > 200))
        bad_epochs = int(np.sum(epoch_max_abs > 300))

        # ----------------------------------------------------
        # Channel-level screening
        # ----------------------------------------------------

        channel_std = np.nanstd(data, axis=(0, 2))

        median_channel_std = float(np.nanmedian(channel_std))

        if median_channel_std > 0:
            channel_ratio = channel_std / median_channel_std
        else:
            channel_ratio = np.zeros_like(channel_std)

        suspicious_channel_idx = np.where(channel_ratio > 3.0)[0]
        bad_channel_idx = np.where(channel_ratio > 5.0)[0]

        suspicious_channels = int(len(suspicious_channel_idx))
        bad_channels = int(len(bad_channel_idx))

        # ----------------------------------------------------
        # Basic validity
        # ----------------------------------------------------

        if n_epochs == 0:
            status = "BAD"
            reasons.append("ZERO_EPOCHS")

        if n_channels != 71:
            status = "REVIEW"
            reasons.append(f"CHANNELS_{n_channels}")

        if n_times != 501:
            status = "REVIEW"
            reasons.append(f"TIMEPOINTS_{n_times}")

        if not np.isclose(sfreq, 500.0, atol=0.1):
            status = "REVIEW"
            reasons.append(f"SFREQ_{sfreq}")

        if nan_percent > 0:
            status = "BAD"
            reasons.append("NAN_PRESENT")

        if inf_percent > 0:
            status = "BAD"
            reasons.append("INF_PRESENT")

        # ----------------------------------------------------
        # Amplitude QC
        # ----------------------------------------------------

        if bad_amp_percent > 1.0:
            status = "REVIEW"
            reasons.append("ELEVATED_BAD_AMPLITUDE")

        if high_amp_percent > 5.0:
            status = "REVIEW"
            reasons.append("ELEVATED_HIGH_AMPLITUDE")

        if suspicious_epochs > max(5, int(n_epochs * 0.05)):
            status = "REVIEW"
            reasons.append(
                f"SUSPICIOUS_EPOCHS_{suspicious_epochs}"
            )

        if bad_epochs > max(2, int(n_epochs * 0.02)):
            status = "REVIEW"
            reasons.append(
                f"BAD_EPOCHS_{bad_epochs}"
            )

        # ----------------------------------------------------
        # Channel QC
        # ----------------------------------------------------

        if suspicious_channels > 0:
            if status == "PASS":
                status = "REVIEW"

            reasons.append(
                f"SUSPICIOUS_CHANNELS_{suspicious_channels}"
            )

        if bad_channels > 0:
            status = "BAD"
            reasons.append(
                f"BAD_CHANNELS_{bad_channels}"
            )

        # ----------------------------------------------------
        # Event QC
        # ----------------------------------------------------

        event_count = 0
        event_types = ""

        if hasattr(epochs, "events") and epochs.events is not None:
            event_count = int(len(epochs.events))

        if hasattr(epochs, "event_id") and epochs.event_id:
            event_types = ";".join(
                sorted(epochs.event_id.keys())
            )

        if event_count == 0:
            status = "REVIEW"
            reasons.append("NO_EVENTS")

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline = epochs.baseline

        if baseline is None:
            status = "REVIEW"
            reasons.append("NO_BASELINE")

        # ----------------------------------------------------
        # Final record
        # ----------------------------------------------------

        reason_text = ";".join(reasons) if reasons else "OK"

        records.append({
            "file": fname,
            "n_epochs": n_epochs,
            "n_channels": n_channels,
            "n_times": n_times,
            "sfreq": sfreq,
            "epoch_start_sec": float(epochs.times[0]),
            "epoch_end_sec": float(epochs.times[-1]),
            "epoch_duration_sec": duration,
            "baseline": str(baseline),
            "nan_percent": nan_percent,
            "inf_percent": inf_percent,
            "global_min": global_min,
            "global_max": global_max,
            "global_std": global_std,
            "max_abs": max_abs,
            "high_amplitude_percent": high_amp_percent,
            "bad_amplitude_percent": bad_amp_percent,
            "suspicious_epochs": suspicious_epochs,
            "bad_epochs": bad_epochs,
            "suspicious_channels": suspicious_channels,
            "bad_channels": bad_channels,
            "event_count": event_count,
            "event_types": event_types,
            "status": status,
            "reasons": reason_text
        })

        print(f"Epochs:              {n_epochs}")
        print(f"Channels:             {n_channels}")
        print(f"Time points:          {n_times}")
        print(f"Sampling rate:        {sfreq}")
        print(f"Baseline:             {baseline}")
        print(f"NaN:                  {nan_percent:.6f}%")
        print(f"Inf:                  {inf_percent:.6f}%")
        print(f"Global STD:           {global_std:.6f}")
        print(f"Max abs:              {max_abs:.6f}")
        print(f">200 uV:              {high_amp_percent:.6f}%")
        print(f">300 uV:              {bad_amp_percent:.6f}%")
        print(f"Suspicious epochs:    {suspicious_epochs}")
        print(f"Bad epochs:           {bad_epochs}")
        print(f"Suspicious channels:  {suspicious_channels}")
        print(f"Bad channels:         {bad_channels}")
        print(f"Events:               {event_count}")
        print(f"STATUS:               {status}")
        print(f"REASONS:              {reason_text}")

    except Exception as e:

        print(f"ERROR: {e}")

        records.append({
            "file": fname,
            "n_epochs": np.nan,
            "n_channels": np.nan,
            "n_times": np.nan,
            "sfreq": np.nan,
            "epoch_start_sec": np.nan,
            "epoch_end_sec": np.nan,
            "epoch_duration_sec": np.nan,
            "baseline": "",
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
            "status": "FAILED",
            "reasons": str(e)
        })

# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(records)

df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

# ============================================================
# SUMMARY
# ============================================================

status_counts = df["status"].value_counts()

summary_lines = []

summary_lines.append("=" * 80)
summary_lines.append("FINAL CLEAN EPOCH QC - 82 RUNS")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append(f"Files found: {len(files)}")
summary_lines.append(f"Records generated: {len(df)}")
summary_lines.append("")
summary_lines.append("STATUS COUNTS")
summary_lines.append("-" * 80)

for status, count in status_counts.items():
    summary_lines.append(f"{status:10s} {count}")

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("EPOCH COUNT DISTRIBUTION")
summary_lines.append("=" * 80)

if "n_epochs" in df.columns:
    summary_lines.append(
        df["n_epochs"].describe().to_string()
    )

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("REVIEW / BAD FILES")
summary_lines.append("=" * 80)

problem_df = df[df["status"] != "PASS"]

if len(problem_df) == 0:
    summary_lines.append("NONE")
else:
    summary_lines.append(
        problem_df[
            ["file", "n_epochs", "suspicious_epochs",
             "bad_epochs", "suspicious_channels",
             "bad_channels", "status", "reasons"]
        ].to_string(index=False)
    )

summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append("IMPORTANT")
summary_lines.append("=" * 80)
summary_lines.append("This script ONLY reads epochs_clean FIF files.")
summary_lines.append("NO RAW DATA WAS MODIFIED.")
summary_lines.append("NO SET FILE WAS MODIFIED.")
summary_lines.append("NO FDT FILE WAS MODIFIED.")
summary_lines.append("NO EPOCH FILE WAS MODIFIED.")

with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("")
print("=" * 80)
print("FINAL CLEAN EPOCH QC COMPLETE")
print("=" * 80)

print("")
print("STATUS COUNTS")
print(status_counts)

print("")
print("Saved:")
print(CSV_OUT)
print(TXT_OUT)

print("")
print("=" * 80)
print("NO DATA WAS MODIFIED.")
print("=" * 80)