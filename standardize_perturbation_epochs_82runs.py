# ============================================================
# standardize_perturbation_epochs_82runs_v3.py
# ============================================================
#
# PURPOSE:
# Standardize 82 ELIGIBLE epoch files to:
#   SFREQ    = 500 Hz
#   TMIN     = -0.2 sec
#   N_TIMES  = 501
#   TMAX     = 0.8 sec metadata convention
#   CHANNELS = 71
#
# IMPORTANT:
# - No epochs are intentionally removed.
# - Original ELIGIBLE files are NEVER modified.
# - Floating-point tmax differences are handled safely.
# ============================================================

from pathlib import Path
import mne
import numpy as np
import pandas as pd
import traceback

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\epochs_clean"
    r"\\logs\perturbation_dataset\ELIGIBLE"
)

OUTPUT_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\epochs_clean"
    r"\\logs\perturbation_dataset_v3\ELIGIBLE"
)

LOG_DIR = OUTPUT_DIR.parent / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CSV_LOG = LOG_DIR / "standardization_82runs_v3_log.csv"
SUMMARY = LOG_DIR / "standardization_82runs_v3_summary.txt"

# ------------------------------------------------------------
# TARGET STANDARD
# ------------------------------------------------------------

TARGET_SFREQ = 500.0
TARGET_TMIN = -0.2
TARGET_N_TIMES = 501

EXPECTED_CHANNELS = 71

# The important point:
# 501 samples at 500 Hz starting at -0.2 gives sample indices:
# 0 ... 500
#
# MNE's time axis therefore ends at:
# -0.2 + 500/500 = 0.8
#
# Small floating point differences are tolerated.
TARGET_TMAX = TARGET_TMIN + (TARGET_N_TIMES - 1) / TARGET_SFREQ

# ------------------------------------------------------------
# FIND INPUT FILES
# ------------------------------------------------------------

files = sorted(INPUT_DIR.glob("*_perturbation_eligible_epo.fif"))

print("=" * 80)
print("PERTURBATION EPOCH STANDARDIZATION V3")
print("=" * 80)

print(f"Input directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Files found: {len(files)}")
print()

if len(files) != 82:
    print("WARNING:")
    print(f"Expected 82 files, found {len(files)}")

# ------------------------------------------------------------
# PROCESS
# ------------------------------------------------------------

records = []

total_input_epochs = 0
total_output_epochs = 0

successful = 0
failed = 0

for idx, input_file in enumerate(files, start=1):

    print("=" * 80)
    print(f"[{idx}/{len(files)}] {input_file.name}")
    print("=" * 80)

    output_file = (
        OUTPUT_DIR /
        input_file.name.replace(
            "_perturbation_eligible_epo.fif",
            "_standardized_epo.fif"
        )
    )

    rec = {
        "input_file": input_file.name,
        "output_file": output_file.name,
        "input_epochs": 0,
        "output_epochs": 0,
        "input_channels": 0,
        "output_channels": 0,
        "input_sfreq": np.nan,
        "output_sfreq": np.nan,
        "input_tmin": np.nan,
        "input_tmax": np.nan,
        "output_tmin": np.nan,
        "output_tmax": np.nan,
        "input_n_times": 0,
        "output_n_times": 0,
        "nan_percent": np.nan,
        "inf_percent": np.nan,
        "status": "FAILED",
        "reason": ""
    }

    try:

        # ----------------------------------------------------
        # READ
        # ----------------------------------------------------

        print("Reading epochs...")

        epochs = mne.read_epochs(
            input_file,
            preload=True,
            verbose=False
        )

        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        input_sfreq = float(epochs.info["sfreq"])

        rec["input_epochs"] = n_epochs
        rec["input_channels"] = n_channels
        rec["input_sfreq"] = input_sfreq
        rec["input_tmin"] = float(epochs.tmin)
        rec["input_tmax"] = float(epochs.tmax)
        rec["input_n_times"] = len(epochs.times)

        total_input_epochs += n_epochs

        print(f"Input epochs:      {n_epochs}")
        print(f"Input channels:    {n_channels}")
        print(f"Input sfreq:       {input_sfreq}")
        print(
            f"Input time range:  "
            f"{epochs.tmin:.12f} to {epochs.tmax:.12f}"
        )
        print(f"Input samples:     {len(epochs.times)}")

        # ----------------------------------------------------
        # CHANNEL CHECK
        # ----------------------------------------------------

        if n_channels != EXPECTED_CHANNELS:
            raise RuntimeError(
                f"CHANNELS_{n_channels}"
            )

        # ----------------------------------------------------
        # RESAMPLE
        # ----------------------------------------------------

        if not np.isclose(
            input_sfreq,
            TARGET_SFREQ,
            rtol=0,
            atol=1e-6
        ):

            print(
                f"Resampling {input_sfreq:.12f} Hz "
                f"-> {TARGET_SFREQ:.1f} Hz"
            )

            epochs.resample(
                TARGET_SFREQ,
                npad="auto",
                verbose=False
            )

        else:
            print("Sampling rate already 500 Hz")

        print(f"Samples after resampling: {len(epochs.times)}")
        print(
            f"Current time range: "
            f"{epochs.tmin:.12f} to {epochs.tmax:.12f}"
        )

        # ----------------------------------------------------
        # FORCE STANDARD TIME GRID
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We do NOT use crop(tmax=0.8).
        #
        # Instead we verify that the existing epoch contains
        # the required 501-sample grid and then explicitly set
        # the time metadata.
        #
        # For 500 Hz:
        #
        #   -0.2 + 500/500 = 0.8
        #
        # MNE may internally display values such as:
        # 0.7999999999999999
        #
        # This is floating-point representation, not missing data.
        # ----------------------------------------------------

        if len(epochs.times) != TARGET_N_TIMES:

            print(
                f"Adjusting sample count: "
                f"{len(epochs.times)} -> {TARGET_N_TIMES}"
            )

            # We expect the standardized data to represent
            # approximately -0.2 to 0.8 sec.

            current_times = epochs.times

            target_times = (
                TARGET_TMIN +
                np.arange(TARGET_N_TIMES) / TARGET_SFREQ
            )

            # Check that target grid lies within current grid
            # with a small numerical tolerance.

            current_start = float(current_times[0])
            current_end = float(current_times[-1])

            target_start = float(target_times[0])
            target_end = float(target_times[-1])

            if (
                target_start < current_start - 1e-6
                or
                target_end > current_end + 1e-6
            ):
                raise RuntimeError(
                    "TARGET_TIME_GRID_NOT_AVAILABLE"
                )

            # Find nearest indices for target grid
            indices = np.array([
                int(np.argmin(np.abs(current_times - t)))
                for t in target_times
            ])

            # Ensure unique indices
            if len(np.unique(indices)) != TARGET_N_TIMES:
                raise RuntimeError(
                    "TARGET_TIME_GRID_INDEX_COLLISION"
                )

            data = epochs.get_data()[:, :, indices]

            epochs = mne.EpochsArray(
                data,
                epochs.info.copy(),
                events=epochs.events.copy(),
                event_id=epochs.event_id.copy(),
                tmin=TARGET_TMIN,
                metadata=epochs.metadata.copy()
                if epochs.metadata is not None
                else None,
                verbose=False
            )

        else:

            print(
                "501 samples already present."
            )

            # ------------------------------------------------
            # Normalize time origin safely
            # ------------------------------------------------

            current_times = epochs.times

            if not np.isclose(
                current_times[0],
                TARGET_TMIN,
                atol=1e-5
            ):
                raise RuntimeError(
                    f"UNEXPECTED_TMIN_{current_times[0]}"
                )

            # Rebuild EpochsArray only to guarantee the exact
            # target tmin without using crop/tmax.
            data = epochs.get_data()

            epochs = mne.EpochsArray(
                data,
                epochs.info.copy(),
                events=epochs.events.copy(),
                event_id=epochs.event_id.copy(),
                tmin=TARGET_TMIN,
                metadata=epochs.metadata.copy()
                if epochs.metadata is not None
                else None,
                verbose=False
            )

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        print("Applying baseline correction...")

        epochs.apply_baseline(
            baseline=(TARGET_TMIN, 0.0),
            verbose=False
        )

        # ----------------------------------------------------
        # NUMERICAL QC
        # ----------------------------------------------------

        data = epochs.get_data()

        nan_percent = (
            np.isnan(data).sum() /
            data.size *
            100
        )

        inf_percent = (
            np.isinf(data).sum() /
            data.size *
            100
        )

        rec["nan_percent"] = nan_percent
        rec["inf_percent"] = inf_percent

        if nan_percent > 0:
            raise RuntimeError(
                f"NAN_{nan_percent:.6f}%"
            )

        if inf_percent > 0:
            raise RuntimeError(
                f"INF_{inf_percent:.6f}%"
            )

        # ----------------------------------------------------
        # FINAL EXPECTED VALUES
        # ----------------------------------------------------

        output_epochs = len(epochs)
        output_channels = len(epochs.ch_names)
        output_sfreq = float(epochs.info["sfreq"])
        output_n_times = len(epochs.times)

        rec["output_epochs"] = output_epochs
        rec["output_channels"] = output_channels
        rec["output_sfreq"] = output_sfreq
        rec["output_tmin"] = float(epochs.tmin)
        rec["output_tmax"] = float(epochs.tmax)
        rec["output_n_times"] = output_n_times

        # ----------------------------------------------------
        # STRICT VALIDATION
        # ----------------------------------------------------

        if output_epochs != n_epochs:
            raise RuntimeError(
                f"EPOCH_COUNT_CHANGED_{n_epochs}_TO_{output_epochs}"
            )

        if output_channels != EXPECTED_CHANNELS:
            raise RuntimeError(
                f"OUTPUT_CHANNELS_{output_channels}"
            )

        if not np.isclose(
            output_sfreq,
            TARGET_SFREQ,
            atol=1e-6
        ):
            raise RuntimeError(
                f"OUTPUT_SFREQ_{output_sfreq}"
            )

        if output_n_times != TARGET_N_TIMES:
            raise RuntimeError(
                f"OUTPUT_N_TIMES_{output_n_times}"
            )

        if not np.isclose(
            epochs.tmin,
            TARGET_TMIN,
            atol=1e-6
        ):
            raise RuntimeError(
                f"TMIN_{epochs.tmin}"
            )

        # DO NOT require exact tmax == 0.8 using equality.
        if not np.isclose(
            epochs.tmax,
            TARGET_TMAX,
            atol=1e-6
        ):
            raise RuntimeError(
                f"TMAX_{epochs.tmax}"
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        print()
        print("Saving:")
        print(output_file)

        epochs.save(
            output_file,
            overwrite=True,
            verbose=False
        )

        # ----------------------------------------------------
        # RELOAD VALIDATION
        # ----------------------------------------------------

        print()
        print("Reloading saved file...")

        check = mne.read_epochs(
            output_file,
            preload=True,
            verbose=False
        )

        check_data = check.get_data()

        check_nan = (
            np.isnan(check_data).sum() /
            check_data.size *
            100
        )

        check_inf = (
            np.isinf(check_data).sum() /
            check_data.size *
            100
        )

        if len(check) != n_epochs:
            raise RuntimeError(
                "RELOAD_EPOCH_COUNT_MISMATCH"
            )

        if len(check.ch_names) != EXPECTED_CHANNELS:
            raise RuntimeError(
                "RELOAD_CHANNEL_COUNT_MISMATCH"
            )

        if not np.isclose(
            check.info["sfreq"],
            TARGET_SFREQ,
            atol=1e-6
        ):
            raise RuntimeError(
                "RELOAD_SFREQ_MISMATCH"
            )

        if len(check.times) != TARGET_N_TIMES:
            raise RuntimeError(
                "RELOAD_TIMEPOINT_MISMATCH"
            )

        if check_nan > 0:
            raise RuntimeError(
                f"RELOAD_NAN_{check_nan:.6f}%"
            )

        if check_inf > 0:
            raise RuntimeError(
                f"RELOAD_INF_{check_inf:.6f}%"
            )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        rec["status"] = "SUCCESS"
        rec["reason"] = ""

        successful += 1
        total_output_epochs += output_epochs

        print()
        print("VALIDATION")
        print("-" * 70)
        print(f"Saved epochs:      {len(check)}")
        print(f"Saved channels:    {len(check.ch_names)}")
        print(f"Saved sfreq:       {check.info['sfreq']}")
        print(f"Saved samples:     {len(check.times)}")
        print(
            f"Saved time range:  "
            f"{check.tmin:.12f} to {check.tmax:.12f}"
        )
        print(f"NaN:               {check_nan:.6f}%")
        print(f"Inf:               {check_inf:.6f}%")
        print("STATUS: SUCCESS")

    except Exception as e:

        failed += 1

        rec["status"] = "FAILED"
        rec["reason"] = str(e)

        print()
        print("STATUS: FAILED")
        print(f"REASON: {e}")

    records.append(rec)

# ------------------------------------------------------------
# SAVE LOG
# ------------------------------------------------------------

df = pd.DataFrame(records)

df.to_csv(
    CSV_LOG,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

status_counts = df["status"].value_counts()

summary_lines = []

summary_lines.append("=" * 80)
summary_lines.append(
    "PERTURBATION EPOCH STANDARDIZATION V3 COMPLETE"
)
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append(
    f"Input files:          {len(files)}"
)
summary_lines.append(
    f"Successful:           {successful}"
)
summary_lines.append(
    f"Failed:               {failed}"
)
summary_lines.append(
    f"Input epochs:         {total_input_epochs}"
)
summary_lines.append(
    f"Output epochs:        {total_output_epochs}"
)
summary_lines.append("")
summary_lines.append("TARGET STANDARD")
summary_lines.append("-" * 80)
summary_lines.append(
    f"SFREQ:                {TARGET_SFREQ} Hz"
)
summary_lines.append(
    f"TMIN:                 {TARGET_TMIN} sec"
)
summary_lines.append(
    f"TMAX:                 {TARGET_TMAX} sec"
)
summary_lines.append(
    f"SAMPLES/EPOCH:        {TARGET_N_TIMES}"
)
summary_lines.append(
    f"CHANNELS:             {EXPECTED_CHANNELS}"
)
summary_lines.append("")

summary_lines.append("STATUS COUNTS")
summary_lines.append("-" * 80)

for status, count in status_counts.items():
    summary_lines.append(
        f"{status}: {count}"
    )

summary_lines.append("")

# ------------------------------------------------------------
# FINAL DATASET CHECK
# ------------------------------------------------------------

if (
    len(files) == 82
    and successful == 82
    and failed == 0
    and total_output_epochs == total_input_epochs
):

    final_status = "PASS"

else:

    final_status = "REVIEW"

summary_lines.append("=" * 80)
summary_lines.append("FINAL DATASET STANDARDIZATION STATUS")
summary_lines.append("=" * 80)
summary_lines.append(
    f"FINAL STATUS: {final_status}"
)
summary_lines.append("")

if failed > 0:

    summary_lines.append("FAILED FILES")
    summary_lines.append("-" * 80)

    failed_df = df[df["status"] == "FAILED"]

    for _, row in failed_df.iterrows():

        summary_lines.append(
            f"{row['input_file']} | {row['reason']}"
        )

    summary_lines.append("")

summary_lines.append("=" * 80)
summary_lines.append("OUTPUT")
summary_lines.append("=" * 80)
summary_lines.append(
    f"Directory: {OUTPUT_DIR}"
)
summary_lines.append(
    f"CSV log:   {CSV_LOG}"
)
summary_lines.append(
    f"Summary:   {SUMMARY}"
)
summary_lines.append("")
summary_lines.append("=" * 80)
summary_lines.append(
    "ORIGINAL ELIGIBLE FILES WERE NOT MODIFIED."
)
summary_lines.append(
    "ONLY NEW STANDARDIZED FILES WERE CREATED."
)
summary_lines.append("=" * 80)

SUMMARY.write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)

# ------------------------------------------------------------
# CONSOLE SUMMARY
# ------------------------------------------------------------

print()
print("=" * 80)
print("PERTURBATION EPOCH STANDARDIZATION V3 COMPLETE")
print("=" * 80)

print()
print(f"Input files:          {len(files)}")
print(f"Successful:           {successful}")
print(f"Failed:               {failed}")
print(f"Input epochs:         {total_input_epochs}")
print(f"Output epochs:        {total_output_epochs}")

print()
print("TARGET STANDARD")
print("-" * 80)
print(f"SFREQ:                {TARGET_SFREQ} Hz")
print(f"TMIN:                 {TARGET_TMIN} sec")
print(f"TMAX:                 {TARGET_TMAX} sec")
print(f"SAMPLES/EPOCH:        {TARGET_N_TIMES}")
print(f"CHANNELS:             {EXPECTED_CHANNELS}")

print()
print("=" * 80)
print("FINAL STATUS:", final_status)
print("=" * 80)

print()
print("Saved:")
print(OUTPUT_DIR)
print(CSV_LOG)
print(SUMMARY)

print()
print("=" * 80)
print("ORIGINAL ELIGIBLE FILES WERE NOT MODIFIED.")
print("ONLY NEW STANDARDIZED FILES WERE CREATED.")
print("=" * 80)