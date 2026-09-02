from pathlib import Path
import mne
import numpy as np
import pandas as pd
import traceback

# ============================================================
# CONFIG
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT_DIR = (
    BASE
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset"
    / "ELIGIBLE"
)

OUTPUT_DIR = (
    BASE
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v4"
    / "ELIGIBLE"
)

LOG_DIR = OUTPUT_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SFREQ = 500.0
TARGET_TMIN = -0.2
TARGET_TMAX = 0.8
TARGET_N_TIMES = 501

EXPECTED_CHANNELS = 71
EXPECTED_FILES = 82
EXPECTED_EPOCHS = 21816

# ============================================================
# HELPERS
# ============================================================

def force_standard_grid(epochs):
    """
    Force exact target time grid using interpolation.
    This avoids floating-point TMAX/TMIN problems.
    """

    target_times = np.linspace(
        TARGET_TMIN,
        TARGET_TMAX,
        TARGET_N_TIMES
    )

    current_times = epochs.times.copy()

    data = epochs.get_data()

    # Interpolate each epoch/channel onto exact target grid
    new_data = np.empty(
        (
            data.shape[0],
            data.shape[1],
            TARGET_N_TIMES
        ),
        dtype=data.dtype
    )

    for e in range(data.shape[0]):
        for ch in range(data.shape[1]):
            new_data[e, ch, :] = np.interp(
                target_times,
                current_times,
                data[e, ch, :]
            )

    info = epochs.info.copy()

    new_epochs = mne.EpochsArray(
        new_data,
        info,
        events=epochs.events.copy(),
        event_id=epochs.event_id.copy(),
        tmin=TARGET_TMIN,
        metadata=epochs.metadata.copy() if epochs.metadata is not None else None,
        verbose=False
    )

    return new_epochs


# ============================================================
# FIND FILES
# ============================================================

files = sorted(INPUT_DIR.glob("*_perturbation_eligible_epo.fif"))

print("=" * 80)
print("PERTURBATION EPOCH STANDARDIZATION V4")
print("=" * 80)

print(f"Input directory:  {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Files found:      {len(files)}")
print()

if len(files) != EXPECTED_FILES:
    print(
        f"WARNING: Expected {EXPECTED_FILES} files, "
        f"found {len(files)}"
    )

# ============================================================
# PROCESS
# ============================================================

records = []

total_input_epochs = 0
total_output_epochs = 0
successful = 0
failed = 0

for idx, path in enumerate(files, start=1):

    print("=" * 80)
    print(f"[{idx}/{len(files)}] {path.name}")
    print("=" * 80)

    rec = {
        "input_file": path.name,
        "output_file": "",
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

        print("Reading epochs...")

        epochs = mne.read_epochs(
            path,
            preload=True,
            verbose=False
        )

        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        sfreq = float(epochs.info["sfreq"])
        n_times = len(epochs.times)
        tmin = float(epochs.tmin)
        tmax = float(epochs.tmax)

        rec["input_epochs"] = n_epochs
        rec["input_channels"] = n_channels
        rec["input_sfreq"] = sfreq
        rec["input_tmin"] = tmin
        rec["input_tmax"] = tmax
        rec["input_n_times"] = n_times

        total_input_epochs += n_epochs

        print(f"Input epochs:      {n_epochs}")
        print(f"Input channels:    {n_channels}")
        print(f"Input sfreq:       {sfreq}")
        print(f"Input time range:  {tmin:.12f} to {tmax:.12f}")
        print(f"Input samples:     {n_times}")

        # ----------------------------------------------------
        # RESAMPLE
        # ----------------------------------------------------

        if not np.isclose(sfreq, TARGET_SFREQ, atol=1e-9):

            print()
            print(
                f"Resampling {sfreq} Hz -> "
                f"{TARGET_SFREQ} Hz"
            )

            epochs.resample(
                TARGET_SFREQ,
                npad="auto",
                verbose=False
            )

        else:
            print("Sampling rate already 500 Hz")

        print(
            f"Samples after resampling: "
            f"{len(epochs.times)}"
        )

        print(
            f"Current time range: "
            f"{epochs.tmin:.12f} to "
            f"{epochs.tmax:.12f}"
        )

        # ----------------------------------------------------
        # EXACT TIME GRID
        # ----------------------------------------------------

        print()
        print("Creating exact target time grid...")

        epochs = force_standard_grid(epochs)

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        print("Applying baseline correction...")

        epochs.apply_baseline(
            baseline=(TARGET_TMIN, 0.0)
        )

        # ----------------------------------------------------
        # VALIDATION BEFORE SAVE
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

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        output_name = (
            path.name.replace(
                "_perturbation_eligible_epo.fif",
                "_standardized_epo.fif"
            )
        )

        output_path = OUTPUT_DIR / output_name

        print()
        print("Saving:")
        print(output_path)

        epochs.save(
            output_path,
            overwrite=True,
            verbose=False
        )

        # ----------------------------------------------------
        # RELOAD VALIDATION
        # ----------------------------------------------------

        print()
        print("Reloading saved file...")

        check = mne.read_epochs(
            output_path,
            preload=True,
            verbose=False
        )

        saved_epochs = len(check)
        saved_channels = len(check.ch_names)
        saved_sfreq = float(check.info["sfreq"])
        saved_samples = len(check.times)
        saved_tmin = float(check.tmin)
        saved_tmax = float(check.tmax)

        print()
        print("VALIDATION")
        print("-" * 70)

        print(f"Saved epochs:      {saved_epochs}")
        print(f"Saved channels:    {saved_channels}")
        print(f"Saved sfreq:       {saved_sfreq}")
        print(f"Saved samples:     {saved_samples}")
        print(
            f"Saved time range:  "
            f"{saved_tmin:.12f} to "
            f"{saved_tmax:.12f}"
        )
        print(f"NaN:               {nan_percent:.6f}%")
        print(f"Inf:               {inf_percent:.6f}%")

        # ----------------------------------------------------
        # STRICT VALIDATION
        # ----------------------------------------------------

        problems = []

        if saved_channels != EXPECTED_CHANNELS:
            problems.append(
                f"CHANNELS_{saved_channels}"
            )

        if not np.isclose(
            saved_sfreq,
            TARGET_SFREQ,
            atol=1e-6
        ):
            problems.append(
                f"SFREQ_{saved_sfreq}"
            )

        if saved_samples != TARGET_N_TIMES:
            problems.append(
                f"SAMPLES_{saved_samples}"
            )

        if not np.isclose(
            saved_tmin,
            TARGET_TMIN,
            atol=1e-9
        ):
            problems.append(
                f"TMIN_{saved_tmin}"
            )

        if not np.isclose(
            saved_tmax,
            TARGET_TMAX,
            atol=1e-9
        ):
            problems.append(
                f"TMAX_{saved_tmax}"
            )

        if nan_percent != 0:
            problems.append("NAN")

        if inf_percent != 0:
            problems.append("INF")

        if saved_epochs != n_epochs:
            problems.append(
                f"EPOCH_COUNT_CHANGED_{n_epochs}_TO_{saved_epochs}"
            )

        if problems:
            rec["reason"] = ";".join(problems)
            rec["status"] = "FAILED"

            print("STATUS: FAILED")
            print("REASONS:", rec["reason"])

            failed += 1

        else:

            rec["output_file"] = output_name
            rec["output_epochs"] = saved_epochs
            rec["output_channels"] = saved_channels
            rec["output_sfreq"] = saved_sfreq
            rec["output_tmin"] = saved_tmin
            rec["output_tmax"] = saved_tmax
            rec["output_n_times"] = saved_samples
            rec["status"] = "SUCCESS"

            total_output_epochs += saved_epochs
            successful += 1

            print("STATUS: SUCCESS")

    except Exception as e:

        failed += 1

        rec["status"] = "FAILED"
        rec["reason"] = str(e)

        print()
        print("STATUS: FAILED")
        print("ERROR:")
        print(str(e))

    records.append(rec)


# ============================================================
# SAVE LOG
# ============================================================

df = pd.DataFrame(records)

log_path = LOG_DIR / "standardization_82runs_v4_log.csv"
summary_path = LOG_DIR / "standardization_82runs_v4_summary.txt"

df.to_csv(
    log_path,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("PERTURBATION EPOCH STANDARDIZATION V4 COMPLETE")
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
print("EXPECTED DATASET")
print("-" * 80)
print(f"Expected files:       {EXPECTED_FILES}")
print(f"Expected epochs:      {EXPECTED_EPOCHS}")

print()
print("FINAL STATUS")
print("-" * 80)

if (
    successful == EXPECTED_FILES
    and failed == 0
    and total_input_epochs == EXPECTED_EPOCHS
    and total_output_epochs == EXPECTED_EPOCHS
):
    final_status = "PASS"
    print("PASS - ALL 82 FILES STANDARDIZED SUCCESSFULLY")
else:
    final_status = "REVIEW"
    print("REVIEW - DATASET STILL REQUIRES INVESTIGATION")

# ============================================================
# WRITE SUMMARY
# ============================================================

with open(summary_path, "w", encoding="utf-8") as f:

    f.write("=" * 80 + "\n")
    f.write("PERTURBATION EPOCH STANDARDIZATION V4 SUMMARY\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Input files: {len(files)}\n")
    f.write(f"Successful: {successful}\n")
    f.write(f"Failed: {failed}\n")
    f.write(f"Input epochs: {total_input_epochs}\n")
    f.write(f"Output epochs: {total_output_epochs}\n\n")

    f.write("TARGET STANDARD\n")
    f.write("-" * 80 + "\n")
    f.write(f"SFREQ: {TARGET_SFREQ}\n")
    f.write(f"TMIN: {TARGET_TMIN}\n")
    f.write(f"TMAX: {TARGET_TMAX}\n")
    f.write(f"SAMPLES: {TARGET_N_TIMES}\n")
    f.write(f"CHANNELS: {EXPECTED_CHANNELS}\n\n")

    f.write(f"FINAL STATUS: {final_status}\n\n")

    if failed > 0:

        f.write("FAILED FILES\n")
        f.write("-" * 80 + "\n")

        for _, row in df[df["status"] == "FAILED"].iterrows():

            f.write(
                f"{row['input_file']} | "
                f"{row['reason']}\n"
            )

    f.write("\n")
    f.write("ORIGINAL ELIGIBLE FILES WERE NOT MODIFIED.\n")
    f.write("ONLY NEW STANDARDIZED FILES WERE CREATED.\n")

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(log_path)
print(summary_path)

print()
print("ORIGINAL ELIGIBLE FILES WERE NOT MODIFIED.")
print("ONLY NEW STANDARDIZED FILES WERE CREATED.")