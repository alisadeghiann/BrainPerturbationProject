# ============================================================
# FINAL QC - PERTURBATION DATASET V5
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import mne

# ============================================================
# PATHS
# ============================================================

BASE = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

INPUT_DIR = (
    BASE
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v5"
    / "ELIGIBLE"
)

OUTPUT_DIR = (
    BASE
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v5"
    / "final_qc"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_OUT = (
    OUTPUT_DIR
    / "final_qc_perturbation_v5.csv"
)

SUMMARY_OUT = (
    OUTPUT_DIR
    / "final_qc_perturbation_v5_summary.txt"
)

# ============================================================
# EXPECTED STANDARD
# ============================================================

EXPECTED_FILES = 82
EXPECTED_EPOCHS = 21816
EXPECTED_CHANNELS = 71
EXPECTED_SFREQ = 500.0
EXPECTED_N_TIMES = 501
EXPECTED_TMIN = -0.2
EXPECTED_TMAX = 0.8

# ============================================================
# EVENT MAPPING
# ============================================================

EXPECTED_EVENT_ID = {
    "left_click": 1,
    "right_click": 2,
    "show_cross": 3,
    "show_dash": 4,
    "show_letter": 5,
    "sound_beep": 6,
    "sound_buzz": 7,
    "right_click/show_cross": 8,
}

# ============================================================
# FIND FILES
# ============================================================

files = sorted(
    INPUT_DIR.glob(
        "*_standardized_epo.fif"
    )
)

print("=" * 80)
print("FINAL QC - PERTURBATION DATASET V5")
print("=" * 80)

print()
print(f"Input directory:")
print(INPUT_DIR)

print()
print(f"Files found: {len(files)}")
print(f"Expected:    {EXPECTED_FILES}")

# ============================================================
# RECORDS
# ============================================================

records = []

# ============================================================
# PROCESS FILES
# ============================================================

for i, f in enumerate(files, 1):

    print()
    print("=" * 80)
    print(f"[{i}/{len(files)}] {f.name}")
    print("=" * 80)

    rec = {
        "file": f.name,
        "status": "ERROR",
        "reasons": "",
        "epochs": 0,
        "channels": 0,
        "sfreq": np.nan,
        "n_times": 0,
        "tmin": np.nan,
        "tmax": np.nan,
        "nan_percent": np.nan,
        "inf_percent": np.nan,
        "global_min": np.nan,
        "global_max": np.nan,
        "global_std": np.nan,
        "max_abs": np.nan,
        "event_count": 0,
        "unique_event_ids": 0,
        "event_mapping": "",
        "channel_check": "",
        "sfreq_check": "",
        "time_check": "",
        "n_times_check": "",
        "nan_check": "",
        "inf_check": "",
        "event_check": "",
    }

    reasons = []

    try:

        # ----------------------------------------------------
        # READ
        # ----------------------------------------------------

        epochs = mne.read_epochs(
            f,
            preload=True,
            verbose=False
        )

        data = epochs.get_data()

        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        n_times = len(epochs.times)

        sfreq = float(
            epochs.info["sfreq"]
        )

        tmin = float(
            epochs.tmin
        )

        tmax = float(
            epochs.tmax
        )

        rec["epochs"] = n_epochs
        rec["channels"] = n_channels
        rec["sfreq"] = sfreq
        rec["n_times"] = n_times
        rec["tmin"] = tmin
        rec["tmax"] = tmax

        print(f"Epochs:        {n_epochs}")
        print(f"Channels:      {n_channels}")
        print(f"Samples:       {n_times}")
        print(f"Sampling rate: {sfreq}")
        print(
            f"Time range:    "
            f"{tmin:.12f} to {tmax:.12f}"
        )

        # ----------------------------------------------------
        # CHANNEL CHECK
        # ----------------------------------------------------

        if n_channels == EXPECTED_CHANNELS:

            rec["channel_check"] = "PASS"

        else:

            rec["channel_check"] = "FAIL"

            reasons.append(
                f"CHANNELS_{n_channels}"
            )

        # ----------------------------------------------------
        # SFREQ CHECK
        # ----------------------------------------------------

        if np.isclose(
            sfreq,
            EXPECTED_SFREQ,
            atol=1e-6
        ):

            rec["sfreq_check"] = "PASS"

        else:

            rec["sfreq_check"] = "FAIL"

            reasons.append(
                f"SFREQ_{sfreq}"
            )

        # ----------------------------------------------------
        # TIME CHECK
        # ----------------------------------------------------

        if (
            np.isclose(tmin, EXPECTED_TMIN, atol=1e-6)
            and
            np.isclose(tmax, EXPECTED_TMAX, atol=1e-6)
        ):

            rec["time_check"] = "PASS"

        else:

            rec["time_check"] = "FAIL"

            reasons.append(
                f"TIME_{tmin}_{tmax}"
            )

        # ----------------------------------------------------
        # TIMEPOINT CHECK
        # ----------------------------------------------------

        if n_times == EXPECTED_N_TIMES:

            rec["n_times_check"] = "PASS"

        else:

            rec["n_times_check"] = "FAIL"

            reasons.append(
                f"TIMEPOINTS_{n_times}"
            )

        # ----------------------------------------------------
        # NAN / INF
        # ----------------------------------------------------

        total_values = data.size

        nan_count = int(
            np.isnan(data).sum()
        )

        inf_count = int(
            np.isinf(data).sum()
        )

        nan_percent = (
            100.0 * nan_count / total_values
            if total_values > 0
            else 0.0
        )

        inf_percent = (
            100.0 * inf_count / total_values
            if total_values > 0
            else 0.0
        )

        rec["nan_percent"] = nan_percent
        rec["inf_percent"] = inf_percent

        if nan_count == 0:

            rec["nan_check"] = "PASS"

        else:

            rec["nan_check"] = "FAIL"

            reasons.append(
                f"NAN_{nan_percent:.6f}%"
            )

        if inf_count == 0:

            rec["inf_check"] = "PASS"

        else:

            rec["inf_check"] = "FAIL"

            reasons.append(
                f"INF_{inf_percent:.6f}%"
            )

        # ----------------------------------------------------
        # DATA RANGE
        # ----------------------------------------------------

        finite = data[
            np.isfinite(data)
        ]

        if finite.size > 0:

            rec["global_min"] = float(
                np.min(finite)
            )

            rec["global_max"] = float(
                np.max(finite)
            )

            rec["global_std"] = float(
                np.std(finite)
            )

            rec["max_abs"] = float(
                np.max(np.abs(finite))
            )

        print(
            f"NaN: {nan_percent:.6f}% | "
            f"Inf: {inf_percent:.6f}%"
        )

        print(
            f"Range: "
            f"{rec['global_min']:.3f} "
            f"to "
            f"{rec['global_max']:.3f}"
        )

        # ----------------------------------------------------
        # EVENT CHECK
        # ----------------------------------------------------

        if epochs.events is None:

            rec["event_check"] = "FAIL"

            reasons.append(
                "NO_EVENTS"
            )

        else:

            event_ids = (
                epochs.events[:, 2]
            )

            rec["event_count"] = len(
                event_ids
            )

            rec["unique_event_ids"] = len(
                np.unique(event_ids)
            )

            if len(event_ids) == n_epochs:

                rec["event_check"] = "PASS"

            else:

                rec["event_check"] = "FAIL"

                reasons.append(
                    f"EVENT_MISMATCH_"
                    f"{len(event_ids)}_"
                    f"{n_epochs}"
                )

            # ------------------------------------------------
            # EVENT MAPPING
            # ------------------------------------------------

            current_mapping = (
                epochs.event_id
            )

            mapping_string = ";".join(
                f"{k}={v}"
                for k, v in sorted(
                    current_mapping.items(),
                    key=lambda x: x[1]
                )
            )

            rec["event_mapping"] = (
                mapping_string
            )

            if current_mapping != EXPECTED_EVENT_ID:

                # Some files may legitimately
                # lack an event type.
                # Check IDs actually present.

                present_ids = set(
                    np.unique(event_ids)
                )

                expected_present = {
                    v
                    for v in EXPECTED_EVENT_ID.values()
                    if v in present_ids
                }

                mapping_values = set(
                    current_mapping.values()
                )

                if not expected_present.issubset(
                    mapping_values
                ):

                    reasons.append(
                        "EVENT_MAPPING_MISMATCH"
                    )

            print(
                f"Events: {rec['event_count']} | "
                f"Unique IDs: "
                f"{rec['unique_event_ids']}"
            )

        # ----------------------------------------------------
        # ZERO EPOCH CHECK
        # ----------------------------------------------------

        if n_epochs <= 0:

            reasons.append(
                "ZERO_EPOCHS"
            )

        # ----------------------------------------------------
        # FINAL STATUS
        # ----------------------------------------------------

        if len(reasons) == 0:

            rec["status"] = "PASS"
            rec["reasons"] = "NONE"

        else:

            rec["status"] = "FAIL"
            rec["reasons"] = ";".join(
                reasons
            )

        print(
            f"STATUS: {rec['status']}"
        )

        if rec["reasons"] != "NONE":

            print(
                f"REASONS: {rec['reasons']}"
            )

    except Exception as e:

        rec["status"] = "ERROR"

        rec["reasons"] = (
            "EXCEPTION_"
            + str(e)
            .replace(";", ",")
            [:500]
        )

        print("ERROR:")
        print(e)

    records.append(rec)

# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)

df.to_csv(
    CSV_OUT,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# COUNTS
# ============================================================

total_files = len(df)

pass_files = int(
    (df["status"] == "PASS").sum()
)

fail_files = int(
    (df["status"] == "FAIL").sum()
)

error_files = int(
    (df["status"] == "ERROR").sum()
)

total_epochs = int(
    df["epochs"].sum()
)

# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.append("=" * 80)
summary.append(
    "FINAL QC - PERTURBATION DATASET V5"
)
summary.append("=" * 80)
summary.append("")

summary.append(
    f"Files found:       {total_files}"
)

summary.append(
    f"Expected files:    {EXPECTED_FILES}"
)

summary.append(
    f"PASS:              {pass_files}"
)

summary.append(
    f"FAIL:              {fail_files}"
)

summary.append(
    f"ERROR:             {error_files}"
)

summary.append(
    f"Total epochs:      {total_epochs}"
)

summary.append(
    f"Expected epochs:   {EXPECTED_EPOCHS}"
)

summary.append("")

# ============================================================
# CHECK COUNTS
# ============================================================

for col in [
    "channel_check",
    "sfreq_check",
    "time_check",
    "n_times_check",
    "nan_check",
    "inf_check",
    "event_check",
]:

    summary.append("=" * 80)

    summary.append(
        col.upper()
    )

    summary.append("=" * 80)

    counts = (
        df[col]
        .value_counts()
    )

    for key, value in counts.items():

        summary.append(
            f"{key}: {value}"
        )

    summary.append("")

# ============================================================
# FAILED FILES
# ============================================================

summary.append("=" * 80)
summary.append("FAILED / ERROR FILES")
summary.append("=" * 80)
summary.append("")

failed = df[
    df["status"] != "PASS"
]

if len(failed) == 0:

    summary.append(
        "NONE"
    )

else:

    for _, row in failed.iterrows():

        summary.append(
            f"{row['file']} | "
            f"{row['status']} | "
            f"{row['reasons']}"
        )

summary.append("")

# ============================================================
# FINAL DECISION
# ============================================================

summary.append("=" * 80)
summary.append("FINAL DECISION")
summary.append("=" * 80)
summary.append("")

if (
    total_files == EXPECTED_FILES
    and
    total_epochs == EXPECTED_EPOCHS
    and
    fail_files == 0
    and
    error_files == 0
):

    final_status = (
        "PASS - V5 DATASET READY "
        "FOR PERTURBATION ANALYSIS"
    )

else:

    final_status = (
        "REVIEW - V5 DATASET "
        "REQUIRES INVESTIGATION"
    )

summary.append(
    final_status
)

summary.append("")

summary.append("=" * 80)
summary.append("IMPORTANT")
summary.append("=" * 80)
summary.append("")

summary.append(
    "READ-ONLY QC."
)

summary.append(
    "NO DATA WAS MODIFIED."
)

summary.append(
    "NO V5 FILE WAS MODIFIED."
)

summary.append(
    "NO V4 FILE WAS MODIFIED."
)

summary.append(
    "NO RAW DATA WAS MODIFIED."
)

# ============================================================
# WRITE SUMMARY
# ============================================================

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )

# ============================================================
# CONSOLE
# ============================================================

print()
print("=" * 80)
print("FINAL QC - PERTURBATION DATASET V5 COMPLETE")
print("=" * 80)

print()
print(f"Files found:     {total_files}")
print(f"Expected files:  {EXPECTED_FILES}")
print(f"PASS:             {pass_files}")
print(f"FAIL:             {fail_files}")
print(f"ERROR:            {error_files}")

print()
print(f"Total epochs:     {total_epochs}")
print(f"Expected epochs:  {EXPECTED_EPOCHS}")

print()
print("=" * 80)
print("FINAL DECISION")
print("=" * 80)

print(final_status)

print()
print("Saved:")
print(CSV_OUT)
print(SUMMARY_OUT)

print()
print("NO DATA WAS MODIFIED.")