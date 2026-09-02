import os
import glob
import numpy as np
import pandas as pd
import mne


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

INPUT_DIR = os.path.join(
    BASE_DIR,
    "epochs_clean",
    "logs",
    "perturbation_dataset_v5",
    "ELIGIBLE"
)

OUTPUT_DIR = os.path.join(
    INPUT_DIR,
    "final_perturbation_qc_v5"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_OUT = os.path.join(
    OUTPUT_DIR,
    "perturbation_final_qc_v5.csv"
)

SUMMARY_OUT = os.path.join(
    OUTPUT_DIR,
    "perturbation_final_qc_v5_summary.txt"
)


# ============================================================
# EXPECTED STANDARD
# ============================================================

EXPECTED_FILES = 82
EXPECTED_EPOCHS = 21816
EXPECTED_CHANNELS = 71
EXPECTED_SFREQ = 500.0
EXPECTED_TIMES = 501
EXPECTED_TMIN = -0.2
EXPECTED_TMAX = 0.8

EXPECTED_EVENT_IDS = {
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
# FILE DISCOVERY
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "*_harmonized_epo.fif"
        )
    )
)

print("=" * 80)
print("FINAL PERTURBATION DATASET QC V5")
print("=" * 80)

print()
print("Input directory:")
print(INPUT_DIR)

print()
print(f"Files found:      {len(files)}")
print(f"Expected files:   {EXPECTED_FILES}")

if len(files) != EXPECTED_FILES:
    print()
    print("WARNING:")
    print("Expected 82 harmonized files.")
    print("QC will continue, but final status will be REVIEW.")


# ============================================================
# RECORDS
# ============================================================

records = []

global_condition_counts = {}
global_event_id_counts = {}

total_epochs = 0


# ============================================================
# PROCESS FILES
# ============================================================

for idx, filepath in enumerate(files, 1):

    filename = os.path.basename(filepath)

    print()
    print("=" * 80)
    print(f"[{idx}/{len(files)}] {filename}")
    print("=" * 80)

    rec = {
        "file": filename,
        "path": filepath,
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
    }

    reasons = []

    try:

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        epochs = mne.read_epochs(
            filepath,
            preload=True,
            verbose=False
        )

        data = epochs.get_data()

        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        n_times = len(epochs.times)
        sfreq = float(epochs.info["sfreq"])

        tmin = float(epochs.tmin)
        tmax = float(epochs.tmax)

        rec["epochs"] = n_epochs
        rec["channels"] = n_channels
        rec["sfreq"] = sfreq
        rec["n_times"] = n_times
        rec["tmin"] = tmin
        rec["tmax"] = tmax

        total_epochs += n_epochs

        print(f"Epochs:        {n_epochs}")
        print(f"Channels:      {n_channels}")
        print(f"Sampling rate: {sfreq}")
        print(f"Samples:       {n_times}")
        print(
            f"Time range:    "
            f"{tmin:.12f} to {tmax:.12f}"
        )

        # ----------------------------------------------------
        # CHANNEL CHECK
        # ----------------------------------------------------

        if n_channels != EXPECTED_CHANNELS:
            reasons.append(
                f"CHANNELS_{n_channels}"
            )

        # ----------------------------------------------------
        # SAMPLING RATE
        # ----------------------------------------------------

        if not np.isclose(
            sfreq,
            EXPECTED_SFREQ,
            atol=1e-6
        ):
            reasons.append(
                f"SFREQ_{sfreq}"
            )

        # ----------------------------------------------------
        # TIMEPOINT CHECK
        # ----------------------------------------------------

        if n_times != EXPECTED_TIMES:
            reasons.append(
                f"TIMEPOINTS_{n_times}"
            )

        # ----------------------------------------------------
        # TIME RANGE
        # ----------------------------------------------------

        if not np.isclose(
            tmin,
            EXPECTED_TMIN,
            atol=1e-9
        ):
            reasons.append(
                f"TMIN_{tmin}"
            )

        if not np.isclose(
            tmax,
            EXPECTED_TMAX,
            atol=1e-9
        ):
            reasons.append(
                f"TMAX_{tmax}"
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

        if nan_count != 0:
            reasons.append(
                f"NAN_{nan_percent:.6f}%"
            )

        if inf_count != 0:
            reasons.append(
                f"INF_{inf_percent:.6f}%"
            )

        # ----------------------------------------------------
        # DATA RANGE
        # ----------------------------------------------------

        finite = data[np.isfinite(data)]

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

        # ----------------------------------------------------
        # EVENTS
        # ----------------------------------------------------

        if epochs.events is None:

            reasons.append("NO_EVENTS")

        else:

            event_ids = epochs.events[:, 2]

            rec["event_count"] = len(
                event_ids
            )

            rec["unique_event_ids"] = len(
                np.unique(event_ids)
            )

            if len(event_ids) != n_epochs:

                reasons.append(
                    f"EVENT_MISMATCH_"
                    f"{len(event_ids)}_{n_epochs}"
                )

            # ----------------------------------------------
            # EVENT COUNTS
            # ----------------------------------------------

            for event_id in np.unique(event_ids):

                count = int(
                    np.sum(
                        event_ids == event_id
                    )
                )

                global_event_id_counts[
                    int(event_id)
                ] = (
                    global_event_id_counts.get(
                        int(event_id),
                        0
                    )
                    + count
                )

            # ----------------------------------------------
            # CONDITION COUNTS
            # ----------------------------------------------

            inverse_mapping = {
                value: key
                for key, value
                in EXPECTED_EVENT_IDS.items()
            }

            for event_id in event_ids:

                event_id = int(event_id)

                condition = inverse_mapping.get(
                    event_id,
                    f"UNKNOWN_ID_{event_id}"
                )

                global_condition_counts[
                    condition
                ] = (
                    global_condition_counts.get(
                        condition,
                        0
                    )
                    + 1
                )

        # ----------------------------------------------------
        # ZERO EPOCHS
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

        # ----------------------------------------------------
        # CONSOLE
        # ----------------------------------------------------

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

        print(
            f"Events: {rec['event_count']} | "
            f"Unique IDs: "
            f"{rec['unique_event_ids']}"
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

        print()
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
# GLOBAL COUNTS
# ============================================================

status_counts = (
    df["status"]
    .value_counts()
)

pass_files = int(
    (df["status"] == "PASS").sum()
)

fail_files = int(
    (df["status"] == "FAIL").sum()
)

error_files = int(
    (df["status"] == "ERROR").sum()
)


# ============================================================
# EVENT DISTRIBUTION DATAFRAME
# ============================================================

event_rows = []

for event_id in sorted(
    global_event_id_counts
):

    condition = None

    for name, eid in EXPECTED_EVENT_IDS.items():

        if eid == event_id:

            condition = name
            break

    if condition is None:

        condition = (
            f"UNKNOWN_ID_{event_id}"
        )

    event_rows.append(
        {
            "event_id": event_id,
            "condition": condition,
            "count":
                global_event_id_counts[
                    event_id
                ]
        }
    )

event_df = pd.DataFrame(
    event_rows
)

event_csv = os.path.join(
    OUTPUT_DIR,
    "global_event_distribution_v5.csv"
)

event_df.to_csv(
    event_csv,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# CONDITION DISTRIBUTION
# ============================================================

condition_rows = []

for condition in sorted(
    global_condition_counts
):

    condition_rows.append(
        {
            "condition": condition,
            "count":
                global_condition_counts[
                    condition
                ]
        }
    )

condition_df = pd.DataFrame(
    condition_rows
)

condition_csv = os.path.join(
    OUTPUT_DIR,
    "global_condition_distribution_v5.csv"
)

condition_df.to_csv(
    condition_csv,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.append("=" * 80)
summary.append(
    "FINAL PERTURBATION DATASET QC V5 SUMMARY"
)
summary.append("=" * 80)
summary.append("")

summary.append(
    f"Files found:       {len(files)}"
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

summary.append("=" * 80)
summary.append("TARGET STANDARD")
summary.append("=" * 80)
summary.append("")

summary.append(
    f"Channels:           {EXPECTED_CHANNELS}"
)

summary.append(
    f"Sampling rate:      {EXPECTED_SFREQ} Hz"
)

summary.append(
    f"Time points:        {EXPECTED_TIMES}"
)

summary.append(
    f"TMIN:               {EXPECTED_TMIN}"
)

summary.append(
    f"TMAX:               {EXPECTED_TMAX}"
)

summary.append("")

summary.append("=" * 80)
summary.append("GLOBAL CONDITION COUNTS")
summary.append("=" * 80)
summary.append("")

for condition in sorted(
    global_condition_counts
):

    summary.append(
        f"{condition}: "
        f"{global_condition_counts[condition]}"
    )

summary.append("")

summary.append("=" * 80)
summary.append("GLOBAL EVENT ID COUNTS")
summary.append("=" * 80)
summary.append("")

for event_id in sorted(
    global_event_id_counts
):

    condition = None

    for name, eid in EXPECTED_EVENT_IDS.items():

        if eid == event_id:

            condition = name
            break

    if condition is None:

        condition = (
            f"UNKNOWN_ID_{event_id}"
        )

    summary.append(
        f"ID {event_id} | "
        f"{condition} | "
        f"{global_event_id_counts[event_id]}"
    )

summary.append("")

summary.append("=" * 80)
summary.append("FAILED / ERROR FILES")
summary.append("=" * 80)
summary.append("")

failed_df = df[
    df["status"] != "PASS"
]

if len(failed_df) == 0:

    summary.append("NONE")

else:

    for _, row in failed_df.iterrows():

        summary.append(
            f"{row['file']} | "
            f"{row['status']} | "
            f"{row['reasons']}"
        )

summary.append("")

summary.append("=" * 80)
summary.append("FINAL DECISION")
summary.append("=" * 80)
summary.append("")

if (
    len(files) == EXPECTED_FILES
    and total_epochs == EXPECTED_EPOCHS
    and pass_files == EXPECTED_FILES
    and fail_files == 0
    and error_files == 0
):

    final_status = (
        "PASS - DATASET READY "
        "FOR PERTURBATION ANALYSIS"
    )

else:

    final_status = (
        "REVIEW - DATASET "
        "REQUIRES INVESTIGATION"
    )

summary.append(final_status)

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
    "NO ORIGINAL V4/V5 FILE WAS MODIFIED."
)

summary.append(
    "NO RAW DATA WAS MODIFIED."
)

summary.append(
    "NO SET/FDT FILE WAS MODIFIED."
)

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )


# ============================================================
# FINAL CONSOLE
# ============================================================

print()
print("=" * 80)
print("FINAL PERTURBATION DATASET QC V5 COMPLETE")
print("=" * 80)

print()
print("STATUS COUNTS")
print(status_counts)

print()
print(
    f"Files found:      {len(files)}"
)

print(
    f"Expected files:   {EXPECTED_FILES}"
)

print(
    f"PASS:             {pass_files}"
)

print(
    f"FAIL:             {fail_files}"
)

print(
    f"ERROR:            {error_files}"
)

print()
print(
    f"Total epochs:     {total_epochs}"
)

print(
    f"Expected epochs:  {EXPECTED_EPOCHS}"
)

print()
print("=" * 80)
print("EVENT MAPPING")
print("=" * 80)

for name, event_id in (
    EXPECTED_EVENT_IDS.items()
):

    print(
        f"{name} = {event_id}"
    )

print()
print("=" * 80)
print("FINAL DECISION")
print("=" * 80)

print(final_status)

print()
print("Saved:")
print(CSV_OUT)
print(event_csv)
print(condition_csv)
print(SUMMARY_OUT)

print()
print("NO DATA WAS MODIFIED.")