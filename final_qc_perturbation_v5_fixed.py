# ============================================================
# FINAL QC - PERTURBATION DATASET V5
# FIXED VERSION
#
# Purpose:
#   Validate the 82 harmonized V5 EEG epoch files.
#
# IMPORTANT:
#   - READ ONLY
#   - NO DATA MODIFICATION
#   - NO RESAMPLING
#   - NO CROPPING
#   - NO BASELINE CORRECTION
#   - NO EVENT MODIFICATION
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import mne


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

INPUT_DIR = (
    BASE_DIR
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v5"
    / "ELIGIBLE"
)

OUTPUT_DIR = (
    INPUT_DIR
    / "final_qc_v5_fixed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_OUT = (
    OUTPUT_DIR
    / "final_qc_perturbation_v5_fixed.csv"
)

SUMMARY_OUT = (
    OUTPUT_DIR
    / "final_qc_perturbation_v5_fixed_summary.txt"
)


# ============================================================
# EXPECTED DATASET
# ============================================================

EXPECTED_FILES = 82
EXPECTED_EPOCHS = 21816

EXPECTED_CHANNELS = 71
EXPECTED_SFREQ = 500.0

EXPECTED_TMIN = -0.2
EXPECTED_TMAX = 0.8

EXPECTED_TIMES = 501


# ============================================================
# CANONICAL EVENT MAPPING
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


EXPECTED_EVENT_REVERSE = {
    value: key
    for key, value in EXPECTED_EVENT_ID.items()
}


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("FINAL QC - PERTURBATION DATASET V5 FIXED")
print("=" * 80)

print()
print("Input directory:")
print(INPUT_DIR)

print()


# ============================================================
# CHECK INPUT DIRECTORY
# ============================================================

if not INPUT_DIR.exists():

    print("ERROR:")
    print("Input directory does not exist.")

    print()
    print("Expected:")
    print(INPUT_DIR)

    raise SystemExit(1)


# ============================================================
# FIND HARMONIZED FILES
# ============================================================

files = sorted(
    INPUT_DIR.glob(
        "*_harmonized_epo.fif"
    )
)


# ============================================================
# FILE DISCOVERY
# ============================================================

print(
    f"Files found: {len(files)}"
)

print(
    f"Expected:    {EXPECTED_FILES}"
)

print()

if len(files) == 0:

    print("=" * 80)
    print("ERROR - NO HARMONIZED FILES FOUND")
    print("=" * 80)

    print()
    print("Files currently present:")

    all_fif = sorted(
        INPUT_DIR.glob("*.fif")
    )

    if len(all_fif) == 0:

        print("  No FIF files found.")

    else:

        for f in all_fif:

            print(
                " ",
                f.name
            )

    raise SystemExit(1)


# ============================================================
# RECORDS
# ============================================================

records = []


# ============================================================
# PROCESS FILES
# ============================================================

for index, file_path in enumerate(
    files,
    start=1
):

    print()
    print("=" * 80)

    print(
        f"[{index}/{len(files)}] "
        f"{file_path.name}"
    )

    print("=" * 80)

    rec = {

        "file": file_path.name,

        "epochs": np.nan,

        "channels": np.nan,

        "samples": np.nan,

        "sfreq": np.nan,

        "tmin": np.nan,

        "tmax": np.nan,

        "nan_percent": np.nan,

        "inf_percent": np.nan,

        "global_min": np.nan,

        "global_max": np.nan,

        "global_std": np.nan,

        "max_abs": np.nan,

        "event_count": np.nan,

        "unique_event_ids": np.nan,

        "event_mapping": "",

        "channel_check": "FAIL",

        "timepoint_check": "FAIL",

        "sfreq_check": "FAIL",

        "time_range_check": "FAIL",

        "nan_check": "FAIL",

        "inf_check": "FAIL",

        "event_check": "FAIL",

        "status": "ERROR",

        "reasons": ""

    }

    reasons = []


    # ========================================================
    # READ FILE
    # ========================================================

    try:

        print(
            "Reading epochs..."
        )

        epochs = mne.read_epochs(
            file_path,
            preload=True,
            verbose=False
        )


        # ====================================================
        # BASIC PROPERTIES
        # ====================================================

        data = epochs.get_data(
            copy=False
        )

        n_epochs = len(epochs)

        n_channels = len(
            epochs.ch_names
        )

        n_times = len(
            epochs.times
        )

        sfreq = float(
            epochs.info["sfreq"]
        )

        tmin = float(
            epochs.times[0]
        )

        tmax = float(
            epochs.times[-1]
        )


        rec["epochs"] = n_epochs
        rec["channels"] = n_channels
        rec["samples"] = n_times
        rec["sfreq"] = sfreq
        rec["tmin"] = tmin
        rec["tmax"] = tmax


        print(
            f"Epochs:        {n_epochs}"
        )

        print(
            f"Channels:      {n_channels}"
        )

        print(
            f"Samples:       {n_times}"
        )

        print(
            f"Sampling rate: {sfreq}"
        )

        print(
            f"Time range:    "
            f"{tmin:.12f} to "
            f"{tmax:.12f}"
        )


        # ====================================================
        # CHANNEL CHECK
        # ====================================================

        if n_channels == EXPECTED_CHANNELS:

            rec["channel_check"] = "PASS"

        else:

            reasons.append(
                f"CHANNELS_{n_channels}"
            )


        # ====================================================
        # TIMEPOINT CHECK
        # ====================================================

        if n_times == EXPECTED_TIMES:

            rec["timepoint_check"] = "PASS"

        else:

            reasons.append(
                f"TIMEPOINTS_{n_times}"
            )


        # ====================================================
        # SAMPLING RATE CHECK
        # ====================================================

        if np.isclose(
            sfreq,
            EXPECTED_SFREQ,
            rtol=0,
            atol=1e-6
        ):

            rec["sfreq_check"] = "PASS"

        else:

            reasons.append(
                f"SFREQ_{sfreq}"
            )


        # ====================================================
        # TIME RANGE CHECK
        # ====================================================

        if (
            np.isclose(
                tmin,
                EXPECTED_TMIN,
                rtol=0,
                atol=1e-9
            )
            and
            np.isclose(
                tmax,
                EXPECTED_TMAX,
                rtol=0,
                atol=1e-9
            )
        ):

            rec["time_range_check"] = "PASS"

        else:

            reasons.append(
                f"TIME_RANGE_{tmin}_{tmax}"
            )


        # ====================================================
        # NAN / INF CHECK
        # ====================================================

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

            reasons.append(
                f"NAN_{nan_percent:.6f}%"
            )


        if inf_count == 0:

            rec["inf_check"] = "PASS"

        else:

            reasons.append(
                f"INF_{inf_percent:.6f}%"
            )


        # ====================================================
        # DATA RANGE
        # ====================================================

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
                np.max(
                    np.abs(finite)
                )
            )

        else:

            reasons.append(
                "NO_FINITE_DATA"
            )


        # ====================================================
        # EVENT CHECK
        # ====================================================

        if epochs.events is None:

            reasons.append(
                "NO_EVENTS"
            )

        else:

            event_ids = epochs.events[:, 2]

            event_count = len(
                event_ids
            )

            unique_ids = sorted(
                np.unique(
                    event_ids
                ).astype(int).tolist()
            )

            rec["event_count"] = event_count

            rec["unique_event_ids"] = len(
                unique_ids
            )


            if event_count == n_epochs:

                event_count_ok = True

            else:

                event_count_ok = False

                reasons.append(
                    f"EVENT_MISMATCH_"
                    f"{event_count}_"
                    f"{n_epochs}"
                )


            # =================================================
            # EVENT MAPPING CHECK
            # =================================================

            mapping_ok = True

            current_mapping = (
                epochs.event_id
            )

            mapping_parts = []

            for name, event_id in sorted(
                current_mapping.items(),
                key=lambda x: x[1]
            ):

                mapping_parts.append(
                    f"{name}={event_id}"
                )

                expected_name = (
                    EXPECTED_EVENT_REVERSE.get(
                        int(event_id)
                    )
                )

                if expected_name is None:

                    mapping_ok = False

                    reasons.append(
                        f"UNKNOWN_EVENT_ID_{event_id}"
                    )

                elif expected_name != name:

                    mapping_ok = False

                    reasons.append(
                        f"EVENT_MAPPING_"
                        f"{name}_{event_id}"
                    )


            rec["event_mapping"] = ";".join(
                mapping_parts
            )


            # =================================================
            # CHECK ACTUAL EVENT IDs
            # =================================================

            for event_id in unique_ids:

                if event_id not in (
                    EXPECTED_EVENT_REVERSE
                ):

                    mapping_ok = False

                    reasons.append(
                        f"UNKNOWN_USED_EVENT_ID_"
                        f"{event_id}"
                    )


            if (
                event_count_ok
                and mapping_ok
            ):

                rec["event_check"] = "PASS"


        # ====================================================
        # ZERO EPOCH CHECK
        # ====================================================

        if n_epochs <= 0:

            reasons.append(
                "ZERO_EPOCHS"
            )


        # ====================================================
        # PRINT QC VALUES
        # ====================================================

        print()

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
            f"Events: "
            f"{rec['event_count']} | "
            f"Unique IDs: "
            f"{rec['unique_event_ids']}"
        )

        print(
            "Event mapping:"
        )

        if rec["event_mapping"]:

            for item in (
                rec["event_mapping"]
                .split(";")
            ):

                print(
                    f"  {item}"
                )


        # ====================================================
        # FINAL FILE STATUS
        # ====================================================

        if len(reasons) == 0:

            rec["status"] = "PASS"

            rec["reasons"] = "NONE"

        else:

            rec["status"] = "FAIL"

            rec["reasons"] = ";".join(
                reasons
            )


        print()

        print(
            f"STATUS: {rec['status']}"
        )

        if rec["reasons"] != "NONE":

            print(
                f"REASONS: "
                f"{rec['reasons']}"
            )


    # ========================================================
    # EXCEPTION
    # ========================================================

    except Exception as e:

        rec["status"] = "ERROR"

        rec["reasons"] = (
            "EXCEPTION_"
            + str(e)
            .replace(";", ",")
            [:500]
        )

        print()

        print(
            "ERROR:"
        )

        print(
            e
        )


    records.append(
        rec
    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    records
)


# ============================================================
# SAVE CSV
# ============================================================

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
    (
        df["status"]
        == "PASS"
    ).sum()
)

fail_files = int(
    (
        df["status"]
        == "FAIL"
    ).sum()
)

error_files = int(
    (
        df["status"]
        == "ERROR"
    ).sum()
)

total_epochs = int(
    df["epochs"]
    .fillna(0)
    .sum()
)


# ============================================================
# EVENT ID DISTRIBUTION
# ============================================================

all_used_event_ids = []

for file_path in files:

    try:

        epochs = mne.read_epochs(
            file_path,
            preload=False,
            verbose=False
        )

        if epochs.events is not None:

            ids = (
                epochs.events[:, 2]
                .astype(int)
                .tolist()
            )

            all_used_event_ids.extend(
                ids
            )

    except Exception:

        pass


unique_global_event_ids = sorted(
    set(
        all_used_event_ids
    )
)


# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.append(
    "=" * 80
)

summary.append(
    "FINAL QC - PERTURBATION DATASET V5 FIXED"
)

summary.append(
    "=" * 80
)

summary.append("")

summary.append(
    f"Input directory:"
)

summary.append(
    str(INPUT_DIR)
)

summary.append("")

summary.append(
    f"Files found:      {total_files}"
)

summary.append(
    f"Expected files:   {EXPECTED_FILES}"
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

summary.append("")

summary.append(
    f"Total epochs:      {total_epochs}"
)

summary.append(
    f"Expected epochs:   {EXPECTED_EPOCHS}"
)

summary.append("")


# ============================================================
# EVENT IDS
# ============================================================

summary.append(
    "=" * 80
)

summary.append(
    "GLOBAL EVENT IDS USED"
)

summary.append(
    "=" * 80
)

summary.append("")

if unique_global_event_ids:

    for event_id in (
        unique_global_event_ids
    ):

        name = (
            EXPECTED_EVENT_REVERSE.get(
                event_id,
                "UNKNOWN"
            )
        )

        count = (
            all_used_event_ids
            .count(event_id)
        )

        summary.append(
            f"ID {event_id} | "
            f"{name} | "
            f"{count} epochs"
        )

else:

    summary.append(
        "NO EVENT IDS FOUND"
    )


summary.append("")


# ============================================================
# CHECK RESULTS
# ============================================================

summary.append(
    "=" * 80
)

summary.append(
    "CHECK RESULTS"
)

summary.append(
    "=" * 80
)

summary.append("")

for col in [

    "channel_check",

    "timepoint_check",

    "sfreq_check",

    "time_range_check",

    "nan_check",

    "inf_check",

    "event_check"

]:

    summary.append(
        col
    )

    counts = (
        df[col]
        .value_counts()
    )

    for key, value in (
        counts.items()
    ):

        summary.append(
            f"  {key}: {value}"
        )

    summary.append("")


# ============================================================
# FAILED FILES
# ============================================================

summary.append(
    "=" * 80
)

summary.append(
    "FAILED / ERROR FILES"
)

summary.append(
    "=" * 80
)

summary.append("")

failed = df[
    df["status"] != "PASS"
]

if len(failed) == 0:

    summary.append(
        "NONE"
    )

else:

    for _, row in (
        failed.iterrows()
    ):

        summary.append(
            f"{row['file']} | "
            f"{row['status']} | "
            f"{row['reasons']}"
        )

summary.append("")


# ============================================================
# FINAL DECISION
# ============================================================

summary.append(
    "=" * 80
)

summary.append(
    "FINAL DECISION"
)

summary.append(
    "=" * 80
)

summary.append("")


if (

    total_files == EXPECTED_FILES

    and
    total_epochs == EXPECTED_EPOCHS

    and
    pass_files == EXPECTED_FILES

    and
    fail_files == 0

    and
    error_files == 0

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


summary.append(
    final_status
)

summary.append("")


# ============================================================
# IMPORTANT
# ============================================================

summary.append(
    "=" * 80
)

summary.append(
    "IMPORTANT"
)

summary.append(
    "=" * 80
)

summary.append("")

summary.append(
    "READ-ONLY QC."
)

summary.append(
    "NO DATA WAS MODIFIED."
)

summary.append(
    "NO EPOCH FILE WAS MODIFIED."
)

summary.append(
    "NO RAW DATA WAS MODIFIED."
)

summary.append(
    "NO SET FILE WAS MODIFIED."
)

summary.append(
    "NO FDT FILE WAS MODIFIED."
)

summary.append(
    "NO RESAMPLING WAS PERFORMED."
)

summary.append(
    "NO CROPPING WAS PERFORMED."
)

summary.append(
    "NO BASELINE CORRECTION WAS PERFORMED."
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
        "\n".join(
            summary
        )
    )


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print()

print(
    "=" * 80
)

print(
    "FINAL QC - PERTURBATION DATASET V5 FIXED COMPLETE"
)

print(
    "=" * 80
)

print()

print(
    "STATUS COUNTS"
)

print(
    df["status"]
    .value_counts()
)

print()

print(
    f"Files found:      {total_files}"
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

print(
    "=" * 80
)

print(
    "FINAL DECISION"
)

print(
    "=" * 80
)

print(
    final_status
)

print()

print(
    "Saved:"
)

print(
    CSV_OUT
)

print(
    SUMMARY_OUT
)

print()

print(
    "NO DATA WAS MODIFIED."
)

print(
    "=" * 80
)