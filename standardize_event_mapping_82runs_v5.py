# ============================================================
# STANDARDIZE EVENT MAPPING - 82 RUNS - V5
# ============================================================
#
# Purpose:
#   Convert run-specific MNE event IDs into one GLOBAL
#   canonical event mapping based on condition names.
#
# Input:
#   perturbation_dataset_v4\ELIGIBLE
#
# Output:
#   perturbation_dataset_v5\ELIGIBLE
#
# IMPORTANT:
#   Original V4 files are NOT modified.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import mne
import shutil

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

INPUT_DIR = (
    BASE_DIR
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v4"
    / "ELIGIBLE"
)

OUTPUT_DIR = (
    BASE_DIR
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v5"
    / "ELIGIBLE"
)

LOG_DIR = (
    BASE_DIR
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v5"
    / "logs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = LOG_DIR / "event_mapping_standardization_82runs_v5.csv"
SUMMARY_OUT = LOG_DIR / "event_mapping_standardization_82runs_v5_summary.txt"

# ============================================================
# EXPECTATIONS
# ============================================================

EXPECTED_FILES = 82
EXPECTED_EPOCHS = 21816

EXPECTED_CHANNELS = 71
EXPECTED_SFREQ = 500.0
EXPECTED_TIMES = 501
EXPECTED_TMIN = -0.2
EXPECTED_TMAX = 0.8

# ============================================================
# GLOBAL CANONICAL EVENT MAPPING
# ============================================================
#
# IMPORTANT:
# These IDs are newly assigned and are NOT assumed to match
# the original IDs inside individual runs.
#
# ============================================================

GLOBAL_EVENT_ID = {
    "left_click": 1,
    "right_click": 2,
    "show_cross": 3,
    "show_dash": 4,
    "show_letter": 5,
    "sound_beep": 6,
    "sound_buzz": 7,

    # Rare ambiguous condition found in the dataset
    "right_click/show_cross": 8,
}

# Reverse mapping
GLOBAL_ID_TO_CONDITION = {
    value: key
    for key, value in GLOBAL_EVENT_ID.items()
}

# ============================================================
# INPUT FILES
# ============================================================

input_files = sorted(
    INPUT_DIR.glob(
        "*_standardized_epo.fif"
    )
)

print("=" * 80)
print("EVENT MAPPING STANDARDIZATION V5")
print("=" * 80)

print()
print("Input directory:")
print(INPUT_DIR)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print(f"Files found: {len(input_files)}")
print(f"Expected:    {EXPECTED_FILES}")

# ============================================================
# CHECK INPUT FILE COUNT
# ============================================================

if len(input_files) != EXPECTED_FILES:

    raise RuntimeError(
        f"Expected {EXPECTED_FILES} input files "
        f"but found {len(input_files)}."
    )

# ============================================================
# RECORDS
# ============================================================

records = []

total_input_epochs = 0
total_output_epochs = 0

successful = 0
failed = 0

# ============================================================
# PROCESS FILES
# ============================================================

for idx, input_file in enumerate(
    input_files,
    start=1
):

    print()
    print("=" * 80)
    print(
        f"[{idx}/{len(input_files)}] "
        f"{input_file.name}"
    )
    print("=" * 80)

    rec = {
        "input_file": input_file.name,
        "output_file": "",
        "input_epochs": 0,
        "output_epochs": 0,
        "input_channels": 0,
        "output_channels": 0,
        "input_sfreq": np.nan,
        "output_sfreq": np.nan,
        "input_n_times": 0,
        "output_n_times": 0,
        "input_tmin": np.nan,
        "input_tmax": np.nan,
        "output_tmin": np.nan,
        "output_tmax": np.nan,
        "original_event_id": "",
        "mapped_conditions": "",
        "unknown_event_ids": "",
        "ambiguous_epochs": 0,
        "status": "",
        "reason": "",
    }

    try:

        # ====================================================
        # READ
        # ====================================================

        print("Reading epochs...")

        epochs = mne.read_epochs(
            input_file,
            preload=True,
            verbose=False
        )

        # ====================================================
        # INPUT INFORMATION
        # ====================================================

        n_epochs = len(epochs)
        n_channels = len(epochs.ch_names)
        n_times = len(epochs.times)
        sfreq = float(epochs.info["sfreq"])

        rec["input_epochs"] = n_epochs
        rec["input_channels"] = n_channels
        rec["input_sfreq"] = sfreq
        rec["input_n_times"] = n_times
        rec["input_tmin"] = float(epochs.tmin)
        rec["input_tmax"] = float(epochs.tmax)

        total_input_epochs += n_epochs

        print(f"Input epochs:     {n_epochs}")
        print(f"Input channels:   {n_channels}")
        print(f"Input sfreq:      {sfreq}")
        print(f"Input samples:    {n_times}")
        print(
            f"Input time:       "
            f"{epochs.tmin:.12f} to "
            f"{epochs.tmax:.12f}"
        )

        # ====================================================
        # BASIC STRUCTURE CHECK
        # ====================================================

        if n_channels != EXPECTED_CHANNELS:
            raise ValueError(
                f"CHANNELS_{n_channels}"
            )

        if n_times != EXPECTED_TIMES:
            raise ValueError(
                f"TIMEPOINTS_{n_times}"
            )

        if not np.isclose(
            sfreq,
            EXPECTED_SFREQ,
            atol=1e-3
        ):
            raise ValueError(
                f"SFREQ_{sfreq}"
            )

        if not np.isclose(
            epochs.tmin,
            EXPECTED_TMIN,
            atol=1e-6
        ):
            raise ValueError(
                f"TMIN_{epochs.tmin}"
            )

        if not np.isclose(
            epochs.tmax,
            EXPECTED_TMAX,
            atol=1e-6
        ):
            raise ValueError(
                f"TMAX_{epochs.tmax}"
            )

        # ====================================================
        # ORIGINAL EVENT ID
        # ====================================================

        if epochs.event_id is None:
            raise ValueError(
                "NO_EVENT_ID"
            )

        original_event_id = dict(
            epochs.event_id
        )

        rec["original_event_id"] = (
            "; ".join(
                f"{name}={code}"
                for name, code
                in sorted(
                    original_event_id.items(),
                    key=lambda x: x[1]
                )
            )
        )

        print()
        print("ORIGINAL EVENT MAPPING:")

        for name, code in sorted(
            original_event_id.items(),
            key=lambda x: x[1]
        ):

            print(
                f"  ID {code:<3} | {name}"
            )

        # ====================================================
        # REVERSE ORIGINAL MAPPING
        # ====================================================

        original_id_to_condition = {
            int(code): name
            for name, code
            in original_event_id.items()
        }

        # ====================================================
        # EVENTS
        # ====================================================

        if epochs.events is None:
            raise ValueError(
                "NO_EVENTS"
            )

        events = epochs.events.copy()

        # ====================================================
        # MAP EVERY EVENT TO GLOBAL CANONICAL ID
        # ====================================================

        new_event_codes = []

        unknown_ids = []
        unknown_conditions = []

        ambiguous_count = 0

        for event_code in events[:, 2]:

            event_code = int(event_code)

            if event_code not in original_id_to_condition:

                unknown_ids.append(
                    event_code
                )

                new_event_codes.append(
                    -1
                )

                continue

            condition = (
                original_id_to_condition[
                    event_code
                ]
            )

            # --------------------------------------------
            # CONDITION VALIDATION
            # --------------------------------------------

            if condition not in GLOBAL_EVENT_ID:

                unknown_conditions.append(
                    condition
                )

                new_event_codes.append(
                    -1
                )

                continue

            # --------------------------------------------
            # AMBIGUOUS CONDITION
            # --------------------------------------------

            if condition == "right_click/show_cross":

                ambiguous_count += 1

            # --------------------------------------------
            # GLOBAL ID
            # --------------------------------------------

            global_code = GLOBAL_EVENT_ID[
                condition
            ]

            new_event_codes.append(
                global_code
            )

        # ====================================================
        # UNKNOWN EVENT CHECK
        # ====================================================

        unknown_ids = sorted(
            set(unknown_ids)
        )

        unknown_conditions = sorted(
            set(unknown_conditions)
        )

        if unknown_ids:

            raise ValueError(
                "UNKNOWN_EVENT_IDS_" +
                ",".join(
                    map(str, unknown_ids)
                )
            )

        if unknown_conditions:

            raise ValueError(
                "UNKNOWN_CONDITIONS_" +
                ",".join(
                    unknown_conditions
                )
            )

        new_event_codes = np.asarray(
            new_event_codes,
            dtype=int
        )

        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if len(new_event_codes) != n_epochs:

            raise ValueError(
                "EVENT_COUNT_MISMATCH"
            )

        if np.any(
            new_event_codes < 1
        ):

            raise ValueError(
                "INVALID_NEW_EVENT_CODE"
            )

        # ====================================================
        # SHOW MAPPING
        # ====================================================

        mapped_conditions = sorted(
            set(
                original_id_to_condition.values()
            )
        )

        rec["mapped_conditions"] = (
            "; ".join(mapped_conditions)
        )

        rec["unknown_event_ids"] = (
            "; ".join(
                map(str, unknown_ids)
            )
        )

        rec["ambiguous_epochs"] = (
            ambiguous_count
        )

        print()
        print("GLOBAL STANDARD MAPPING:")

        for name, code in GLOBAL_EVENT_ID.items():

            print(
                f"  ID {code:<3} | {name}"
            )

        print()
        print(
            f"Ambiguous epochs in this file: "
            f"{ambiguous_count}"
        )

        # ====================================================
        # APPLY NEW EVENT CODES
        # ====================================================

        print()
        print(
            "Applying canonical event IDs..."
        )

        epochs.events[:, 2] = (
            new_event_codes
        )

        # ====================================================
        # APPLY GLOBAL EVENT_ID
        # ====================================================

        epochs.event_id = dict(
            GLOBAL_EVENT_ID
        )

        # ====================================================
        # VERIFY AFTER MAPPING
        # ====================================================

        verify_codes = (
            epochs.events[:, 2]
        )

        if len(
            verify_codes
        ) != n_epochs:

            raise ValueError(
                "VERIFY_EVENT_COUNT_FAILED"
            )

        if not set(
            np.unique(verify_codes)
        ).issubset(
            set(GLOBAL_EVENT_ID.values())
        ):

            raise ValueError(
                "VERIFY_EVENT_CODES_FAILED"
            )

        # ====================================================
        # DATA CHECK
        # ====================================================

        data = epochs.get_data()

        if np.isnan(data).any():

            raise ValueError(
                "NAN_DATA"
            )

        if np.isinf(data).any():

            raise ValueError(
                "INF_DATA"
            )

        # ====================================================
        # OUTPUT FILE
        # ====================================================

        output_name = (
            input_file.stem
            .replace(
                "_standardized_epo",
                "_harmonized_epo"
            )
            + ".fif"
        )

        output_file = (
            OUTPUT_DIR
            / output_name
        )

        rec["output_file"] = (
            output_file.name
        )

        # ====================================================
        # SAVE
        # ====================================================

        print()
        print("Saving:")
        print(output_file)

        epochs.save(
            output_file,
            overwrite=True,
            verbose=False
        )

        # ====================================================
        # RELOAD VALIDATION
        # ====================================================

        print()
        print("Reloading saved file...")

        check = mne.read_epochs(
            output_file,
            preload=True,
            verbose=False
        )

        # ====================================================
        # OUTPUT INFORMATION
        # ====================================================

        out_epochs = len(check)
        out_channels = len(
            check.ch_names
        )
        out_times = len(
            check.times
        )
        out_sfreq = float(
            check.info["sfreq"]
        )

        rec["output_epochs"] = out_epochs
        rec["output_channels"] = out_channels
        rec["output_sfreq"] = out_sfreq
        rec["output_n_times"] = out_times
        rec["output_tmin"] = float(
            check.tmin
        )
        rec["output_tmax"] = float(
            check.tmax
        )

        total_output_epochs += out_epochs

        # ====================================================
        # VALIDATION
        # ====================================================

        validation_errors = []

        if out_epochs != n_epochs:
            validation_errors.append(
                "EPOCH_COUNT_CHANGED"
            )

        if out_channels != EXPECTED_CHANNELS:
            validation_errors.append(
                "CHANNEL_COUNT_CHANGED"
            )

        if out_times != EXPECTED_TIMES:
            validation_errors.append(
                "TIMEPOINT_COUNT_CHANGED"
            )

        if not np.isclose(
            out_sfreq,
            EXPECTED_SFREQ,
            atol=1e-3
        ):
            validation_errors.append(
                "SFREQ_CHANGED"
            )

        if not np.isclose(
            check.tmin,
            EXPECTED_TMIN,
            atol=1e-6
        ):
            validation_errors.append(
                "TMIN_CHANGED"
            )

        if not np.isclose(
            check.tmax,
            EXPECTED_TMAX,
            atol=1e-6
        ):
            validation_errors.append(
                "TMAX_CHANGED"
            )

        # ====================================================
        # EVENT VALIDATION
        # ====================================================

        if check.event_id != GLOBAL_EVENT_ID:

            validation_errors.append(
                "GLOBAL_EVENT_ID_MISMATCH"
            )

        saved_event_codes = (
            check.events[:, 2]
        )

        if not set(
            np.unique(
                saved_event_codes
            )
        ).issubset(
            set(GLOBAL_EVENT_ID.values())
        ):

            validation_errors.append(
                "INVALID_SAVED_EVENT_CODES"
            )

        # ====================================================
        # DATA VALIDATION
        # ====================================================

        check_data = check.get_data()

        if np.isnan(
            check_data
        ).any():

            validation_errors.append(
                "NAN_AFTER_SAVE"
            )

        if np.isinf(
            check_data
        ).any():

            validation_errors.append(
                "INF_AFTER_SAVE"
            )

        # ====================================================
        # FINAL FILE STATUS
        # ====================================================

        if validation_errors:

            rec["status"] = "FAILED"

            rec["reason"] = ";".join(
                validation_errors
            )

            failed += 1

            print()
            print(
                "STATUS: FAILED"
            )

            print(
                "REASONS:",
                rec["reason"]
            )

        else:

            rec["status"] = "SUCCESS"

            rec["reason"] = "NONE"

            successful += 1

            print()
            print(
                "VALIDATION"
            )

            print(
                "-" * 70
            )

            print(
                f"Saved epochs:     {out_epochs}"
            )

            print(
                f"Saved channels:   {out_channels}"
            )

            print(
                f"Saved sfreq:      {out_sfreq}"
            )

            print(
                f"Saved samples:    {out_times}"
            )

            print(
                f"Saved time range: "
                f"{check.tmin:.12f} to "
                f"{check.tmax:.12f}"
            )

            print(
                f"Event IDs:        "
                f"{check.event_id}"
            )

            print(
                "STATUS: SUCCESS"
            )

    except Exception as e:

        failed += 1

        rec["status"] = "FAILED"

        rec["reason"] = (
            str(e)
            .replace(";", ",")
            [:1000]
        )

        print()
        print("STATUS: FAILED")
        print("REASON:")
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
# SUMMARY
# ============================================================

summary = []

summary.append("=" * 80)
summary.append(
    "EVENT MAPPING STANDARDIZATION V5 SUMMARY"
)
summary.append("=" * 80)
summary.append("")

summary.append(
    f"Input files:       {len(input_files)}"
)

summary.append(
    f"Expected files:    {EXPECTED_FILES}"
)

summary.append(
    f"Successful:        {successful}"
)

summary.append(
    f"Failed:            {failed}"
)

summary.append(
    f"Input epochs:      {total_input_epochs}"
)

summary.append(
    f"Output epochs:     {total_output_epochs}"
)

summary.append("")

summary.append("=" * 80)
summary.append(
    "GLOBAL CANONICAL EVENT MAPPING"
)
summary.append("=" * 80)
summary.append("")

for condition, code in GLOBAL_EVENT_ID.items():

    summary.append(
        f"{code} = {condition}"
    )

summary.append("")

summary.append("=" * 80)
summary.append(
    "EXPECTED DATASET"
)
summary.append("=" * 80)
summary.append("")

summary.append(
    f"Expected files:    {EXPECTED_FILES}"
)

summary.append(
    f"Expected epochs:   {EXPECTED_EPOCHS}"
)

summary.append("")

summary.append("=" * 80)
summary.append(
    "FINAL DECISION"
)
summary.append("=" * 80)
summary.append("")

if (
    len(input_files) == EXPECTED_FILES
    and successful == EXPECTED_FILES
    and failed == 0
    and total_input_epochs == EXPECTED_EPOCHS
    and total_output_epochs == EXPECTED_EPOCHS
):

    final_status = (
        "PASS - EVENT MAPPING "
        "SUCCESSFULLY HARMONIZED"
    )

else:

    final_status = (
        "REVIEW - EVENT MAPPING "
        "REQUIRES INVESTIGATION"
    )

summary.append(final_status)

summary.append("")

summary.append("=" * 80)
summary.append(
    "IMPORTANT"
)
summary.append("=" * 80)
summary.append("")

summary.append(
    "Original V4 files were NOT modified."
)

summary.append(
    "Original EEG data was NOT modified."
)

summary.append(
    "Only event codes were harmonized in new files."
)

summary.append(
    "Condition names were used to determine "
    "the canonical event ID."
)

summary.append(
    "Numeric event IDs from individual runs "
    "were NOT assumed to be globally consistent."
)

summary.append(
    "right_click/show_cross was retained as "
    "a separate ambiguous condition (ID 8)."
)

summary.append(
    "Ambiguous epochs should be excluded from "
    "condition-specific inferential analyses unless "
    "their original meaning is established."
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
print(
    "EVENT MAPPING STANDARDIZATION V5 COMPLETE"
)
print("=" * 80)

print()
print(
    f"Input files:       {len(input_files)}"
)

print(
    f"Successful:        {successful}"
)

print(
    f"Failed:            {failed}"
)

print(
    f"Input epochs:      {total_input_epochs}"
)

print(
    f"Output epochs:     {total_output_epochs}"
)

print()
print(
    "GLOBAL EVENT MAPPING"
)

for condition, code in GLOBAL_EVENT_ID.items():

    print(
        f"ID {code} = {condition}"
    )

print()
print(
    "FINAL STATUS:"
)

print(final_status)

print()
print("Saved:")
print(OUTPUT_DIR)
print(CSV_OUT)
print(SUMMARY_OUT)

print()
print(
    "ORIGINAL V4 FILES WERE NOT MODIFIED."
)