import os
import glob
import re
import numpy as np
import pandas as pd
import mne

# ============================================================
# BUILD PERTURBATION MASTER DATASET
# 82 RUNS
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

EPOCH_DIR = os.path.join(BASE_DIR, "epochs_clean")

SELECTION_FILE = os.path.join(
    EPOCH_DIR,
    "logs",
    "final_selection",
    "epoch_final_selection_82runs.csv"
)

RECOMMENDATION_FILE = os.path.join(
    EPOCH_DIR,
    "logs",
    "final_selection",
    "epoch_keep_recommendations_82runs.csv"
)

BALANCE_FILE = os.path.join(
    EPOCH_DIR,
    "logs",
    "condition_harmonization",
    "condition_harmonization_qc_82runs.csv"
)

OUT_DIR = os.path.join(
    EPOCH_DIR,
    "logs",
    "perturbation_master"
)

os.makedirs(OUT_DIR, exist_ok=True)

MASTER_CSV = os.path.join(
    OUT_DIR,
    "perturbation_master_82runs.csv"
)

SUMMARY_TXT = os.path.join(
    OUT_DIR,
    "perturbation_master_82runs_summary.txt"
)

# ------------------------------------------------------------
# LOAD QC TABLES
# ------------------------------------------------------------

print("=" * 80)
print("BUILDING PERTURBATION MASTER DATASET")
print("=" * 80)

print()
print("Loading QC tables...")

selection = pd.read_csv(SELECTION_FILE)
recommendation = pd.read_csv(RECOMMENDATION_FILE)
balance = pd.read_csv(BALANCE_FILE)

print(f"Selection records:       {len(selection)}")
print(f"Recommendation records:  {len(recommendation)}")
print(f"Balance records:         {len(balance)}")

print()
print("Selection columns:")
print(list(selection.columns))

print()
print("Recommendation columns:")
print(list(recommendation.columns))

# ------------------------------------------------------------
# DISCOVER EPOCH FILES
# ------------------------------------------------------------

files = sorted(
    glob.glob(
        os.path.join(
            EPOCH_DIR,
            "*_clean_epo.fif"
        )
    )
)

print()
print(f"Epoch files discovered: {len(files)}")

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def get_subject_run(filename):

    subject_match = re.search(
        r"(sub-\d+)",
        filename
    )

    run_match = re.search(
        r"run-(\d+)",
        filename
    )

    subject = (
        subject_match.group(1)
        if subject_match
        else None
    )

    run = (
        int(run_match.group(1))
        if run_match
        else None
    )

    return subject, run


def normalize_condition(x):

    if pd.isna(x):
        return None

    x = str(x)

    # Handle numpy/string representations
    x = x.strip()

    replacements = {
        "np.str_('left_click')": "left_click",
        "np.str_('right_click')": "right_click",
        "np.str_('show_cross')": "show_cross",
        "np.str_('show_dash')": "show_dash",
        "np.str_('show_letter')": "show_letter",
        "np.str_('sound_beep')": "sound_beep",
        "np.str_('sound_buzz')": "sound_buzz",
    }

    return replacements.get(x, x)


# ------------------------------------------------------------
# PREPARE LOOKUP TABLES
# ------------------------------------------------------------

selection_by_file = {}

if "file" in selection.columns:

    for _, row in selection.iterrows():

        selection_by_file[
            str(row["file"])
        ] = row.to_dict()


recommendation_by_file_epoch = {}

if (
    "file" in recommendation.columns
    and "epoch" in recommendation.columns
):

    for _, row in recommendation.iterrows():

        key = (
            str(row["file"]),
            int(row["epoch"])
        )

        recommendation_by_file_epoch[key] = row.to_dict()

# ------------------------------------------------------------
# BUILD MASTER
# ------------------------------------------------------------

records = []

total_epochs = 0

for file_index, filepath in enumerate(files, 1):

    filename = os.path.basename(filepath)

    subject, run = get_subject_run(filename)

    print()
    print("=" * 80)
    print(
        f"[{file_index}/{len(files)}] "
        f"{filename}"
    )
    print("=" * 80)

    try:

        epochs = mne.read_epochs(
            filepath,
            preload=True,
            verbose=False
        )

        n_epochs = len(epochs)

        print(
            f"Epochs: {n_epochs}"
        )

        total_epochs += n_epochs

        # ----------------------------------------------------
        # EVENT ID
        # ----------------------------------------------------

        inverse_event_id = {}

        if epochs.event_id:

            inverse_event_id = {
                int(v): str(k)
                for k, v in epochs.event_id.items()
            }

        # ----------------------------------------------------
        # EXTRACT CONDITIONS
        # ----------------------------------------------------

        epoch_conditions = []

        for event_code in epochs.events[:, 2]:

            condition = inverse_event_id.get(
                int(event_code),
                f"UNKNOWN_{int(event_code)}"
            )

            condition = normalize_condition(
                condition
            )

            epoch_conditions.append(
                condition
            )

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        data = epochs.get_data(
            copy=False
        )

        # ----------------------------------------------------
        # FILE LEVEL SELECTION
        # ----------------------------------------------------

        file_selection = selection_by_file.get(
            filename,
            {}
        )

        # ----------------------------------------------------
        # EPOCH LOOP
        # ----------------------------------------------------

        for i in range(n_epochs):

            epoch_number = i + 1

            condition = epoch_conditions[i]

            epoch_data = data[i]

            max_abs = float(
                np.max(
                    np.abs(epoch_data)
                )
            )

            std = float(
                np.std(epoch_data)
            )

            nan_percent = float(
                np.mean(
                    np.isnan(epoch_data)
                ) * 100
            )

            inf_percent = float(
                np.mean(
                    np.isinf(epoch_data)
                ) * 100
            )

            high_amp_percent = float(
                np.mean(
                    np.abs(epoch_data) > 200
                ) * 100
            )

            bad_amp_percent = float(
                np.mean(
                    np.abs(epoch_data) > 300
                ) * 100
            )

            # ------------------------------------------------
            # RECOMMENDATION
            # ------------------------------------------------

            rec = recommendation_by_file_epoch.get(
                (
                    filename,
                    epoch_number
                ),
                {}
            )

            recommendation_value = rec.get(
                "recommendation",
                "UNKNOWN"
            )

            artifact_type = rec.get(
                "artifact_type",
                ""
            )

            diagnostic_class = rec.get(
                "diagnostic_class",
                ""
            )

            # ------------------------------------------------
            # FINAL PERTURBATION ELIGIBILITY
            # ------------------------------------------------

            if recommendation_value == "KEEP":

                perturbation_status = "ELIGIBLE"

            elif recommendation_value == "KEEP_REVIEW":

                perturbation_status = "ELIGIBLE_REVIEW"

            elif recommendation_value == "EXCLUDE_RECOMMENDED":

                perturbation_status = "EXCLUDE"

            else:

                perturbation_status = "REVIEW"

            # ------------------------------------------------
            # RECORD
            # ------------------------------------------------

            records.append({

                "subject": subject,
                "run": run,
                "file": filename,
                "epoch": epoch_number,

                "condition": condition,

                "n_epochs_in_run": n_epochs,

                "max_abs": max_abs,
                "global_std": std,

                "nan_percent": nan_percent,
                "inf_percent": inf_percent,

                "high_amplitude_percent":
                    high_amp_percent,

                "bad_amplitude_percent":
                    bad_amp_percent,

                "recommendation":
                    recommendation_value,

                "artifact_type":
                    artifact_type,

                "diagnostic_class":
                    diagnostic_class,

                "perturbation_status":
                    perturbation_status,

            })

        print(
            f"Master records added: {n_epochs}"
        )

    except Exception as e:

        print()
        print("ERROR:")
        print(str(e))

# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)

# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

df = df.sort_values(
    [
        "subject",
        "run",
        "epoch"
    ]
).reset_index(
    drop=True
)

# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 80)
print("MASTER DATASET VALIDATION")
print("=" * 80)

print()
print(
    f"Total master records: {len(df)}"
)

print(
    f"Expected epochs from files: {total_epochs}"
)

if len(df) != total_epochs:

    print()
    print(
        "WARNING: RECORD COUNT MISMATCH"
    )

else:

    print(
        "Record count: PASS"
    )

print()
print("PERTURBATION STATUS")

print(
    df["perturbation_status"]
    .value_counts()
)

print()
print("RECOMMENDATION")

print(
    df["recommendation"]
    .value_counts()
)

print()
print("CONDITIONS")

print(
    df["condition"]
    .value_counts()
)

# ============================================================
# SUBJECT SUMMARY
# ============================================================

subject_summary = (
    df.groupby(
        "subject"
    )
    .agg(
        total_epochs=(
            "epoch",
            "count"
        ),
        eligible=(
            "perturbation_status",
            lambda x: np.sum(
                x == "ELIGIBLE"
            )
        ),
        eligible_review=(
            "perturbation_status",
            lambda x: np.sum(
                x == "ELIGIBLE_REVIEW"
            )
        ),
        excluded=(
            "perturbation_status",
            lambda x: np.sum(
                x == "EXCLUDE"
            )
        ),
    )
    .reset_index()
)

subject_summary_file = os.path.join(
    OUT_DIR,
    "perturbation_subject_summary_82runs.csv"
)

subject_summary.to_csv(
    subject_summary_file,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# CONDITION SUMMARY
# ============================================================

condition_summary = (
    df.groupby(
        [
            "condition",
            "perturbation_status"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)

condition_summary_file = os.path.join(
    OUT_DIR,
    "perturbation_condition_summary_82runs.csv"
)

condition_summary.to_csv(
    condition_summary_file,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SAVE MASTER
# ============================================================

df.to_csv(
    MASTER_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

eligible = int(
    np.sum(
        df["perturbation_status"]
        == "ELIGIBLE"
    )
)

eligible_review = int(
    np.sum(
        df["perturbation_status"]
        == "ELIGIBLE_REVIEW"
    )
)

excluded = int(
    np.sum(
        df["perturbation_status"]
        == "EXCLUDE"
    )
)

summary = []

summary.append(
    "PERTURBATION MASTER DATASET - 82 RUNS"
)

summary.append("=" * 80)

summary.append(
    f"Epoch files: {len(files)}"
)

summary.append(
    f"Master records: {len(df)}"
)

summary.append(
    f"Expected records: {total_epochs}"
)

summary.append("")

summary.append(
    "PERTURBATION ELIGIBILITY"
)

summary.append(
    f"ELIGIBLE: {eligible}"
)

summary.append(
    f"ELIGIBLE_REVIEW: {eligible_review}"
)

summary.append(
    f"EXCLUDE: {excluded}"
)

summary.append("")

summary.append(
    "NO DATA WAS MODIFIED."
)

summary.append(
    "NO ORIGINAL EPOCH FILE WAS MODIFIED."
)

summary.append(
    "NO RAW DATA WAS MODIFIED."
)

with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("PERTURBATION MASTER DATASET COMPLETE")
print("=" * 80)

print()
print(
    f"Total records: {len(df)}"
)

print(
    f"ELIGIBLE:       {eligible}"
)

print(
    f"ELIGIBLE_REVIEW:{eligible_review}"
)

print(
    f"EXCLUDE:        {excluded}"
)

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(MASTER_CSV)
print(subject_summary_file)
print(condition_summary_file)
print(SUMMARY_TXT)

print()
print("=" * 80)
print("NO DATA WAS MODIFIED.")
print("NO EPOCH FILE WAS MODIFIED.")
print("=" * 80)