from pathlib import Path
import numpy as np
import pandas as pd
import mne

# ============================================================
# PROJECT
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

SUBJECT = "sub-024"

# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR = (
    BASE
    / "data"
    / SUBJECT
    / "ses-01"
    / "eeg"
)

CLEAN_DIR = (
    BASE
    / "epochs_clean"
)

HARMONIZED_DIR = (
    BASE
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v5"
    / "ELIGIBLE"
)


# ============================================================
# PRINT SEPARATOR
# ============================================================

def separator(title):

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# LOAD ORIGINAL EVENTS
#
# IMPORTANT:
# Original BIDS files use:
# run-1
# run-2
# run-3
# ============================================================

def load_events(run):

    filename = (
        f"{SUBJECT}_ses-01_task-WorkingMemory"
        f"_run-{run}_events.tsv"
    )

    path = DATA_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"\nEvents file not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        sep="\t"
    )

    return df, path


# ============================================================
# LOAD CLEAN EPOCHS
#
# IMPORTANT:
# Clean files also use:
# run-1
# run-2
# run-3
# ============================================================

def load_clean_epochs(run):

    filename = (
        f"{SUBJECT}_ses-01_task-WorkingMemory"
        f"_run-{run}_clean_epo.fif"
    )

    path = CLEAN_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"\nClean epochs file not found:\n{path}"
        )

    epochs = mne.read_epochs(
        path,
        preload=False,
        verbose=False
    )

    return epochs, path


# ============================================================
# LOAD HARMONIZED EPOCHS
#
# IMPORTANT:
# Harmonized files use:
# run-01
# run-02
# run-03
# ============================================================

def load_harmonized_epochs(run):

    run_str = f"{run:02d}"

    filename = (
        f"{SUBJECT}_run-{run_str}"
        f"_harmonized_epo.fif"
    )

    path = HARMONIZED_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"\nHarmonized epochs file not found:\n{path}"
        )

    epochs = mne.read_epochs(
        path,
        preload=False,
        verbose=False
    )

    return epochs, path


# ============================================================
# EVENT SUMMARY
# ============================================================

def summarize_events(df):

    return {

        "total_events":
            len(df),

        "show_letter":
            int(
                (
                    df["event_type"]
                    == "show_letter"
                ).sum()
            ),

        "to_remember":
            int(
                (
                    df["task_role"]
                    == "to_remember"
                ).sum()
            ),

        "to_ignore":
            int(
                (
                    df["task_role"]
                    == "to_ignore"
                ).sum()
            ),

        "ignored_correct":
            int(
                (
                    df["task_role"]
                    == "ignored_correct"
                ).sum()
            ),

        "ignored_incorrect":
            int(
                (
                    df["task_role"]
                    == "ignored_incorrect"
                ).sum()
            ),

        "probe_target":
            int(
                (
                    df["task_role"]
                    == "probe_target"
                ).sum()
            ),

        "work_memory":
            int(
                (
                    df["task_role"]
                    == "work_memory"
                ).sum()
            ),
    }


# ============================================================
# GET EPOCH EVENT TIMES
# ============================================================

def get_epoch_event_times(epochs):

    sfreq = epochs.info["sfreq"]

    sample_indices = epochs.events[:, 0]

    return sample_indices / sfreq


# ============================================================
# MATCH EPOCHS TO ORIGINAL EVENTS
# ============================================================

def match_epochs_to_events(
    epoch_times,
    events_df,
    tolerance=0.10
):

    original_times = (
        events_df["onset"]
        .to_numpy()
    )

    results = []

    for epoch_index, epoch_time in enumerate(
        epoch_times
    ):

        distances = np.abs(
            original_times - epoch_time
        )

        nearest_index = np.argmin(
            distances
        )

        nearest_distance = (
            distances[nearest_index]
        )

        if nearest_distance <= tolerance:

            row = events_df.iloc[
                nearest_index
            ]

            results.append(
                {
                    "epoch_index":
                        epoch_index,

                    "epoch_time":
                        epoch_time,

                    "matched":
                        True,

                    "distance_sec":
                        nearest_distance,

                    "event_type":
                        row["event_type"],

                    "task_role":
                        row["task_role"],

                    "letter":
                        row["letter"],

                    "trial":
                        row["trial"],

                    "memory_cond":
                        row["memory_cond"],

                    "value":
                        row["value"],

                    "original_onset":
                        row["onset"],
                }
            )

        else:

            results.append(
                {
                    "epoch_index":
                        epoch_index,

                    "epoch_time":
                        epoch_time,

                    "matched":
                        False,

                    "distance_sec":
                        nearest_distance,

                    "event_type":
                        None,

                    "task_role":
                        None,

                    "letter":
                        None,

                    "trial":
                        None,

                    "memory_cond":
                        None,

                    "value":
                        None,

                    "original_onset":
                        None,
                }
            )

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

separator(
    f"{SUBJECT} EPOCH COMPARISON"
)

print("Project:")
print(BASE)

print("\nSubject:")
print(SUBJECT)


# ============================================================
# RESULTS
# ============================================================

all_results = []


# ============================================================
# RUN 1, 2, 3
# ============================================================

for run in [1, 2, 3]:

    separator(
        f"RUN {run}"
    )

    # --------------------------------------------------------
    # ORIGINAL EVENTS
    # --------------------------------------------------------

    events_df, events_path = (
        load_events(run)
    )

    print(
        "\nOriginal events file:"
    )

    print(events_path)

    print(
        f"\nOriginal events: "
        f"{len(events_df)}"
    )

    # --------------------------------------------------------
    # EVENT SUMMARY
    # --------------------------------------------------------

    summary = summarize_events(
        events_df
    )

    print(
        "\n--- ORIGINAL EVENTS SUMMARY ---"
    )

    for key, value in summary.items():

        print(
            f"{key:25s}: {value}"
        )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    clean_epochs, clean_path = (
        load_clean_epochs(run)
    )

    print(
        "\n--- CLEAN EPOCHS ---"
    )

    print("File:")
    print(clean_path)

    print(
        f"Number of epochs : "
        f"{len(clean_epochs)}"
    )

    print(
        f"Sampling rate    : "
        f"{clean_epochs.info['sfreq']}"
    )

    print(
        f"Channels         : "
        f"{len(clean_epochs.ch_names)}"
    )

    print(
        "\nClean event_id:"
    )

    print(
        clean_epochs.event_id
    )

    print(
        "\nClean epoch time range:"
    )

    print(
        f"{clean_epochs.tmin:.3f} "
        f"to "
        f"{clean_epochs.tmax:.3f} sec"
    )

    # --------------------------------------------------------
    # HARMONIZED
    # --------------------------------------------------------

    harmonized_epochs, harmonized_path = (
        load_harmonized_epochs(run)
    )

    print(
        "\n--- HARMONIZED EPOCHS ---"
    )

    print("File:")
    print(harmonized_path)

    print(
        f"Number of epochs : "
        f"{len(harmonized_epochs)}"
    )

    print(
        f"Sampling rate    : "
        f"{harmonized_epochs.info['sfreq']}"
    )

    print(
        f"Channels         : "
        f"{len(harmonized_epochs.ch_names)}"
    )

    print(
        "\nHarmonized event_id:"
    )

    print(
        harmonized_epochs.event_id
    )

    print(
        "\nHarmonized epoch time range:"
    )

    print(
        f"{harmonized_epochs.tmin:.3f} "
        f"to "
        f"{harmonized_epochs.tmax:.3f} sec"
    )

    # --------------------------------------------------------
    # TIMES
    # --------------------------------------------------------

    clean_times = (
        get_epoch_event_times(
            clean_epochs
        )
    )

    harmonized_times = (
        get_epoch_event_times(
            harmonized_epochs
        )
    )

    # --------------------------------------------------------
    # CLEAN MATCH
    # --------------------------------------------------------

    print(
        "\n--- CLEAN EPOCH → ORIGINAL EVENT MATCH ---"
    )

    clean_matches = (
        match_epochs_to_events(
            clean_times,
            events_df
        )
    )

    clean_matched = int(
        clean_matches["matched"].sum()
    )

    clean_unmatched = int(
        (~clean_matches["matched"]).sum()
    )

    print(
        f"Matched: "
        f"{clean_matched} / "
        f"{len(clean_matches)}"
    )

    print(
        f"Unmatched: "
        f"{clean_unmatched}"
    )

    print(
        "\nMatched task_role counts:"
    )

    if clean_matched > 0:

        print(
            clean_matches.loc[
                clean_matches["matched"],
                "task_role"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    # --------------------------------------------------------
    # HARMONIZED MATCH
    # --------------------------------------------------------

    print(
        "\n--- HARMONIZED EPOCH → ORIGINAL EVENT MATCH ---"
    )

    harmonized_matches = (
        match_epochs_to_events(
            harmonized_times,
            events_df
        )
    )

    harmonized_matched = int(
        harmonized_matches["matched"].sum()
    )

    harmonized_unmatched = int(
        (~harmonized_matches["matched"]).sum()
    )

    print(
        f"Matched: "
        f"{harmonized_matched} / "
        f"{len(harmonized_matches)}"
    )

    print(
        f"Unmatched: "
        f"{harmonized_unmatched}"
    )

    print(
        "\nMatched task_role counts:"
    )

    if harmonized_matched > 0:

        print(
            harmonized_matches.loc[
                harmonized_matches["matched"],
                "task_role"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    # --------------------------------------------------------
    # CONDITION COUNTS
    # --------------------------------------------------------

    print(
        "\n--- CONDITION COUNTS AFTER MATCHING ---"
    )

    clean_remember = int(
        (
            clean_matches["task_role"]
            == "to_remember"
        ).sum()
    )

    clean_ignore = int(
        (
            clean_matches["task_role"]
            == "to_ignore"
        ).sum()
    )

    harmonized_remember = int(
        (
            harmonized_matches["task_role"]
            == "to_remember"
        ).sum()
    )

    harmonized_ignore = int(
        (
            harmonized_matches["task_role"]
            == "to_ignore"
        ).sum()
    )

    print(
        f"CLEAN       | "
        f"remember = {clean_remember:4d} | "
        f"ignore = {clean_ignore:4d} | "
        f"total = "
        f"{clean_remember + clean_ignore:4d}"
    )

    print(
        f"HARMONIZED  | "
        f"remember = {harmonized_remember:4d} | "
        f"ignore = {harmonized_ignore:4d} | "
        f"total = "
        f"{harmonized_remember + harmonized_ignore:4d}"
    )

    # --------------------------------------------------------
    # FIRST 30 HARMONIZED MATCHES
    # --------------------------------------------------------

    print(
        "\n--- FIRST 30 HARMONIZED MATCHES ---"
    )

    display_columns = [
        "epoch_index",
        "epoch_time",
        "distance_sec",
        "event_type",
        "task_role",
        "letter",
        "trial",
        "memory_cond",
        "value",
        "original_onset",
    ]

    print(
        harmonized_matches[
            display_columns
        ]
        .head(30)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    all_results.append(
        {
            "subject":
                SUBJECT,

            "run":
                run,

            "original_events":
                len(events_df),

            "original_remember":
                summary["to_remember"],

            "original_ignore":
                summary["to_ignore"],

            "clean_epochs":
                len(clean_epochs),

            "harmonized_epochs":
                len(harmonized_epochs),

            "clean_matched":
                clean_matched,

            "harmonized_matched":
                harmonized_matched,

            "clean_remember":
                clean_remember,

            "clean_ignore":
                clean_ignore,

            "harmonized_remember":
                harmonized_remember,

            "harmonized_ignore":
                harmonized_ignore,
        }
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

separator(
    "FINAL SUB-024 COMPARISON SUMMARY"
)

results_df = pd.DataFrame(
    all_results
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# IMPORTANT INTERPRETATION
# ============================================================

separator(
    "INTERPRETATION"
)

print(
    """
READ-ONLY ANALYSIS

No EEG, FIF, TSV, CSV, or QC file was modified.

We are comparing:

Original BIDS events.tsv
        ↓
Clean epochs
        ↓
Harmonized epochs

The critical quantities are:

original_remember
original_ignore

clean_remember
clean_ignore

harmonized_remember
harmonized_ignore

If the original events contain Remember/Ignore but the
harmonized epochs do not map to them, then the problem is
the condition mapping and NOT missing behavioral data.
"""
)

print(
    "\nDONE."
)