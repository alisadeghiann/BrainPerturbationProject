import os
import glob
import numpy as np
import h5py
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "qc"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIND ALL SET FILES
# ============================================================

set_files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "sub-*",
            "ses-*",
            "eeg",
            "*.set"
        )
    )
)


print("=" * 80)
print("SAMPLING RATE CHECK")
print("=" * 80)

print()
print(
    f"Total .set files found: {len(set_files)}"
)


if len(set_files) == 0:

    raise FileNotFoundError(
        "No .set files found."
    )


# ============================================================
# READ SET METADATA
# ============================================================

results = []


for file_number, set_file in enumerate(
    set_files,
    start=1
):

    print()
    print("=" * 80)

    print(
        f"PROCESSING "
        f"{file_number}/{len(set_files)}"
    )

    print("=" * 80)

    filename = os.path.basename(
        set_file
    )

    path_parts = set_file.split(
        os.sep
    )

    subject = next(
        (
            part
            for part in path_parts
            if part.startswith("sub-")
        ),
        "unknown"
    )

    session = next(
        (
            part
            for part in path_parts
            if part.startswith("ses-")
        ),
        "unknown"
    )

    try:

        with h5py.File(
            set_file,
            "r"
        ) as f:

            # ------------------------------------------------
            # Sampling rate
            # ------------------------------------------------

            srate = float(
                np.array(
                    f["srate"]
                ).squeeze()
            )

            # ------------------------------------------------
            # Number of channels
            # ------------------------------------------------

            nbchan = int(
                np.array(
                    f["nbchan"]
                ).squeeze()
            )

            # ------------------------------------------------
            # Number of samples
            # ------------------------------------------------

            pnts = int(
                np.array(
                    f["pnts"]
                ).squeeze()
            )

            # ------------------------------------------------
            # Number of trials
            # ------------------------------------------------

            trials = int(
                np.array(
                    f["trials"]
                ).squeeze()
            )

            # ------------------------------------------------
            # Duration
            # ------------------------------------------------

            duration = (
                pnts
                / srate
            )

        results.append({

            "subject":
                subject,

            "session":
                session,

            "file":
                filename,

            "sampling_rate":
                srate,

            "channels":
                nbchan,

            "samples":
                pnts,

            "trials":
                trials,

            "duration_seconds":
                duration

        })

        print(
            f"Subject: {subject}"
        )

        print(
            f"Sampling rate: {srate} Hz"
        )

        print(
            f"Channels: {nbchan}"
        )

        print(
            f"Samples: {pnts}"
        )

        print(
            f"Duration: {duration:.2f} sec"
        )

    except Exception as e:

        print(
            f"ERROR: {repr(e)}"
        )

        results.append({

            "subject":
                subject,

            "session":
                session,

            "file":
                filename,

            "error":
                repr(e)

        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    results
)


# ============================================================
# SAVE COMPLETE REPORT
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "sampling_rate_report.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SAMPLING RATE SUMMARY
# ============================================================

print()
print("=" * 80)

print(
    "SAMPLING RATE SUMMARY"
)

print("=" * 80)

print()

if "sampling_rate" in df.columns:

    rate_counts = (
        df[
            "sampling_rate"
        ]
        .value_counts()
        .sort_index()
    )

    for rate, count in rate_counts.items():

        print(
            f"{rate} Hz : "
            f"{count} files"
        )


# ============================================================
# SUBJECT-LEVEL SUMMARY
# ============================================================

print()
print("=" * 80)

print(
    "SAMPLING RATE BY SUBJECT"
)

print("=" * 80)

print()

if (
    "subject" in df.columns
    and "sampling_rate" in df.columns
):

    subject_summary = (
        df.groupby(
            "subject"
        )[
            "sampling_rate"
        ]
        .agg(
            [
                "min",
                "max",
                "nunique"
            ]
        )
        .reset_index()
    )

    subject_summary.columns = [
        "subject",
        "min_srate",
        "max_srate",
        "number_of_different_rates"
    ]

    print(
        subject_summary.to_string(
            index=False
        )
    )


# ============================================================
# CHECK FOR MIXED SAMPLING RATES
# ============================================================

print()
print("=" * 80)

print(
    "CONSISTENCY CHECK"
)

print("=" * 80)

print()

unique_rates = sorted(
    df[
        "sampling_rate"
    ]
    .dropna()
    .unique()
)


print(
    "Unique sampling rates:"
)

for rate in unique_rates:

    print(
        f"- {rate} Hz"
    )


if len(unique_rates) == 1:

    print()
    print(
        "RESULT: All files have the same sampling rate."
    )

else:

    print()
    print(
        "RESULT: Multiple sampling rates detected."
    )

    print()
    print(
        "This must be handled before preprocessing."
    )


# ============================================================
# FIND SUBJECTS WITH MIXED RATES
# ============================================================

if (
    "subject" in df.columns
    and "sampling_rate" in df.columns
):

    mixed_subjects = (
        df.groupby(
            "subject"
        )[
            "sampling_rate"
        ]
        .nunique()
    )

    mixed_subjects = (
        mixed_subjects[
            mixed_subjects > 1
        ]
    )

    print()
    print("=" * 80)

    print(
        "SUBJECTS WITH MIXED SAMPLING RATES"
    )

    print("=" * 80)

    print()

    if len(
        mixed_subjects
    ) == 0:

        print(
            "None."
        )

    else:

        for subject in mixed_subjects.index:

            print(
                subject
            )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)

print(
    "SAMPLING RATE CHECK COMPLETE"
)

print("=" * 80)

print()

print(
    "Report saved to:"
)

print(
    output_file
)

print()

print(
    "DONE."
)