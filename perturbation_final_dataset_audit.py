import os
import re
import traceback
import numpy as np
import pandas as pd
import mne

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

DATASET_DIR = os.path.join(
    BASE_DIR,
    "epochs_clean",
    "logs",
    "perturbation_dataset"
)

ELIGIBLE_DIR = os.path.join(
    DATASET_DIR,
    "ELIGIBLE"
)

REVIEW_DIR = os.path.join(
    DATASET_DIR,
    "ELIGIBLE_REVIEW"
)

MANIFEST = os.path.join(
    DATASET_DIR,
    "perturbation_dataset_manifest.csv"
)

OUTPUT_DIR = os.path.join(
    DATASET_DIR,
    "final_audit"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIT_CSV = os.path.join(
    OUTPUT_DIR,
    "perturbation_final_dataset_audit.csv"
)

SUMMARY_TXT = os.path.join(
    OUTPUT_DIR,
    "perturbation_final_dataset_audit_summary.txt"
)

# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_CHANNELS = 71
EXPECTED_SFREQ = 500.0
EXPECTED_TIMES = 501

EXPECTED_ELIGIBLE = 21816
EXPECTED_ELIGIBLE_REVIEW = 1378

# ============================================================
# HELPERS
# ============================================================

def parse_filename(filename):

    subject_match = re.search(
        r"(sub-\d+)",
        filename
    )

    run_match = re.search(
        r"run-(\d+)",
        filename
    )

    if subject_match:
        subject = subject_match.group(1)
    else:
        subject = None

    if run_match:
        run = int(run_match.group(1))
    else:
        run = None

    return subject, run


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return np.nan


# ============================================================
# START
# ============================================================

print("=" * 80)
print("PERTURBATION FINAL DATASET AUDIT")
print("=" * 80)

# ============================================================
# CHECK MASTER MANIFEST
# ============================================================

print("\nLoading manifest:")

print(MANIFEST)

if not os.path.exists(MANIFEST):

    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST}"
    )

manifest = pd.read_csv(MANIFEST)

print(
    f"Manifest records: {len(manifest)}"
)

# ============================================================
# MANIFEST COUNTS
# ============================================================

print("\nMANIFEST DATASET GROUPS")
print("-" * 80)

print(
    manifest[
        "dataset_group"
    ].value_counts()
)

manifest_eligible = manifest[
    manifest["dataset_group"] == "ELIGIBLE"
].copy()

manifest_review = manifest[
    manifest["dataset_group"] == "ELIGIBLE_REVIEW"
].copy()

print(
    f"\nManifest ELIGIBLE: "
    f"{len(manifest_eligible)}"
)

print(
    f"Manifest ELIGIBLE_REVIEW: "
    f"{len(manifest_review)}"
)

# ============================================================
# FIND FILES
# ============================================================

print("\n")
print("=" * 80)
print("SEARCHING DATASET FILES")
print("=" * 80)

eligible_files = sorted(
    [
        os.path.join(
            ELIGIBLE_DIR,
            f
        )
        for f in os.listdir(ELIGIBLE_DIR)
        if f.endswith(
            "_perturbation_eligible_epo.fif"
        )
    ]
)

review_files = sorted(
    [
        os.path.join(
            REVIEW_DIR,
            f
        )
        for f in os.listdir(REVIEW_DIR)
        if f.endswith(
            "_perturbation_review_epo.fif"
        )
    ]
)

print(
    f"ELIGIBLE files:        {len(eligible_files)}"
)

print(
    f"ELIGIBLE_REVIEW files: {len(review_files)}"
)

# ============================================================
# AUDIT FUNCTION
# ============================================================

records = []

total_eligible_epochs = 0
total_review_epochs = 0

failed_files = []

# ============================================================
# AUDIT ELIGIBLE
# ============================================================

print("\n")
print("=" * 80)
print("AUDITING ELIGIBLE DATASET")
print("=" * 80)

for i, filepath in enumerate(
    eligible_files,
    1
):

    filename = os.path.basename(filepath)

    print(
        f"\n[{i}/{len(eligible_files)}] "
        f"{filename}"
    )

    try:

        epochs = mne.read_epochs(
            filepath,
            preload=True,
            verbose=False
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

        data = epochs.get_data(
            copy=False
        )

        nan_percent = (
            np.isnan(data).mean()
            * 100
        )

        inf_percent = (
            np.isinf(data).mean()
            * 100
        )

        finite_data = data[
            np.isfinite(data)
        ]

        if len(finite_data) > 0:

            global_min = float(
                np.min(finite_data)
            )

            global_max = float(
                np.max(finite_data)
            )

            global_std = float(
                np.std(finite_data)
            )

        else:

            global_min = np.nan
            global_max = np.nan
            global_std = np.nan

        subject, run = parse_filename(
            filename
        )

        status = "PASS"
        reasons = []

        if n_channels != EXPECTED_CHANNELS:

            status = "FAIL"

            reasons.append(
                f"CHANNELS_{n_channels}"
            )

        if n_times != EXPECTED_TIMES:

            status = "FAIL"

            reasons.append(
                f"TIMES_{n_times}"
            )

        if not np.isclose(
            sfreq,
            EXPECTED_SFREQ,
            atol=0.1
        ):

            status = "FAIL"

            reasons.append(
                f"SFREQ_{sfreq}"
            )

        if nan_percent > 0:

            status = "FAIL"

            reasons.append(
                "NAN_PRESENT"
            )

        if inf_percent > 0:

            status = "FAIL"

            reasons.append(
                "INF_PRESENT"
            )

        total_eligible_epochs += n_epochs

        records.append({

            "dataset_group":
                "ELIGIBLE",

            "file":
                filename,

            "subject":
                subject,

            "run":
                run,

            "n_epochs":
                n_epochs,

            "n_channels":
                n_channels,

            "n_times":
                n_times,

            "sfreq":
                sfreq,

            "nan_percent":
                nan_percent,

            "inf_percent":
                inf_percent,

            "global_min":
                global_min,

            "global_max":
                global_max,

            "global_std":
                global_std,

            "status":
                status,

            "reasons":
                ";".join(reasons)

        })

        print(
            f"Epochs: {n_epochs} | "
            f"Channels: {n_channels} | "
            f"SFREQ: {sfreq} | "
            f"STATUS: {status}"
        )

        if reasons:

            print(
                "REASONS:",
                ";".join(reasons)
            )

    except Exception as e:

        failed_files.append(
            filename
        )

        records.append({

            "dataset_group":
                "ELIGIBLE",

            "file":
                filename,

            "subject":
                parse_filename(filename)[0],

            "run":
                parse_filename(filename)[1],

            "n_epochs":
                np.nan,

            "n_channels":
                np.nan,

            "n_times":
                np.nan,

            "sfreq":
                np.nan,

            "nan_percent":
                np.nan,

            "inf_percent":
                np.nan,

            "global_min":
                np.nan,

            "global_max":
                np.nan,

            "global_std":
                np.nan,

            "status":
                "FAIL",

            "reasons":
                str(e)

        })

        print(
            "STATUS: FAIL"
        )

        print(
            "ERROR:",
            str(e)
        )

# ============================================================
# AUDIT REVIEW
# ============================================================

print("\n")
print("=" * 80)
print("AUDITING ELIGIBLE_REVIEW DATASET")
print("=" * 80)

for i, filepath in enumerate(
    review_files,
    1
):

    filename = os.path.basename(filepath)

    print(
        f"\n[{i}/{len(review_files)}] "
        f"{filename}"
    )

    try:

        epochs = mne.read_epochs(
            filepath,
            preload=True,
            verbose=False
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

        data = epochs.get_data(
            copy=False
        )

        nan_percent = (
            np.isnan(data).mean()
            * 100
        )

        inf_percent = (
            np.isinf(data).mean()
            * 100
        )

        finite_data = data[
            np.isfinite(data)
        ]

        if len(finite_data) > 0:

            global_min = float(
                np.min(finite_data)
            )

            global_max = float(
                np.max(finite_data)
            )

            global_std = float(
                np.std(finite_data)
            )

        else:

            global_min = np.nan
            global_max = np.nan
            global_std = np.nan

        subject, run = parse_filename(
            filename
        )

        status = "PASS"
        reasons = []

        if n_channels != EXPECTED_CHANNELS:

            status = "FAIL"

            reasons.append(
                f"CHANNELS_{n_channels}"
            )

        if n_times != EXPECTED_TIMES:

            status = "FAIL"

            reasons.append(
                f"TIMES_{n_times}"
            )

        if not np.isclose(
            sfreq,
            EXPECTED_SFREQ,
            atol=0.1
        ):

            status = "FAIL"

            reasons.append(
                f"SFREQ_{sfreq}"
            )

        if nan_percent > 0:

            status = "FAIL"

            reasons.append(
                "NAN_PRESENT"
            )

        if inf_percent > 0:

            status = "FAIL"

            reasons.append(
                "INF_PRESENT"
            )

        total_review_epochs += n_epochs

        records.append({

            "dataset_group":
                "ELIGIBLE_REVIEW",

            "file":
                filename,

            "subject":
                subject,

            "run":
                run,

            "n_epochs":
                n_epochs,

            "n_channels":
                n_channels,

            "n_times":
                n_times,

            "sfreq":
                sfreq,

            "nan_percent":
                nan_percent,

            "inf_percent":
                inf_percent,

            "global_min":
                global_min,

            "global_max":
                global_max,

            "global_std":
                global_std,

            "status":
                status,

            "reasons":
                ";".join(reasons)

        })

        print(
            f"Epochs: {n_epochs} | "
            f"Channels: {n_channels} | "
            f"SFREQ: {sfreq} | "
            f"STATUS: {status}"
        )

    except Exception as e:

        failed_files.append(
            filename
        )

        records.append({

            "dataset_group":
                "ELIGIBLE_REVIEW",

            "file":
                filename,

            "subject":
                parse_filename(filename)[0],

            "run":
                parse_filename(filename)[1],

            "n_epochs":
                np.nan,

            "n_channels":
                np.nan,

            "n_times":
                np.nan,

            "sfreq":
                np.nan,

            "nan_percent":
                np.nan,

            "inf_percent":
                np.nan,

            "global_min":
                np.nan,

            "global_max":
                np.nan,

            "global_std":
                np.nan,

            "status":
                "FAIL",

            "reasons":
                str(e)

        })

        print(
            "STATUS: FAIL"
        )

        print(
            "ERROR:",
            str(e)
        )

# ============================================================
# SAVE AUDIT
# ============================================================

audit_df = pd.DataFrame(
    records
)

audit_df.to_csv(
    AUDIT_CSV,
    index=False
)

# ============================================================
# VALIDATION
# ============================================================

print("\n")
print("=" * 80)
print("FINAL AUDIT VALIDATION")
print("=" * 80)

eligible_status = audit_df[
    audit_df["dataset_group"]
    == "ELIGIBLE"
]["status"]

review_status = audit_df[
    audit_df["dataset_group"]
    == "ELIGIBLE_REVIEW"
]["status"]

eligible_pass = int(
    (eligible_status == "PASS").sum()
)

review_pass = int(
    (review_status == "PASS").sum()
)

eligible_fail = int(
    (eligible_status == "FAIL").sum()
)

review_fail = int(
    (review_status == "FAIL").sum()
)

print(
    f"\nELIGIBLE files: "
    f"{len(eligible_files)}"
)

print(
    f"ELIGIBLE PASS: "
    f"{eligible_pass}"
)

print(
    f"ELIGIBLE FAIL: "
    f"{eligible_fail}"
)

print(
    f"\nELIGIBLE epochs: "
    f"{total_eligible_epochs}"
)

print(
    f"Expected ELIGIBLE epochs: "
    f"{EXPECTED_ELIGIBLE}"
)

print(
    "\nELIGIBLE COUNT CHECK:",
    "PASS"
    if total_eligible_epochs
    == EXPECTED_ELIGIBLE
    else "FAIL"
)

print(
    f"\nREVIEW files: "
    f"{len(review_files)}"
)

print(
    f"REVIEW PASS: "
    f"{review_pass}"
)

print(
    f"REVIEW FAIL: "
    f"{review_fail}"
)

print(
    f"\nREVIEW epochs: "
    f"{total_review_epochs}"
)

print(
    f"Expected REVIEW epochs: "
    f"{EXPECTED_ELIGIBLE_REVIEW}"
)

print(
    "\nREVIEW COUNT CHECK:",
    "PASS"
    if total_review_epochs
    == EXPECTED_ELIGIBLE_REVIEW
    else "FAIL"
)

# ============================================================
# MASTER CONSISTENCY
# ============================================================

manifest_eligible_count = len(
    manifest_eligible
)

manifest_review_count = len(
    manifest_review
)

print("\n")
print("=" * 80)
print("MANIFEST CONSISTENCY")
print("=" * 80)

print(
    f"Manifest ELIGIBLE records: "
    f"{manifest_eligible_count}"
)

print(
    f"Actual ELIGIBLE epochs: "
    f"{total_eligible_epochs}"
)

print(
    "ELIGIBLE MANIFEST CHECK:",
    "PASS"
    if manifest_eligible_count
    == total_eligible_epochs
    else "FAIL"
)

print(
    f"\nManifest REVIEW records: "
    f"{manifest_review_count}"
)

print(
    f"Actual REVIEW epochs: "
    f"{total_review_epochs}"
)

print(
    "REVIEW MANIFEST CHECK:",
    "PASS"
    if manifest_review_count
    == total_review_epochs
    else "FAIL"
)

# ============================================================
# GLOBAL STATUS
# ============================================================

global_checks = [

    total_eligible_epochs
    == EXPECTED_ELIGIBLE,

    total_review_epochs
    == EXPECTED_ELIGIBLE_REVIEW,

    eligible_fail == 0,

    review_fail == 0,

    manifest_eligible_count
    == total_eligible_epochs,

    manifest_review_count
    == total_review_epochs

]

if all(global_checks):

    final_status = "PASS"

else:

    final_status = "REVIEW"

# ============================================================
# SUMMARY
# ============================================================

with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "PERTURBATION FINAL DATASET AUDIT\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"FINAL STATUS: {final_status}\n\n"
    )

    f.write(
        f"ELIGIBLE files: "
        f"{len(eligible_files)}\n"
    )

    f.write(
        f"ELIGIBLE epochs: "
        f"{total_eligible_epochs}\n"
    )

    f.write(
        f"Expected ELIGIBLE: "
        f"{EXPECTED_ELIGIBLE}\n\n"
    )

    f.write(
        f"ELIGIBLE_REVIEW files: "
        f"{len(review_files)}\n"
    )

    f.write(
        f"ELIGIBLE_REVIEW epochs: "
        f"{total_review_epochs}\n"
    )

    f.write(
        f"Expected REVIEW: "
        f"{EXPECTED_ELIGIBLE_REVIEW}\n\n"
    )

    f.write(
        f"ELIGIBLE PASS: "
        f"{eligible_pass}\n"
    )

    f.write(
        f"ELIGIBLE FAIL: "
        f"{eligible_fail}\n\n"
    )

    f.write(
        f"REVIEW PASS: "
        f"{review_pass}\n"
    )

    f.write(
        f"REVIEW FAIL: "
        f"{review_fail}\n\n"
    )

    f.write(
        f"Failed files: "
        f"{len(failed_files)}\n"
    )

    if failed_files:

        f.write(
            "\nFAILED FILES\n"
        )

        for filename in failed_files:

            f.write(
                f"{filename}\n"
            )

    f.write(
        "\n\n"
    )

    f.write(
        "RAW DATA WAS NOT MODIFIED.\n"
    )

    f.write(
        "NO SET FILE WAS MODIFIED.\n"
    )

    f.write(
        "NO FDT FILE WAS MODIFIED.\n"
    )

    f.write(
        "NO EPOCH FILE WAS MODIFIED.\n"
    )

# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("PERTURBATION FINAL DATASET AUDIT COMPLETE")
print("=" * 80)

print(
    f"\nFINAL STATUS: {final_status}"
)

print(
    f"\nELIGIBLE epochs: "
    f"{total_eligible_epochs}"
)

print(
    f"ELIGIBLE_REVIEW epochs: "
    f"{total_review_epochs}"
)

print(
    f"Failed files: "
    f"{len(failed_files)}"
)

print("\nSaved:")
print(AUDIT_CSV)
print(SUMMARY_TXT)

print("\n")
print("=" * 80)
print("NO DATA WAS MODIFIED.")
print("=" * 80)