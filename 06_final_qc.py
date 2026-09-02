# ============================================================
# 06_FINAL_QC.py
# Final automated quality control for EEG dataset
# ============================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import mne

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

DATA_DIR = PROJECT_ROOT / "data"
QC_DIR = PROJECT_ROOT / "qc"

QC_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = QC_DIR / "final_qc_report.csv"

# ============================================================
# SETTINGS
# ============================================================

# Thresholds are intentionally conservative.
# This stage ONLY detects problems.
# It does NOT delete or repair anything.

FLAT_STD = 1.0

# A channel whose STD is > 5x the median channel STD
# is suspicious.
CHANNEL_STD_RATIO = 5.0

# >10x is highly suspicious.
CHANNEL_STD_RATIO_HIGH = 10.0

# Saturation limits seen in your dataset
SATURATION_LEVELS = [500, 1000]

# Percentage of samples at saturation that triggers warning
SATURATION_PERCENT = 0.01

# ============================================================
# FIND EEG FILES
# ============================================================

files = sorted(DATA_DIR.rglob("*_eeg.set"))

print("=" * 80)
print("FINAL EEG QUALITY CONTROL")
print("=" * 80)

print(f"Project: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")
print(f"EEG files found: {len(files)}")

if len(files) == 0:
    raise FileNotFoundError(
        f"No EEG .set files found in:\n{DATA_DIR}"
    )

print("=" * 80)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def get_event_count(raw):
    """
    Try to obtain the number of events.
    This is deliberately tolerant because EEGLAB event
    structures can vary.
    """

    try:
        events, event_id = mne.events_from_annotations(
            raw,
            verbose=False
        )

        return len(events)

    except Exception:
        return 0


def detect_saturation(data):
    """
    Detect values close to common clipping/saturation levels.
    """

    total = data.size

    if total == 0:
        return {
            "sat_500_percent": 0.0,
            "sat_1000_percent": 0.0,
            "saturation_flag": False
        }

    # Absolute values near 500
    near_500 = np.abs(np.abs(data) - 500) < 0.05

    # Absolute values near 1000
    near_1000 = np.abs(np.abs(data) - 1000) < 0.05

    p500 = 100 * np.sum(near_500) / total
    p1000 = 100 * np.sum(near_1000) / total

    flag = (
        p500 >= SATURATION_PERCENT
        or
        p1000 >= SATURATION_PERCENT
    )

    return {
        "sat_500_percent": p500,
        "sat_1000_percent": p1000,
        "saturation_flag": flag
    }


# ============================================================
# MAIN LOOP
# ============================================================

results = []

for i, file in enumerate(files, start=1):

    print()
    print("=" * 80)
    print(f"PROCESSING {i}/{len(files)}")
    print("=" * 80)

    print(file.name)

    result = {
        "subject": "",
        "session": "",
        "run": "",
        "file": file.name,
        "path": str(file),

        "sampling_rate": np.nan,
        "n_channels": np.nan,
        "n_eeg_channels": np.nan,
        "n_eog_channels": np.nan,
        "n_samples": np.nan,
        "duration_sec": np.nan,

        "median_channel_std": np.nan,
        "max_channel_std": np.nan,
        "median_channel_ptp": np.nan,
        "max_channel_ptp": np.nan,

        "min_value": np.nan,
        "max_value": np.nan,

        "channels_over_5x": 0,
        "channels_over_10x": 0,
        "flat_channels": 0,

        "sat_500_percent": np.nan,
        "sat_1000_percent": np.nan,

        "event_count": 0,

        "sampling_flag": "",
        "amplitude_flag": "",
        "channel_flag": "",
        "event_flag": "",
        "overall_status": "",

        "error": ""
    }

    # --------------------------------------------------------
    # Subject / session / run
    # --------------------------------------------------------

    parts = file.name.split("_")

    for p in parts:

        if p.startswith("sub-"):
            result["subject"] = p

        elif p.startswith("ses-"):
            result["session"] = p

        elif p.startswith("run-"):
            result["run"] = p.replace(".set", "")

    # --------------------------------------------------------
    # Read EEG
    # --------------------------------------------------------

    try:

        raw = mne.io.read_raw_eeglab(
            str(file),
            preload=True,
            verbose=False
        )

        data = raw.get_data()

        sfreq = raw.info["sfreq"]

        result["sampling_rate"] = sfreq
        result["n_channels"] = raw.info["nchan"]
        result["n_samples"] = data.shape[1]
        result["duration_sec"] = data.shape[1] / sfreq

        # ----------------------------------------------------
        # Channel types
        # ----------------------------------------------------

        eeg_picks = mne.pick_types(
            raw.info,
            eeg=True,
            eog=False,
            exclude=[]
        )

        eog_picks = mne.pick_types(
            raw.info,
            eeg=False,
            eog=True,
            exclude=[]
        )

        result["n_eeg_channels"] = len(eeg_picks)
        result["n_eog_channels"] = len(eog_picks)

        # ----------------------------------------------------
        # EEG data only
        # ----------------------------------------------------

        eeg_data = data[eeg_picks]

        # ----------------------------------------------------
        # Channel statistics
        # ----------------------------------------------------

        channel_std = np.std(
            eeg_data,
            axis=1
        )

        channel_ptp = (
            np.max(eeg_data, axis=1)
            -
            np.min(eeg_data, axis=1)
        )

        median_std = np.median(channel_std)

        result["median_channel_std"] = median_std
        result["max_channel_std"] = np.max(channel_std)

        result["median_channel_ptp"] = np.median(channel_ptp)
        result["max_channel_ptp"] = np.max(channel_ptp)

        result["min_value"] = np.min(eeg_data)
        result["max_value"] = np.max(eeg_data)

        # ----------------------------------------------------
        # Bad channel detection
        # ----------------------------------------------------

        if median_std > 0:

            over_5 = channel_std > (
                CHANNEL_STD_RATIO * median_std
            )

            over_10 = channel_std > (
                CHANNEL_STD_RATIO_HIGH * median_std
            )

        else:

            over_5 = np.zeros_like(
                channel_std,
                dtype=bool
            )

            over_10 = np.zeros_like(
                channel_std,
                dtype=bool
            )

        flat = channel_std < FLAT_STD

        result["channels_over_5x"] = int(
            np.sum(over_5)
        )

        result["channels_over_10x"] = int(
            np.sum(over_10)
        )

        result["flat_channels"] = int(
            np.sum(flat)
        )

        # ----------------------------------------------------
        # Saturation
        # ----------------------------------------------------

        sat = detect_saturation(eeg_data)

        result["sat_500_percent"] = sat[
            "sat_500_percent"
        ]

        result["sat_1000_percent"] = sat[
            "sat_1000_percent"
        ]

        # ----------------------------------------------------
        # Events
        # ----------------------------------------------------

        result["event_count"] = get_event_count(raw)

        # ====================================================
        # FLAGS
        # ====================================================

        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        if np.isclose(sfreq, 250):
            result["sampling_flag"] = "OK_250Hz"

        elif np.isclose(sfreq, 500):
            result["sampling_flag"] = "OK_500Hz"

        elif np.isclose(sfreq, 500.0586756, atol=0.01):
            result["sampling_flag"] = "RESAMPLE_NEEDED_500.0587Hz"

        elif np.isclose(sfreq, 1000):
            result["sampling_flag"] = "RESAMPLE_NEEDED_1000Hz"

        else:
            result["sampling_flag"] = "UNUSUAL_RATE"

        # ----------------------------------------------------
        # Amplitude
        # ----------------------------------------------------

        if (
            result["sat_1000_percent"] > 0
            or
            result["sat_500_percent"] > 0
        ):
            result["amplitude_flag"] = "SATURATION_PRESENT"

        else:
            result["amplitude_flag"] = "NO_CLEAR_SATURATION"

        # ----------------------------------------------------
        # Channel flag
        # ----------------------------------------------------

        if result["channels_over_10x"] > 0:

            result["channel_flag"] = (
                "HIGHLY_SUSPICIOUS_CHANNELS"
            )

        elif result["channels_over_5x"] > 0:

            result["channel_flag"] = (
                "SUSPICIOUS_CHANNELS"
            )

        elif result["flat_channels"] > 0:

            result["channel_flag"] = (
                "FLAT_CHANNELS"
            )

        else:

            result["channel_flag"] = "OK"

        # ----------------------------------------------------
        # Event flag
        # ----------------------------------------------------

        if result["event_count"] > 0:

            result["event_flag"] = "EVENTS_PRESENT"

        else:

            result["event_flag"] = "NO_EVENTS_DETECTED"

        # ====================================================
        # OVERALL STATUS
        # ====================================================

        serious_channel_problem = (
            result["channels_over_10x"] > 0
            or
            result["flat_channels"] >= 3
        )

        serious_saturation = (
            result["sat_1000_percent"] > 0.1
        )

        if serious_channel_problem:

            result["overall_status"] = "QUESTIONABLE"

        elif serious_saturation:

            result["overall_status"] = "QUESTIONABLE"

        elif (
            result["sampling_flag"]
            in [
                "RESAMPLE_NEEDED_500.0587Hz",
                "RESAMPLE_NEEDED_1000Hz"
            ]
        ):

            result["overall_status"] = "REVIEW"

        else:

            result["overall_status"] = "GOOD"

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print(
            f"Sampling rate : {sfreq:.6f} Hz"
        )

        print(
            f"Channels      : "
            f"{result['n_channels']}"
        )

        print(
            f"EEG channels  : "
            f"{result['n_eeg_channels']}"
        )

        print(
            f"EOG channels  : "
            f"{result['n_eog_channels']}"
        )

        print(
            f"Duration      : "
            f"{result['duration_sec']:.2f} sec"
        )

        print(
            f"Median STD    : "
            f"{result['median_channel_std']:.3f}"
        )

        print(
            f"Max STD       : "
            f"{result['max_channel_std']:.3f}"
        )

        print(
            f">5x channels  : "
            f"{result['channels_over_5x']}"
        )

        print(
            f">10x channels : "
            f"{result['channels_over_10x']}"
        )

        print(
            f"Flat channels : "
            f"{result['flat_channels']}"
        )

        print(
            f"Events        : "
            f"{result['event_count']}"
        )

        print(
            f"STATUS        : "
            f"{result['overall_status']}"
        )

    except Exception as e:

        result["overall_status"] = "ERROR"

        result["error"] = str(e)

        print(
            f"ERROR: {e}"
        )

    results.append(result)


# ============================================================
# SAVE REPORT
# ============================================================

df = pd.DataFrame(results)

# Sort most problematic files first
status_order = {
    "ERROR": 0,
    "QUESTIONABLE": 1,
    "REVIEW": 2,
    "GOOD": 3
}

df["_sort"] = df["overall_status"].map(
    status_order
).fillna(99)

df = df.sort_values(
    ["_sort", "subject", "run"]
)

df = df.drop(
    columns=["_sort"]
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("FINAL QC SUMMARY")
print("=" * 80)

print(
    f"Total files: {len(df)}"
)

print()

print("STATUS COUNTS:")
print(
    df["overall_status"].value_counts()
)

print()
print("SAMPLING RATES:")
print(
    df["sampling_rate"]
    .round(6)
    .value_counts()
    .sort_index()
)

print()
print("FILES WITH >10x CHANNELS:")
print(
    df[
        df["channels_over_10x"] > 0
    ][
        [
            "subject",
            "file",
            "sampling_rate",
            "median_channel_std",
            "max_channel_std",
            "channels_over_10x",
            "overall_status"
        ]
    ].to_string(index=False)
)

print()
print("FILES WITH >5x CHANNELS:")
print(
    df[
        df["channels_over_5x"] > 0
    ][
        [
            "subject",
            "file",
            "channels_over_5x",
            "overall_status"
        ]
    ].to_string(index=False)
)

print()
print("FILES WITH FLAT CHANNELS:")
print(
    df[
        df["flat_channels"] > 0
    ][
        [
            "subject",
            "file",
            "flat_channels",
            "overall_status"
        ]
    ].to_string(index=False)
)

print()
print("=" * 80)
print("REPORT SAVED")
print("=" * 80)

print(
    OUTPUT_FILE
)

print()
print("IMPORTANT:")
print(
    "Original EEG files were NOT modified."
)

print(
    "No channels were deleted."
)

print(
    "No subjects were removed."
)

print(
    "No resampling or preprocessing was performed."
)

print("=" * 80)
print("DONE.")
print("=" * 80)