import os
import glob
import numpy as np
import pandas as pd
import mne

# ============================================================
# EPOCH ARTIFACT DURATION QC - 82 RUNS
# READ-ONLY
# ============================================================

BASE_DIR = r"C:\Users\Ali\Desktop\BrainPerturbationProject"

INPUT_DIR = os.path.join(BASE_DIR, "epochs_clean")

OUT_DIR = os.path.join(
    INPUT_DIR,
    "logs",
    "artifact_duration_qc"
)

os.makedirs(OUT_DIR, exist_ok=True)

CSV_OUT = os.path.join(
    OUT_DIR,
    "epoch_artifact_duration_qc_82runs.csv"
)

SUMMARY_OUT = os.path.join(
    OUT_DIR,
    "epoch_artifact_duration_qc_summary.txt"
)

files = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "*_clean_epo.fif")
    )
)

print("=" * 80)
print("EPOCH ARTIFACT DURATION QC - 82 RUNS")
print("=" * 80)

print(f"Files found: {len(files)}")

records = []

for file_idx, path in enumerate(files, 1):

    fname = os.path.basename(path)

    print("")
    print("=" * 80)
    print(f"[{file_idx}/{len(files)}] {fname}")
    print("=" * 80)

    try:

        epochs = mne.read_epochs(
            path,
            preload=True,
            verbose=False
        )

        data = epochs.get_data(copy=False)

        n_epochs, n_channels, n_times = data.shape

        sfreq = float(
            epochs.info["sfreq"]
        )

        dt_ms = 1000.0 / sfreq

        absdata = np.abs(data)

        # ----------------------------------------------------
        # Threshold masks
        # ----------------------------------------------------

        mask200 = absdata > 200
        mask300 = absdata > 300
        mask400 = absdata > 400

        # ----------------------------------------------------
        # Per epoch calculations
        # ----------------------------------------------------

        for ep in range(n_epochs):

            x200 = mask200[ep]
            x300 = mask300[ep]
            x400 = mask400[ep]

            total_samples = n_channels * n_times

            # -----------------------------------------------
            # Overall sample percentages
            # -----------------------------------------------

            pct200 = (
                np.sum(x200)
                / total_samples
                * 100
            )

            pct300 = (
                np.sum(x300)
                / total_samples
                * 100
            )

            pct400 = (
                np.sum(x400)
                / total_samples
                * 100
            )

            # -----------------------------------------------
            # Number of affected channels
            # -----------------------------------------------

            channel_max = np.max(
                absdata[ep],
                axis=1
            )

            channels200 = int(
                np.sum(channel_max > 200)
            )

            channels300 = int(
                np.sum(channel_max > 300)
            )

            channels400 = int(
                np.sum(channel_max > 400)
            )

            # -----------------------------------------------
            # Temporal duration
            #
            # For each time point:
            # if ANY channel exceeds threshold,
            # that time point is considered contaminated.
            # -----------------------------------------------

            time200 = np.any(
                mask200[ep],
                axis=0
            )

            time300 = np.any(
                mask300[ep],
                axis=0
            )

            time400 = np.any(
                mask400[ep],
                axis=0
            )

            time_pct200 = (
                np.mean(time200) * 100
            )

            time_pct300 = (
                np.mean(time300) * 100
            )

            time_pct400 = (
                np.mean(time400) * 100
            )

            duration200_ms = (
                np.sum(time200) * dt_ms
            )

            duration300_ms = (
                np.sum(time300) * dt_ms
            )

            duration400_ms = (
                np.sum(time400) * dt_ms
            )

            # -----------------------------------------------
            # Longest continuous contaminated segment
            # -----------------------------------------------

            def longest_run(mask):

                if not np.any(mask):
                    return 0

                padded = np.concatenate(
                    ([False], mask, [False])
                )

                diff = np.diff(
                    padded.astype(np.int8)
                )

                starts = np.where(
                    diff == 1
                )[0]

                ends = np.where(
                    diff == -1
                )[0]

                if len(starts) == 0:
                    return 0

                return int(
                    np.max(ends - starts)
                )

            longest200 = longest_run(
                time200
            )

            longest300 = longest_run(
                time300
            )

            longest400 = longest_run(
                time400
            )

            # milliseconds

            longest200_ms = (
                longest200 * dt_ms
            )

            longest300_ms = (
                longest300 * dt_ms
            )

            longest400_ms = (
                longest400 * dt_ms
            )

            # -----------------------------------------------
            # Maximum amplitude
            # -----------------------------------------------

            max_abs = float(
                np.max(absdata[ep])
            )

            # -----------------------------------------------
            # Mean amplitude
            # -----------------------------------------------

            mean_abs = float(
                np.mean(absdata[ep])
            )

            # -----------------------------------------------
            # Classification
            #
            # IMPORTANT:
            # This is ONLY a diagnostic classification.
            # NO epochs are removed.
            # -----------------------------------------------

            if (
                time_pct400 >= 20
                or longest400_ms >= 100
            ):
                diagnostic_class = "SEVERE"

            elif (
                time_pct300 >= 10
                or longest300_ms >= 100
            ):
                diagnostic_class = "HIGH"

            elif (
                time_pct200 >= 5
                or longest200_ms >= 100
            ):
                diagnostic_class = "MODERATE"

            elif (
                time_pct200 > 0
            ):
                diagnostic_class = "BRIEF"

            else:
                diagnostic_class = "CLEAN"

            records.append({

                "file": fname,

                "epoch": ep + 1,

                "max_abs": max_abs,

                "mean_abs": mean_abs,

                "pct_samples_gt200": pct200,

                "pct_samples_gt300": pct300,

                "pct_samples_gt400": pct400,

                "channels_gt200": channels200,

                "channels_gt300": channels300,

                "channels_gt400": channels400,

                "time_pct_gt200": time_pct200,

                "time_pct_gt300": time_pct300,

                "time_pct_gt400": time_pct400,

                "duration_gt200_ms": duration200_ms,

                "duration_gt300_ms": duration300_ms,

                "duration_gt400_ms": duration400_ms,

                "longest_gt200_ms": longest200_ms,

                "longest_gt300_ms": longest300_ms,

                "longest_gt400_ms": longest400_ms,

                "diagnostic_class": diagnostic_class

            })

        # ----------------------------------------------------
        # File summary
        # ----------------------------------------------------

        current = pd.DataFrame(
            records
        )

        current = current[
            current["file"] == fname
        ]

        print(
            f"Epochs: {len(current)}"
        )

        print(
            current[
                "diagnostic_class"
            ].value_counts().to_string()
        )

    except Exception as e:

        print("")
        print(f"ERROR: {e}")

# ============================================================
# SAVE CSV
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
    "EPOCH ARTIFACT DURATION QC - 82 RUNS"
)
summary.append("=" * 80)

summary.append("")
summary.append(
    f"Files found: {len(files)}"
)

summary.append(
    f"Epoch records: {len(df)}"
)

summary.append("")

summary.append("=" * 80)
summary.append("DIAGNOSTIC CLASS COUNTS")
summary.append("=" * 80)

summary.append(
    df["diagnostic_class"]
    .value_counts()
    .to_string()
)

summary.append("")

# ------------------------------------------------------------
# Distribution
# ------------------------------------------------------------

summary.append("=" * 80)
summary.append("ARTIFACT METRIC DISTRIBUTION")
summary.append("=" * 80)

metrics = [
    "max_abs",
    "pct_samples_gt200",
    "pct_samples_gt300",
    "pct_samples_gt400",
    "time_pct_gt200",
    "time_pct_gt300",
    "time_pct_gt400",
    "duration_gt200_ms",
    "duration_gt300_ms",
    "duration_gt400_ms",
    "longest_gt200_ms",
    "longest_gt300_ms",
    "longest_gt400_ms",
]

summary.append(
    df[metrics].describe().to_string()
)

summary.append("")

# ------------------------------------------------------------
# Most problematic files
# ------------------------------------------------------------

summary.append("=" * 80)
summary.append(
    "FILES WITH MOST SEVERE ARTIFACTS"
)
summary.append("=" * 80)

file_stats = (
    df.groupby("file")
    .agg(
        epochs=("epoch", "count"),
        severe=(
            "diagnostic_class",
            lambda x: np.sum(
                x == "SEVERE"
            )
        ),
        high=(
            "diagnostic_class",
            lambda x: np.sum(
                x == "HIGH"
            )
        ),
        moderate=(
            "diagnostic_class",
            lambda x: np.sum(
                x == "MODERATE"
            )
        ),
        brief=(
            "diagnostic_class",
            lambda x: np.sum(
                x == "BRIEF"
            )
        ),
        clean=(
            "diagnostic_class",
            lambda x: np.sum(
                x == "CLEAN"
            )
        ),
        max_amplitude=(
            "max_abs",
            "max"
        ),
        median_time_gt200=(
            "time_pct_gt200",
            "median"
        ),
        max_time_gt200=(
            "time_pct_gt200",
            "max"
        ),
        max_longest_gt200=(
            "longest_gt200_ms",
            "max"
        )
    )
)

file_stats = file_stats.sort_values(
    [
        "severe",
        "high",
        "max_longest_gt200"
    ],
    ascending=False
)

summary.append(
    file_stats.to_string()
)

summary.append("")

# ------------------------------------------------------------
# Top 100 worst epochs
# ------------------------------------------------------------

summary.append("=" * 80)
summary.append(
    "TOP 100 WORST EPOCHS"
)
summary.append("=" * 80)

worst = df.sort_values(
    [
        "longest_gt400_ms",
        "longest_gt300_ms",
        "longest_gt200_ms",
        "max_abs"
    ],
    ascending=False
).head(100)

summary.append(
    worst[
        [
            "file",
            "epoch",
            "max_abs",
            "time_pct_gt200",
            "time_pct_gt300",
            "time_pct_gt400",
            "longest_gt200_ms",
            "longest_gt300_ms",
            "longest_gt400_ms",
            "channels_gt200",
            "channels_gt300",
            "diagnostic_class"
        ]
    ].to_string(index=False)
)

summary.append("")

summary.append("=" * 80)
summary.append("IMPORTANT")
summary.append("=" * 80)

summary.append(
    "THIS SCRIPT IS READ-ONLY."
)

summary.append(
    "NO EPOCHS WERE REMOVED."
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

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )

# ============================================================
# FINAL
# ============================================================

print("")
print("=" * 80)
print(
    "ARTIFACT DURATION QC COMPLETE"
)
print("=" * 80)

print("")
print("DIAGNOSTIC CLASS COUNTS")

print(
    df["diagnostic_class"]
    .value_counts()
)

print("")
print("Saved:")
print(CSV_OUT)
print(SUMMARY_OUT)

print("")
print("=" * 80)
print("NO DATA WAS MODIFIED.")
print("=" * 80)