# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import mne
from pathlib import Path

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

# مسیر واقعی فایل‌های نهایی
INPUT_DIR = BASE / "final_dataset" / "perturbation"

# اگر مسیر بالا فایل نداشت، کل پروژه را برای FIF جستجو می‌کنیم
files = sorted(INPUT_DIR.glob("*_final_epo.fif"))

if len(files) == 0:
    print("=" * 80)
    print("SEARCHING FOR FINAL EPOCH FILES")
    print("=" * 80)

    files = sorted(
        BASE.rglob("*_final_epo.fif")
    )

OUTPUT_DIR = BASE / "features" / "scientific_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "scientific_features_v2.csv"

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def get_group_indices(ch_names, prefixes):

    result = []

    for i, ch in enumerate(ch_names):

        name = ch.upper()

        if any(name.startswith(p) for p in prefixes):
            result.append(i)

    return result


def safe_mean(x):

    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return np.nan

    return float(np.nanmean(x))


def safe_ratio(a, b):

    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan

    if abs(b) < 1e-12:
        return np.nan

    return float(a / b)


def band_power(data, sfreq, low, high):

    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=low,
        fmax=high,
        n_fft=min(256, data.shape[-1]),
        n_overlap=0,
        verbose=False
    )

    return np.mean(psd, axis=-1)


print("=" * 80)
print("SCIENTIFIC EEG FEATURE ENGINEERING V2")
print("=" * 80)

print(f"Files found: {len(files)}")

if len(files) == 0:
    raise RuntimeError(
        "No *_final_epo.fif files found anywhere inside the project."
    )


rows = []

groups_printed = False


for file_idx, fif_file in enumerate(files, start=1):

    print(
        f"[{file_idx}/{len(files)}] "
        f"{fif_file.name}"
    )

    epochs = mne.read_epochs(
        fif_file,
        preload=True,
        verbose=False
    )

    data = epochs.get_data()

    sfreq = float(
        epochs.info["sfreq"]
    )

    ch_names = list(
        epochs.ch_names
    )

    n_epochs = data.shape[0]
    n_channels = data.shape[1]
    n_times = data.shape[2]

    frontal = get_group_indices(
        ch_names,
        ["AF", "F", "FP"]
    )

    central = get_group_indices(
        ch_names,
        ["C"]
    )

    parietal = get_group_indices(
        ch_names,
        ["P"]
    )

    occipital = get_group_indices(
        ch_names,
        ["O"]
    )

    temporal = get_group_indices(
        ch_names,
        ["T"]
    )

    if not groups_printed:

        print()
        print("CHANNEL GROUPS")
        print("-" * 80)

        print("Frontal:", len(frontal))
        print("Central:", len(central))
        print("Parietal:", len(parietal))
        print("Occipital:", len(occipital))
        print("Temporal:", len(temporal))

        groups_printed = True


    powers = {}

    for band, (low, high) in BANDS.items():

        powers[band] = band_power(
            data,
            sfreq,
            low,
            high
        )


    for epoch_idx in range(n_epochs):

        row = {
            "file": fif_file.name,
            "subject": (
                fif_file.name
                .split("_")[0]
            ),
            "run": next(
                (
                    x
                    for x in fif_file.name.split("_")
                    if x.startswith("run-")
                ),
                None
            ),
            "epoch": epoch_idx,
            "sfreq": sfreq,
            "n_channels": n_channels,
            "n_timepoints": n_times,
        }


        # ====================================================
        # GLOBAL POWER
        # ====================================================

        global_power = {}

        for band in BANDS:

            value = safe_mean(
                powers[band][epoch_idx]
            )

            global_power[band] = value

            row[f"{band}_abs"] = value


        total_power = sum(
            v
            for v in global_power.values()
            if np.isfinite(v)
        )


        for band in BANDS:

            row[f"{band}_rel"] = safe_ratio(
                global_power[band],
                total_power
            )


        # ====================================================
        # GLOBAL RATIOS
        # ====================================================

        row["theta_alpha_ratio"] = safe_ratio(
            global_power["theta"],
            global_power["alpha"]
        )

        row["theta_beta_ratio"] = safe_ratio(
            global_power["theta"],
            global_power["beta"]
        )

        row["alpha_beta_ratio"] = safe_ratio(
            global_power["alpha"],
            global_power["beta"]
        )

        row["delta_theta_ratio"] = safe_ratio(
            global_power["delta"],
            global_power["theta"]
        )

        row["theta_alpha_beta_ratio"] = safe_ratio(
            global_power["theta"],
            global_power["alpha"]
            + global_power["beta"]
        )


        # ====================================================
        # REGIONAL POWER
        # ====================================================

        regions = {
            "frontal": frontal,
            "central": central,
            "parietal": parietal,
            "occipital": occipital,
            "temporal": temporal,
        }


        regional = {}


        for region_name, indices in regions.items():

            if len(indices) == 0:
                continue

            regional[region_name] = {}

            for band in BANDS:

                value = safe_mean(
                    powers[band][
                        epoch_idx,
                        indices
                    ]
                )

                regional[
                    region_name
                ][band] = value

                row[
                    f"{band}_{region_name}"
                ] = value


        # ====================================================
        # REGIONAL RATIOS
        # ====================================================

        for region_name, values in regional.items():

            row[
                f"theta_alpha_{region_name}_ratio"
            ] = safe_ratio(
                values["theta"],
                values["alpha"]
            )

            row[
                f"alpha_beta_{region_name}_ratio"
            ] = safe_ratio(
                values["alpha"],
                values["beta"]
            )


        # ====================================================
        # FRONTAL / PARIETAL
        # ====================================================

        if (
            "frontal" in regional
            and "parietal" in regional
        ):

            for band in BANDS:

                f = regional["frontal"][band]
                p = regional["parietal"][band]

                if (
                    np.isfinite(f)
                    and np.isfinite(p)
                ):

                    row[
                        f"{band}_frontoparietal_diff"
                    ] = f - p

                    row[
                        f"{band}_frontoparietal_ratio"
                    ] = safe_ratio(f, p)

                else:

                    row[
                        f"{band}_frontoparietal_diff"
                    ] = np.nan

                    row[
                        f"{band}_frontoparietal_ratio"
                    ] = np.nan


        rows.append(row)


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


id_cols = [
    "file",
    "subject",
    "run",
    "epoch"
]

numeric_cols = df.select_dtypes(
    include=[np.number]
).columns


# ============================================================
# NUMERIC CLEANING
# ============================================================

df[numeric_cols] = df[
    numeric_cols
].replace(
    [np.inf, -np.inf],
    np.nan
)


nan_count = int(
    df[numeric_cols]
    .isna()
    .sum()
    .sum()
)


inf_count = int(
    np.isinf(
        df[numeric_cols]
        .to_numpy(dtype=float)
    ).sum()
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_keys = int(
    df.duplicated(
        subset=[
            "file",
            "epoch"
        ]
    ).sum()
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("SCIENTIFIC FEATURE ENGINEERING V2 COMPLETE")
print("=" * 80)

print(f"Rows:              {len(df):,}")
print(f"Columns:           {len(df.columns):,}")
print(f"Numeric columns:   {len(numeric_cols):,}")
print(f"Subjects:          {df['subject'].nunique():,}")
print(f"Runs:              {df['run'].nunique():,}")
print(f"NaN values:        {nan_count:,}")
print(f"Inf values:        {inf_count:,}")
print(f"Duplicate keys:    {duplicate_keys:,}")


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_FILE)

print()
print("READ-ONLY INPUT")
print("No previous files modified.")

print()
print("=" * 80)

if (
    len(df) > 0
    and duplicate_keys == 0
    and inf_count == 0
):

    print(
        "STATUS: PASS - SCIENTIFIC V2 FEATURES CREATED"
    )

else:

    print(
        "STATUS: REVIEW REQUIRED"
    )

print("=" * 80)