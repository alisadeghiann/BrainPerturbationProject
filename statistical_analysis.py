import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

PROJECT_DIR = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

PROCESSED = PROJECT_DIR / "processed"

RUNS = [1, 2]

BANDS = [
    "theta",
    "alpha",
    "beta"
]

print("=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

all_results = {}


for run in RUNS:

    print("\n" + "=" * 70)
    print(f"RUN {run}")
    print("=" * 70)

    psd_file = (
        PROCESSED /
        f"sub-009_ses-01_task-WorkingMemory_run-{run}_psd.npz"
    )

    print("Loading PSD:")
    print(psd_file)

    data = np.load(psd_file)

    print("Available keys:")
    print(data.files)

    run_results = {}

    for band in BANDS:

        print("\n" + "-" * 60)
        print(band.upper())
        print("-" * 60)

        remember_key = f"{band}_remember"
        ignore_key = f"{band}_ignore"

        remember = data[remember_key]
        ignore = data[ignore_key]

        print(
            "Remember shape:",
            remember.shape
        )

        print(
            "Ignore shape:",
            ignore.shape
        )

        n_channels = remember.shape[1]

        t_values = np.zeros(n_channels)
        p_values = np.zeros(n_channels)
        effect_sizes = np.zeros(n_channels)

        # ----------------------------------------------------
        # CHANNEL-WISE STATISTICAL TEST
        # ----------------------------------------------------

        for ch in range(n_channels):

            remember_ch = remember[:, ch]
            ignore_ch = ignore[:, ch]

            t, p = ttest_ind(
                remember_ch,
                ignore_ch,
                equal_var=False,
                nan_policy="omit"
            )

            t_values[ch] = t
            p_values[ch] = p

            # Cohen's d
            n1 = len(remember_ch)
            n2 = len(ignore_ch)

            var1 = np.var(
                remember_ch,
                ddof=1
            )

            var2 = np.var(
                ignore_ch,
                ddof=1
            )

            pooled_sd = np.sqrt(
                (
                    (n1 - 1) * var1
                    +
                    (n2 - 1) * var2
                )
                /
                (
                    n1 + n2 - 2
                )
            )

            if pooled_sd > 0:

                effect_sizes[ch] = (
                    np.mean(remember_ch)
                    -
                    np.mean(ignore_ch)
                ) / pooled_sd

            else:

                effect_sizes[ch] = 0


        # ----------------------------------------------------
        # FDR CORRECTION
        # ----------------------------------------------------

        reject, p_fdr, _, _ = multipletests(
            p_values,
            alpha=0.05,
            method="fdr_bh"
        )

        # ----------------------------------------------------
        # CHANNEL NAMES
        # ----------------------------------------------------

        channel_names = [
            "FP1", "FPZ", "FP2",
            "AF7", "AF3", "AFZ", "AF4", "AF8",
            "F9", "F7", "F5", "F3", "F1", "FZ",
            "F2", "F4", "F6", "F8", "F10",
            "FT9", "FT7", "FC5", "FC3", "FC1",
            "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
            "T7", "C5", "C3", "C1", "CZ", "C2",
            "C4", "C6", "T8",
            "TP9", "TP7", "CP5", "CP3", "CP1",
            "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
            "P7", "P5", "P3", "P1", "PZ",
            "P2", "P4", "P6", "P8",
            "PO9", "PO7", "PO3", "POZ", "PO4",
            "PO8", "PO10",
            "O1", "OZ", "O2"
        ]

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        significant = np.where(reject)[0]

        print(
            "\nChannels surviving FDR:",
            len(significant)
        )

        if len(significant) == 0:

            print(
                "No channels survived FDR correction."
            )

        else:

            print(
                "\nSignificant channels:"
            )

            sorted_indices = significant[
                np.argsort(p_fdr[significant])
            ]

            for ch in sorted_indices:

                print(
                    f"{channel_names[ch]:>5} "
                    f"t={t_values[ch]:8.3f} "
                    f"p={p_values[ch]:.6f} "
                    f"FDR={p_fdr[ch]:.6f} "
                    f"d={effect_sizes[ch]:7.3f}"
                )


        # ----------------------------------------------------
        # TOP EFFECTS
        # ----------------------------------------------------

        print(
            "\nTop 10 absolute effects:"
        )

        top_effects = np.argsort(
            np.abs(effect_sizes)
        )[::-1][:10]

        for ch in top_effects:

            print(
                f"{channel_names[ch]:>5} "
                f"d={effect_sizes[ch]:7.3f} "
                f"p={p_values[ch]:.6f} "
                f"FDR={p_fdr[ch]:.6f}"
            )


        run_results[band] = {
            "t": t_values,
            "p": p_values,
            "p_fdr": p_fdr,
            "effect": effect_sizes,
            "reject": reject
        }


    all_results[run] = run_results


# ============================================================
# SAVE STATISTICAL RESULTS
# ============================================================

output_file = (
    PROCESSED /
    "sub-009_statistical_results.npz"
)

save_data = {}

for run in RUNS:

    for band in BANDS:

        result = all_results[run][band]

        save_data[
            f"run{run}_{band}_t"
        ] = result["t"]

        save_data[
            f"run{run}_{band}_p"
        ] = result["p"]

        save_data[
            f"run{run}_{band}_p_fdr"
        ] = result["p_fdr"]

        save_data[
            f"run{run}_{band}_effect"
        ] = result["effect"]

        save_data[
            f"run{run}_{band}_reject"
        ] = result["reject"]


np.savez(
    output_file,
    **save_data
)


print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nResults saved:"
)

print(output_file)

print("\n[DONE]")