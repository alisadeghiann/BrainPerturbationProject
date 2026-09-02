from pathlib import Path
import numpy as np
import h5py

DATA = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\data"
)

TARGETS = {
    "sub-004": [3, 4],
    "sub-008": [1],
    "sub-009": [3],
    "sub-015": [1],
    "sub-018": [1],
    "sub-024": [3],
}


def scalar(h5, key):
    return int(np.asarray(h5[key][()]).squeeze())


for subject, runs in TARGETS.items():

    for run in runs:

        files = list(
            (
                DATA
                / subject
                / "ses-01"
                / "eeg"
            ).glob(
                f"*run-{run}_eeg.set"
            )
        )

        if not files:
            print(
                f"NOT FOUND: {subject} run-{run}"
            )
            continue

        set_file = files[0]
        fdt_file = set_file.with_suffix(".fdt")

        print("\n" + "=" * 70)
        print(subject, f"RUN {run}")
        print("=" * 70)

        with h5py.File(
            set_file,
            "r"
        ) as h5:

            n_channels = scalar(
                h5,
                "nbchan"
            )

            n_samples = scalar(
                h5,
                "pnts"
            )

            srate = float(
                np.asarray(
                    h5["srate"][()]
                ).squeeze()
            )

        data = np.fromfile(
            fdt_file,
            dtype="<f4"
        ).reshape(
            (n_channels, n_samples),
            order="F"
        )

        std = np.std(
            data,
            axis=1
        )

        median_std = np.median(std)

        high = np.where(
            std > median_std * 10
        )[0]

        low = np.where(
            std < median_std * 0.05
        )[0]

        print(
            "Sampling rate:",
            srate,
            "Hz"
        )

        print(
            "Median channel STD:",
            round(median_std, 3)
        )

        print(
            "HIGH variance channels:",
            (high + 1).tolist()
        )

        print(
            "HIGH variance STD:",
            np.round(
                std[high],
                3
            ).tolist()
        )

        print(
            "LOW variance channels:",
            (low + 1).tolist()
        )