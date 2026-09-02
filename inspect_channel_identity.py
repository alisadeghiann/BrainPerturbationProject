from pathlib import Path
import h5py
import numpy as np


DATA = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject\data"
)

FILES = [
    ("sub-004", 3),
    ("sub-004", 4),
    ("sub-008", 1),
    ("sub-009", 3),
    ("sub-015", 1),
    ("sub-018", 1),
    ("sub-024", 3),
]


def decode_string(h5, ref):

    try:

        obj = h5[ref]

        arr = np.asarray(
            obj[()]
        ).squeeze()

        if arr.dtype.kind in "iu":

            return "".join(
                chr(int(x))
                for x in arr.flatten()
                if int(x) != 0
            )

        if arr.dtype.kind == "S":

            return b"".join(
                arr.flatten()
            ).decode(
                "utf-8",
                errors="ignore"
            )

        return str(arr)

    except Exception:

        return "UNKNOWN"


for subject, run in FILES:

    eeg_dir = (
        DATA
        / subject
        / "ses-01"
        / "eeg"
    )

    files = list(
        eeg_dir.glob(
            f"*run-{run}_eeg.set"
        )
    )

    if not files:

        print(
            f"NOT FOUND: {subject} run-{run}"
        )

        continue

    set_file = files[0]

    print("\n" + "=" * 70)
    print(subject, f"RUN {run}")
    print("=" * 70)

    with h5py.File(
        set_file,
        "r"
    ) as h5:

        labels = h5["chanlocs"]["labels"]
        types = h5["chanlocs"]["type"]

        print(
            "\nLAST 10 CHANNELS:"
        )

        for i in range(
            max(0, labels.shape[0] - 10),
            labels.shape[0]
        ):

            label = decode_string(
                h5,
                labels[i, 0]
            )

            ch_type = decode_string(
                h5,
                types[i, 0]
            )

            print(
                f"{i+1:3d} | "
                f"{label:15s} | "
                f"{ch_type}"
            )