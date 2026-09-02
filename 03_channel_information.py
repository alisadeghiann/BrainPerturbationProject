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
# FIND FIRST SET FILE
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

if len(set_files) == 0:
    raise FileNotFoundError(
        "No .set files found."
    )


set_file = set_files[0]


# ============================================================
# HDF5 REFERENCE READER
# ============================================================

def resolve_hdf5_reference(
    h5file,
    value
):

    """
    Resolve an HDF5 object reference and
    try to extract its numerical or text value.
    """

    try:

        # ----------------------------------------------------
        # HDF5 object reference
        # ----------------------------------------------------

        if isinstance(
            value,
            h5py.Reference
        ):

            if not value:

                return None

            obj = h5file[
                value
            ]

        else:

            return value

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        if isinstance(
            obj,
            h5py.Dataset
        ):

            data = np.array(
                obj
            )

            # -----------------------------------------------
            # Numeric scalar
            # -----------------------------------------------

            if data.size == 1:

                value = data.flatten()[0]

                if np.issubdtype(
                    data.dtype,
                    np.number
                ):

                    return float(
                        value
                    )

            # -----------------------------------------------
            # Character data
            # -----------------------------------------------

            if np.issubdtype(
                data.dtype,
                np.integer
            ):

                chars = []

                for x in data.flatten():

                    x = int(x)

                    if x != 0:

                        chars.append(
                            chr(x)
                        )

                text = "".join(
                    chars
                )

                if text:

                    return text

            # -----------------------------------------------
            # Object references inside dataset
            # -----------------------------------------------

            if data.dtype == object:

                values = []

                for x in data.flatten():

                    result = resolve_hdf5_reference(
                        h5file,
                        x
                    )

                    if result is not None:

                        values.append(
                            result
                        )

                if len(values) == 1:

                    return values[0]

                return values

        # ----------------------------------------------------
        # Group
        # ----------------------------------------------------

        if isinstance(
            obj,
            h5py.Group
        ):

            return {
                key:
                resolve_hdf5_reference(
                    h5file,
                    obj[key]
                )
                for key in obj.keys()
            }

    except Exception:

        return None

    return None


# ============================================================
# READ CHANLOCS
# ============================================================

print()
print("=" * 80)

print(
    "EXTRACTING CHANNEL INFORMATION"
)

print("=" * 80)

print()
print(
    f"File:"
)

print(
    set_file
)


with h5py.File(
    set_file,
    "r"
) as f:

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    nbchan = int(
        np.array(
            f["nbchan"]
        ).squeeze()
    )

    srate = float(
        np.array(
            f["srate"]
        ).squeeze()
    )

    print()
    print(
        f"Channels: {nbchan}"
    )

    print(
        f"Sampling rate: {srate} Hz"
    )

    # ========================================================
    # CHANLOCS
    # ========================================================

    chanlocs = f[
        "chanlocs"
    ]

    labels_dataset = chanlocs[
        "labels"
    ]

    type_dataset = chanlocs[
        "type"
    ]

    X_dataset = chanlocs[
        "X"
    ]

    Y_dataset = chanlocs[
        "Y"
    ]

    Z_dataset = chanlocs[
        "Z"
    ]

    # ========================================================
    # EXTRACT DATA
    # ========================================================

    labels_raw = np.array(
        labels_dataset
    )

    types_raw = np.array(
        type_dataset
    )

    X_raw = np.array(
        X_dataset
    )

    Y_raw = np.array(
        Y_dataset
    )

    Z_raw = np.array(
        Z_dataset
    )

    print()
    print(
        "Labels raw shape:",
        labels_raw.shape
    )

    print(
        "Types raw shape:",
        types_raw.shape
    )

    # ========================================================
    # RESOLVE LABELS
    # ========================================================

    labels = []

    for i in range(
        nbchan
    ):

        try:

            value = labels_raw[
                i,
                0
            ]

        except Exception:

            value = labels_raw[
                i
            ]

        result = resolve_hdf5_reference(
            f,
            value
        )

        if result is None:

            result = (
                f"UNKNOWN_{i + 1}"
            )

        labels.append(
            str(result)
        )

    # ========================================================
    # RESOLVE TYPES
    # ========================================================

    channel_types = []

    for i in range(
        nbchan
    ):

        try:

            value = types_raw[
                i,
                0
            ]

        except Exception:

            value = types_raw[
                i
            ]

        result = resolve_hdf5_reference(
            f,
            value
        )

        if result is None:

            result = "UNKNOWN"

        channel_types.append(
            str(result)
        )

    # ========================================================
    # RESOLVE COORDINATES
    # ========================================================

    X = []

    Y = []

    Z = []

    for i in range(
        nbchan
    ):

        # ----------------------------------------------------
        # X
        # ----------------------------------------------------

        try:

            value = X_raw[
                i,
                0
            ]

        except Exception:

            value = X_raw[
                i
            ]

        result = resolve_hdf5_reference(
            f,
            value
        )

        X.append(
            result
        )

        # ----------------------------------------------------
        # Y
        # ----------------------------------------------------

        try:

            value = Y_raw[
                i,
                0
            ]

        except Exception:

            value = Y_raw[
                i
            ]

        result = resolve_hdf5_reference(
            f,
            value
        )

        Y.append(
            result
        )

        # ----------------------------------------------------
        # Z
        # ----------------------------------------------------

        try:

            value = Z_raw[
                i,
                0
            ]

        except Exception:

            value = Z_raw[
                i
            ]

        result = resolve_hdf5_reference(
            f,
            value
        )

        Z.append(
            result
        )


# ============================================================
# CREATE TABLE
# ============================================================

rows = []

for i in range(
    nbchan
):

    rows.append({

        "channel_index":
            i + 1,

        "channel_name":
            labels[i],

        "channel_type":
            channel_types[i],

        "X":
            X[i],

        "Y":
            Y[i],

        "Z":
            Z[i]

    })


df = pd.DataFrame(
    rows
)


# ============================================================
# SAVE CSV
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "channel_information.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# PRINT CHANNELS
# ============================================================

print()
print("=" * 80)

print(
    "CHANNEL INFORMATION"
)

print("=" * 80)

print()

for _, row in df.iterrows():

    print(
        f"{int(row['channel_index']):3d} "
        f"| "
        f"{str(row['channel_name']):15s} "
        f"| "
        f"{str(row['channel_type'])}"
    )


# ============================================================
# TYPE SUMMARY
# ============================================================

print()
print("=" * 80)

print(
    "CHANNEL TYPE SUMMARY"
)

print("=" * 80)

print()

print(
    df[
        "channel_type"
    ].value_counts().to_string()
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)

print(
    "CHANNEL INFORMATION COMPLETE"
)

print("=" * 80)

print()

print(
    "Saved to:"
)

print(
    output_file
)

print()
print(
    "DONE."
)