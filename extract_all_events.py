from pathlib import Path
import h5py
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

DATA_DIR = PROJECT_ROOT / "data"

OUT_DIR = PROJECT_ROOT / "qc" / "events"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HDF5 / MATLAB v7.3 DECODING
# ============================================================

def decode_reference(f, ref):
    """
    Resolve an HDF5 object reference and decode its content.
    """

    if not ref:
        return None

    try:
        obj = f[ref]
    except Exception:
        return None

    try:
        data = obj[()]
    except Exception:
        return None

    return decode_value(f, data)


def decode_value(f, value):
    """
    Recursively decode MATLAB v7.3 values.
    Handles:
      - strings
      - numeric scalars
      - arrays
      - HDF5 object references
    """

    # --------------------------------------------------------
    # HDF5 reference
    # --------------------------------------------------------

    if isinstance(value, h5py.h5r.Reference):
        return decode_reference(f, value)

    # --------------------------------------------------------
    # numpy array
    # --------------------------------------------------------

    if isinstance(value, np.ndarray):

        # Empty
        if value.size == 0:
            return None

        # Object/reference array
        if value.dtype == object:

            decoded = []

            for item in value.flatten():
                decoded.append(
                    decode_value(f, item)
                )

            if len(decoded) == 1:
                return decoded[0]

            return decoded

        # Numeric array
        arr = np.asarray(value).squeeze()

        if arr.size == 0:
            return None

        # MATLAB uint16 character array
        if arr.dtype.kind in ("u", "i"):

            flat = arr.flatten()

            # Detect printable character data
            if np.all(
                (flat >= 0) &
                (flat <= 127)
            ):

                chars = []

                for x in flat:
                    x = int(x)

                    if x == 0:
                        continue

                    chars.append(chr(x))

                text = "".join(chars)

                # If it looks like text, return text
                if text:
                    return text

        # Scalar
        if arr.size == 1:
            return arr.item()

        return arr.tolist()

    # --------------------------------------------------------
    # numpy scalar
    # --------------------------------------------------------

    if isinstance(value, np.generic):
        return value.item()

    # --------------------------------------------------------
    # Python scalar
    # --------------------------------------------------------

    return value


# ============================================================
# SPECIAL MATLAB STRING DECODER
# ============================================================

def decode_matlab_string(f, dataset):

    """
    Decode MATLAB v7.3 strings stored as object references.
    """

    raw = dataset[()]

    # Direct uint16 string
    if isinstance(raw, np.ndarray):

        if raw.dtype.kind in ("u", "i"):

            arr = raw.squeeze().flatten()

            chars = [
                chr(int(x))
                for x in arr
                if int(x) != 0
            ]

            return "".join(chars)

    # Reference
    if raw.dtype == object:

        values = []

        for ref in raw.flatten():

            value = decode_reference(
                f,
                ref
            )

            values.append(value)

        if len(values) == 1:
            return values[0]

        return values

    return decode_value(f, raw)


# ============================================================
# FIND EVENT GROUP
# ============================================================

def find_event_group(f):

    if "event" in f:
        return "event"

    # Some files may use another structure
    possible = [
        key for key in f.keys()
        if "event" in key.lower()
    ]

    if possible:
        return possible[0]

    return None


# ============================================================
# READ EVENT FIELD
# ============================================================

def read_event_field(f, event_group, field):

    path = f"{event_group}/{field}"

    if path not in f:
        return None

    dataset = f[path]

    raw = dataset[()]

    # --------------------------------------------------------
    # Direct numeric data
    # --------------------------------------------------------

    if isinstance(raw, np.ndarray):

        if raw.dtype != object:

            arr = np.asarray(raw).squeeze()

            if arr.size == 1:
                return [arr.item()]

            return arr.tolist()

    # --------------------------------------------------------
    # Object/reference data
    # --------------------------------------------------------

    values = []

    for item in raw.flatten():

        value = decode_reference(
            f,
            item
        )

        values.append(value)

    return values


# ============================================================
# NORMALIZE LIST LENGTH
# ============================================================

def normalize_length(values, n):

    if values is None:

        return [None] * n

    values = list(values)

    if len(values) == n:
        return values

    if len(values) == 1 and n > 1:
        return values * n

    if len(values) < n:
        return values + [None] * (
            n - len(values)
        )

    return values[:n]


# ============================================================
# CONVERT TO SIMPLE SCALAR
# ============================================================

def simplify(value):

    if value is None:
        return None

    # numpy scalar
    if isinstance(value, np.generic):
        return value.item()

    # single-element list
    if isinstance(value, list):

        if len(value) == 0:
            return None

        if len(value) == 1:
            return simplify(value[0])

        return str(value)

    return value


# ============================================================
# EXTRACT ONE FILE
# ============================================================

def extract_events(set_path):

    print()
    print("=" * 80)
    print("PROCESSING")
    print(set_path.name)
    print("=" * 80)

    with h5py.File(
        set_path,
        "r"
    ) as f:

        event_group = find_event_group(f)

        if event_group is None:

            raise RuntimeError(
                "EVENT group not found"
            )

        # ----------------------------------------------------
        # AVAILABLE FIELDS
        # ----------------------------------------------------

        group = f[event_group]

        fields = list(group.keys())

        print("Event fields:")
        print(fields)

        # ----------------------------------------------------
        # DETERMINE NUMBER OF EVENTS
        # ----------------------------------------------------

        n_events = 0

        for field in fields:

            obj = group[field]

            try:
                shape = obj.shape

                if shape and shape[0] > n_events:
                    n_events = shape[0]

            except Exception:
                pass

        if n_events == 0:

            raise RuntimeError(
                "No events detected"
            )

        print(
            f"Number of events: {n_events}"
        )

        # ----------------------------------------------------
        # READ ALL FIELDS
        # ----------------------------------------------------

        event_data = {}

        for field in fields:

            print(
                f"Decoding field: {field}"
            )

            values = read_event_field(
                f,
                event_group,
                field
            )

            values = normalize_length(
                values,
                n_events
            )

            values = [
                simplify(x)
                for x in values
            ]

            event_data[field] = values

    # ========================================================
    # BUILD DATAFRAME
    # ========================================================

    df = pd.DataFrame(event_data)

    # ========================================================
    # ADD FILE INFORMATION
    # ========================================================

    # BIDS-style extraction
    name = set_path.name

    subject = None
    session = None
    run = None

    parts = name.split("_")

    for part in parts:

        if part.startswith("sub-"):
            subject = part

        elif part.startswith("ses-"):
            session = part

        elif part.startswith("run-"):
            run = part.replace(
                ".set",
                ""
            )

    df.insert(
        0,
        "subject",
        subject
    )

    df.insert(
        1,
        "session",
        session
    )

    df.insert(
        2,
        "run",
        run
    )

    df.insert(
        3,
        "source_file",
        name
    )

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("EEG EVENT EXTRACTION")
    print("=" * 80)

    set_files = sorted(
        DATA_DIR.rglob("*_eeg.set")
    )

    print(
        f"SET files found: {len(set_files)}"
    )

    print()

    all_events = []

    file_summary = []

    errors = []

    # ========================================================
    # PROCESS ALL FILES
    # ========================================================

    for i, set_path in enumerate(
        set_files,
        1
    ):

        print()
        print(
            f"PROCESSING {i}/{len(set_files)}"
        )

        try:

            df = extract_events(
                set_path
            )

            # ------------------------------------------------
            # Save individual event table
            # ------------------------------------------------

            individual_name = (
                set_path.stem
                + "_events.csv"
            )

            individual_path = (
                OUT_DIR /
                individual_name
            )

            df.to_csv(
                individual_path,
                index=False,
                encoding="utf-8-sig"
            )

            all_events.append(df)

            # ------------------------------------------------
            # Summary
            # ------------------------------------------------

            file_summary.append({

                "subject":
                    df["subject"].iloc[0],

                "session":
                    df["session"].iloc[0],

                "run":
                    df["run"].iloc[0],

                "file":
                    set_path.name,

                "events":
                    len(df),

                "status":
                    "OK"

            })

            print()
            print(
                f"SUCCESS: {len(df)} events"
            )

        except Exception as e:

            print()
            print(
                "ERROR:",
                str(e)
            )

            errors.append({

                "file":
                    set_path.name,

                "error":
                    str(e)

            })

            file_summary.append({

                "subject":
                    None,

                "session":
                    None,

                "run":
                    None,

                "file":
                    set_path.name,

                "events":
                    0,

                "status":
                    "ERROR"

            })

    # ========================================================
    # COMBINE ALL EVENTS
    # ========================================================

    if all_events:

        combined = pd.concat(
            all_events,
            ignore_index=True
        )

    else:

        combined = pd.DataFrame()

    combined_path = (
        OUT_DIR /
        "ALL_EVENTS_83_RUNS.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        file_summary
    )

    summary_path = (
        OUT_DIR /
        "EVENT_EXTRACTION_SUMMARY.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # ERROR REPORT
    # ========================================================

    error_df = pd.DataFrame(
        errors
    )

    error_path = (
        OUT_DIR /
        "EVENT_EXTRACTION_ERRORS.csv"
    )

    error_df.to_csv(
        error_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL EVENT EXTRACTION SUMMARY")
    print("=" * 80)

    print()

    print(
        "Total SET files:",
        len(set_files)
    )

    print()

    print("STATUS:")

    if not summary_df.empty:

        print(
            summary_df[
                "status"
            ].value_counts()
        )

    print()

    print("TOTAL EVENTS:")

    if not combined.empty:

        print(
            len(combined)
        )

    else:

        print(0)

    print()

    print("EVENT TYPES:")

    if (
        not combined.empty
        and "type" in combined.columns
    ):

        print(
            combined[
                "type"
            ].value_counts(
                dropna=False
            ).head(30)
        )

    print()

    print("TASK ROLES:")

    if (
        not combined.empty
        and "task_role" in combined.columns
    ):

        print(
            combined[
                "task_role"
            ].value_counts(
                dropna=False
            )
        )

    print()

    print("MEMORY CONDITIONS:")

    if (
        not combined.empty
        and "memory_cond" in combined.columns
    ):

        print(
            combined[
                "memory_cond"
            ].value_counts(
                dropna=False
            )
        )

    print()

    print("EVENT VALUE TYPES:")

    if (
        not combined.empty
        and "value" in combined.columns
    ):

        print(
            combined[
                "value"
            ].value_counts(
                dropna=False
            ).head(50)
        )

    print()

    print("ERROR FILES:")

    if len(error_df) == 0:

        print("None")

    else:

        print(
            error_df.to_string(
                index=False
            )
        )

    print()
    print("OUTPUT DIRECTORY:")
    print(OUT_DIR)

    print()
    print("COMBINED EVENTS:")
    print(combined_path)

    print()
    print("SUMMARY:")
    print(summary_path)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)