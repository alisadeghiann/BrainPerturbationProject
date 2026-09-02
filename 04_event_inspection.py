import h5py
import numpy as np

FILE = r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-001\ses-01\eeg\sub-001_ses-01_task-WorkingMemory_run-1_eeg.set"


def decode_matlab_string(f, ref):
    """
    Decode MATLAB/EEGLAB HDF5 string references.
    """
    try:
        obj = f[ref]

        if isinstance(obj, h5py.Dataset):
            data = obj[()]

            if data.dtype.kind in ["u", "i"]:
                return "".join(chr(int(x)) for x in data.flatten())

            return str(data)

        return str(obj)

    except Exception as e:
        return f"<ERROR: {e}>"


with h5py.File(FILE, "r") as f:

    print("=" * 80)
    print("EVENT INSPECTION")
    print("=" * 80)

    print("\nFILE:")
    print(FILE)

    print("\nTOP LEVEL KEYS:")
    print(list(f.keys()))

    # ---------------------------------------------------------
    # EVENT GROUP
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("EVENT GROUP")
    print("=" * 80)

    event = f["event"]

    print("Event object:")
    print(event)

    print("\nEvent members:")

    for key in event.keys():

        obj = event[key]

        if isinstance(obj, h5py.Dataset):
            print(
                f"DATASET: /event/{key} | "
                f"shape={obj.shape} | "
                f"dtype={obj.dtype}"
            )

        elif isinstance(obj, h5py.Group):
            print(
                f"GROUP: /event/{key} | "
                f"members={list(obj.keys())}"
            )

    # ---------------------------------------------------------
    # EVENT DESCRIPTION
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("EVENT DESCRIPTION")
    print("=" * 80)

    if "eventdescription" in f:

        obj = f["eventdescription"]

        print("eventdescription:")
        print(obj)

        print("shape:", obj.shape)
        print("dtype:", obj.dtype)

        try:
            raw = obj[()]
            print("\nRaw eventdescription:")
            print(raw)
        except Exception as e:
            print("Could not read:", e)

    # ---------------------------------------------------------
    # EVENT REFERENCES
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("EVENT DATASETS")
    print("=" * 80)

    for key in event.keys():

        obj = event[key]

        if isinstance(obj, h5py.Dataset):

            print("\n" + "-" * 60)
            print("EVENT FIELD:", key)
            print("Shape:", obj.shape)
            print("Dtype:", obj.dtype)

            try:
                data = obj[()]

                print("Raw data:")
                print(data[:20])

            except Exception as e:
                print("Could not read:", e)

    # ---------------------------------------------------------
    # REFERENCE INFORMATION
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("REFERENCE OBJECTS")
    print("=" * 80)

    if "#refs#" in f:

        refs = f["#refs#"]

        print("Number of reference objects:", len(refs))

        for key in list(refs.keys())[:30]:

            obj = refs[key]

            print(
                f"{key} | "
                f"type={type(obj).__name__}"
            )

    print("\n" + "=" * 80)
    print("EVENT INSPECTION COMPLETE")
    print("=" * 80)