import h5py
import numpy as np

FILE = r"C:\Users\Ali\Desktop\BrainPerturbationProject\data\sub-001\ses-01\eeg\sub-001_ses-01_task-WorkingMemory_run-1_eeg.set"


def decode_ref(f, ref):
    """Decode an HDF5 object reference."""

    try:
        obj = f[ref]

        if isinstance(obj, h5py.Dataset):
            value = obj[()]

            # MATLAB numeric scalar
            if np.isscalar(value):
                return value.item()

            value = np.asarray(value).flatten()

            # Character array
            if value.dtype.kind in ["u", "i"]:
                try:
                    chars = [chr(int(x)) for x in value]

                    # If all values look like ASCII
                    if all(0 <= ord(c) < 128 for c in chars):
                        return "".join(chars)
                except:
                    pass

            return value.tolist()

        return str(obj)

    except Exception as e:
        return f"ERROR: {e}"


with h5py.File(FILE, "r") as f:

    event = f["event"]

    print("=" * 100)
    print("DECODING EEG EVENTS")
    print("=" * 100)

    print("\nNumber of events:", event["type"].shape[0])

    fields = [
        "type",
        "value",
        "latency",
        "sample",
        "trial",
        "letter",
        "memory_cond",
        "task_role",
        "urevent"
    ]

    decoded = {}

    for field in fields:

        print("\n" + "=" * 80)
        print("FIELD:", field)
        print("=" * 80)

        raw = event[field][()]

        values = []

        for i, ref_array in enumerate(raw):

            ref = ref_array[0]

            value = decode_ref(f, ref)

            values.append(value)

        decoded[field] = values

        print("First 30 values:")

        for i, value in enumerate(values[:30]):
            print(f"{i+1:3d} | {value}")

        # Unique values for categorical fields
        if field in ["type", "value", "letter", "memory_cond", "task_role"]:

            print("\nUnique values:")

            unique = []

            for value in values:
                value_str = str(value)

                if value_str not in unique:
                    unique.append(value_str)

            for value in unique[:100]:
                print("  ", value)

    print("\n" + "=" * 100)
    print("EVENT DECODING COMPLETE")
    print("=" * 100)