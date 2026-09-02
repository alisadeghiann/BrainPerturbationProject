from pathlib import Path

# ============================================================
# PROJECT
# ============================================================

BASE = Path(
    r"C:\Users\Ali\Desktop\BrainPerturbationProject"
)

print("=" * 100)
print("SEARCHING FOR THE SCRIPT THAT CREATED V5")
print("=" * 100)

print("\nProject:")
print(BASE)

# ============================================================
# TERMS WE ARE LOOKING FOR
# ============================================================

TERMS = [
    "perturbation_dataset_v5",
    "dataset_v5",
    "harmonized_epo",
    "harmonized",
    "harmonize",
    "HARMONIZED",
    "ELIGIBLE",
    "eligible",
    "perturbation_eligible",
    "standardized_epo",
    "metadata",
    "event_id",
    "task_role",
    "to_remember",
    "to_ignore",
]

# ============================================================
# SEARCH PYTHON FILES
# ============================================================

python_files = sorted(
    BASE.rglob("*.py")
)

print(
    f"\nPython files found: {len(python_files)}"
)

matches = []

for path in python_files:

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        continue

    text_lower = text.lower()

    found_terms = []

    for term in TERMS:

        if term.lower() in text_lower:
            found_terms.append(term)

    if found_terms:

        matches.append(
            (
                path,
                found_terms,
                text
            )
        )


# ============================================================
# PRINT MATCHED FILES
# ============================================================

print("\n" + "=" * 100)
print("FILES CONTAINING RELEVANT TERMS")
print("=" * 100)

print(
    f"\nMatched Python files: {len(matches)}"
)

for i, (path, terms, text) in enumerate(
    matches,
    start=1
):

    print("\n" + "-" * 100)

    print(
        f"[{i}] {path}"
    )

    print(
        "MATCHES:"
    )

    print(
        ", ".join(terms)
    )


# ============================================================
# PRINT IMPORTANT CODE CONTEXT
# ============================================================

print("\n" + "=" * 100)
print("IMPORTANT CODE CONTEXT")
print("=" * 100)

for i, (path, terms, text) in enumerate(
    matches,
    start=1
):

    lines = text.splitlines()

    relevant_line_numbers = set()

    for line_number, line in enumerate(
        lines
    ):

        line_lower = line.lower()

        # We especially care about these
        important_terms = [
            "perturbation_dataset_v5",
            "dataset_v5",
            "harmonized_epo",
            "harmonized",
            "harmonize",
            "eligible",
            "to_remember",
            "to_ignore",
            "metadata",
            "drop(",
            "dropna",
            "query(",
            "isin(",
            "filter",
            "exclude",
            "reject",
            "save",
            "write",
            "epochs[",
            "selection",
        ]

        if any(
            term.lower() in line_lower
            for term in important_terms
        ):

            relevant_line_numbers.add(
                line_number
            )

    if not relevant_line_numbers:
        continue

    print("\n" + "#" * 100)

    print(
        f"FILE [{i}]: {path}"
    )

    print(
        "MATCHED TERMS:",
        ", ".join(terms)
    )

    print("#" * 100)

    # Merge nearby lines into readable blocks
    sorted_lines = sorted(
        relevant_line_numbers
    )

    blocks = []

    current_block = []

    previous = None

    for line_number in sorted_lines:

        if (
            previous is None
            or line_number <= previous + 5
        ):

            current_block.append(
                line_number
            )

        else:

            blocks.append(
                current_block
            )

            current_block = [
                line_number
            ]

        previous = line_number

    if current_block:
        blocks.append(
            current_block
        )

    # Print blocks
    for block in blocks:

        start = max(
            0,
            block[0] - 3
        )

        end = min(
            len(lines),
            block[-1] + 4
        )

        print(
            f"\n--- Lines "
            f"{start + 1}-{end} ---"
        )

        for j in range(
            start,
            end
        ):

            print(
                f"{j + 1:5d}: "
                f"{lines[j]}"
            )


# ============================================================
# SEARCH ALL FILE NAMES
# ============================================================

print("\n" + "=" * 100)
print("FILES / DIRECTORIES WITH V5 OR HARMONIZED IN NAME")
print("=" * 100)

name_matches = []

for path in BASE.rglob("*"):

    name = path.name.lower()

    if (
        "v5" in name
        or "harmon" in name
        or "eligible" in name
        or "perturbation" in name
    ):

        name_matches.append(path)


for path in sorted(name_matches):

    print(path)


# ============================================================
# SEARCH FOR OUTPUT DIRECTORY REFERENCES
# ============================================================

print("\n" + "=" * 100)
print("EXACT SEARCH FOR V5 OUTPUT PATH")
print("=" * 100)

exact_matches = []

target_strings = [
    "perturbation_dataset_v5",
    "perturbation_dataset\\v5",
    "v5\\ELIGIBLE",
    "v5/ELIGIBLE",
]

for path in python_files:

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        continue

    for target in target_strings:

        if target.lower() in text.lower():

            exact_matches.append(
                (
                    path,
                    target
                )
            )

for path, target in exact_matches:

    print(
        f"\nFOUND: {target}"
    )

    print(path)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 100)
print("SEARCH FINISHED")
print("=" * 100)

print(
    """
READ-ONLY SEARCH.

No EEG/FIF/TSV/CSV files were modified.
No files were deleted.
No files were regenerated.

The purpose is ONLY to identify the script responsible
for creating the v5 harmonized dataset.
"""
)

print("\nDONE.")