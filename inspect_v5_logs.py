from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT
# ============================================================

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

V5_DIR = (
    BASE
    / "epochs_clean"
    / "logs"
    / "perturbation_dataset_v5"
)

SUBJECT = "sub-024"


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("V5 LOG / OUTPUT INSPECTION")
print("=" * 100)

print("\nProject:")
print(BASE)

print("\nV5 directory:")
print(V5_DIR)

print("\nV5 directory exists:")
print(V5_DIR.exists())


# ============================================================
# LIST EVERYTHING INSIDE V5
# ============================================================

print("\n" + "=" * 100)
print("ALL FILES INSIDE V5")
print("=" * 100)

if V5_DIR.exists():

    files = sorted(
        [
            p
            for p in V5_DIR.rglob("*")
            if p.is_file()
        ]
    )

    print(f"\nTotal files: {len(files)}")

    for p in files:
        print(p)

else:

    print("V5 directory does not exist.")


# ============================================================
# SEARCH FOR SUB-024 FILES
# ============================================================

print("\n" + "=" * 100)
print("SUB-024 FILES")
print("=" * 100)

sub_files = sorted(
    [
        p
        for p in V5_DIR.rglob("*")
        if p.is_file()
        and SUBJECT in p.name
    ]
)

print(
    f"\nNumber of sub-024 files: "
    f"{len(sub_files)}"
)

for p in sub_files:
    print(p)


# ============================================================
# SEARCH CSV / TSV / TXT LOGS
# ============================================================

print("\n" + "=" * 100)
print("TABULAR / TEXT LOG FILES")
print("=" * 100)

log_files = sorted(
    [
        p
        for p in V5_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in [".csv", ".tsv", ".txt", ".log", ".json"]
    ]
)

print(
    f"\nNumber of possible log files: "
    f"{len(log_files)}"
)

for p in log_files:
    print(p)


# ============================================================
# INSPECT CSV / TSV FILES
# ============================================================

print("\n" + "=" * 100)
print("READABLE LOG CONTENT")
print("=" * 100)

for p in log_files:

    print("\n" + "-" * 100)
    print("FILE:")
    print(p)
    print("-" * 100)

    try:

        suffix = p.suffix.lower()

        if suffix == ".csv":

            df = pd.read_csv(p)

        elif suffix == ".tsv":

            df = pd.read_csv(
                p,
                sep="\t"
            )

        else:

            print(
                "Text/JSON file - printing first 10000 characters:"
            )

            try:
                text = p.read_text(
                    encoding="utf-8",
                    errors="replace"
                )

                print(
                    text[:10000]
                )

            except Exception as e:

                print(
                    f"Could not read text: {e}"
                )

            continue

        print(
            f"Shape: {df.shape}"
        )

        print(
            "\nColumns:"
        )

        print(
            df.columns.tolist()
        )

        print(
            "\nFirst 20 rows:"
        )

        print(
            df.head(20)
            .to_string(index=False)
        )

        # ----------------------------------------------------
        # Search for subject
        # ----------------------------------------------------

        subject_mask = pd.Series(
            False,
            index=df.index
        )

        for col in df.columns:

            try:

                mask = (
                    df[col]
                    .astype(str)
                    .str.contains(
                        SUBJECT,
                        case=False,
                        na=False
                    )
                )

                subject_mask = (
                    subject_mask
                    | mask
                )

            except Exception:
                pass

        if subject_mask.any():

            print(
                "\nRows containing sub-024:"
            )

            print(
                df.loc[
                    subject_mask
                ]
                .head(100)
                .to_string(index=False)
            )

        # ----------------------------------------------------
        # Search relevant column names
        # ----------------------------------------------------

        keywords = [
            "subject",
            "run",
            "trial",
            "epoch",
            "condition",
            "remember",
            "ignore",
            "eligible",
            "reject",
            "exclude",
            "reason",
            "quality",
            "artifact",
            "response",
            "accuracy",
            "perturb",
        ]

        relevant_columns = []

        for col in df.columns:

            col_lower = str(col).lower()

            if any(
                keyword in col_lower
                for keyword in keywords
            ):

                relevant_columns.append(col)

        if relevant_columns:

            print(
                "\nRelevant columns:"
            )

            print(
                relevant_columns
            )

            print(
                "\nRelevant column values:"
            )

            print(
                df[
                    relevant_columns
                ]
                .head(100)
                .to_string(index=False)
            )

    except Exception as e:

        print(
            f"Could not inspect file: {e}"
        )


# ============================================================
# SEARCH ENTIRE PROJECT FOR V5 REFERENCES
# ============================================================

print("\n" + "=" * 100)
print("SEARCHING PROJECT FOR V5 / HARMONIZED REFERENCES")
print("=" * 100)

search_terms = [
    "perturbation_dataset_v5",
    "harmonized_epo",
    "harmonized",
    "ELIGIBLE",
    "eligible",
]


code_files = sorted(
    [
        p
        for p in BASE.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in [".py", ".txt", ".md", ".json"]
    ]
)

found_any = False

for p in code_files:

    try:

        text = p.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception:
        continue

    matches = []

    for term in search_terms:

        if term.lower() in text.lower():

            matches.append(term)

    if matches:

        found_any = True

        print("\n" + "-" * 100)
        print("FILE:")
        print(p)
        print("MATCHES:")
        print(matches)
        print("-" * 100)

        lines = text.splitlines()

        for i, line in enumerate(lines):

            if any(
                term.lower() in line.lower()
                for term in search_terms
            ):

                start = max(
                    0,
                    i - 3
                )

                end = min(
                    len(lines),
                    i + 4
                )

                print(
                    f"\nLines {start + 1}-{end}:"
                )

                for j in range(
                    start,
                    end
                ):

                    print(
                        f"{j + 1:5d}: "
                        f"{lines[j]}"
                    )


if not found_any:

    print(
        "\nNo references to v5/harmonized "
        "were found in Python/text/JSON files."
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 100)
print("INSPECTION FINISHED")
print("=" * 100)

print(
    """
IMPORTANT:

This script is READ-ONLY.

It does NOT:
- modify EEG files
- modify FIF files
- modify TSV files
- modify CSV files
- delete anything
- regenerate anything

Its only purpose is to discover:
1. What logs exist for v5
2. What information they contain
3. Which files/scripts created the harmonized dataset
4. Why epochs may have been excluded
"""
)

print("\nDONE.")