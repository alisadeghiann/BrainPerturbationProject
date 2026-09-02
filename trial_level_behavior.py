import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\Ali\Desktop\BrainPerturbationProject")

INPUT = (
    BASE
    / "features"
    / "behavior_aligned"
    / "epoch_event_map.csv"
)

OUTPUT_DIR = BASE / "features" / "behavior_aligned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "trial_level_behavior.csv"

print("=" * 90)
print("TRIAL-LEVEL BEHAVIORAL RECONSTRUCTION")
print("=" * 90)

df = pd.read_csv(INPUT)

print(f"Input rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

required = ["subject", "run", "epoch", "event_name"]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# ---------------------------------------------------------
# Sort
# ---------------------------------------------------------

df = df.sort_values(
    ["subject", "run", "epoch"]
).reset_index(drop=True)

# ---------------------------------------------------------
# Identify trial boundaries
# ---------------------------------------------------------

# A new trial normally begins with show_cross.
df["new_trial"] = (
    (df["event_name"] == "show_cross")
    .astype(int)
)

# Cumulative trial number within subject/run
df["trial"] = (
    df.groupby(["subject", "run"])["new_trial"]
      .cumsum()
)

# Remove anything before first identifiable trial
df = df[df["trial"] > 0].copy()

# ---------------------------------------------------------
# Build trial-level table
# ---------------------------------------------------------

records = []

for (subject, run, trial), g in df.groupby(
    ["subject", "run", "trial"],
    sort=True
):

    events = g["event_name"].astype(str).tolist()

    record = {
        "subject": subject,
        "run": run,
        "trial": int(trial),

        "n_epochs": len(g),

        "has_fixation": int("show_cross" in events),
        "has_letters": int("show_letter" in events),
        "has_wm_period": int("show_dash" in events),

        "has_left_click": int("left_click" in events),
        "has_right_click": int("right_click" in events),

        "has_correct_feedback": int("sound_beep" in events),
        "has_incorrect_feedback": int("sound_buzz" in events),
    }

    # -----------------------------------------------------
    # Extract letter/event information where available
    # -----------------------------------------------------

    letters = g[g["event_name"] == "show_letter"]

    record["n_letters"] = len(letters)

    # -----------------------------------------------------
    # Event sequence
    # -----------------------------------------------------

    record["event_sequence"] = "|".join(events)

    # -----------------------------------------------------
    # Determine broad behavioral outcome
    # -----------------------------------------------------

    if "sound_beep" in events:
        record["feedback"] = "correct"

    elif "sound_buzz" in events:
        record["feedback"] = "incorrect"

    else:
        record["feedback"] = "unknown"

    # -----------------------------------------------------
    # Response type
    # -----------------------------------------------------

    if "right_click" in events:
        record["response_type"] = "right_click"

    elif "left_click" in events:
        record["response_type"] = "left_click"

    else:
        record["response_type"] = "none"

    # -----------------------------------------------------
    # Trial completeness
    # -----------------------------------------------------

    record["complete_trial"] = int(
        record["has_fixation"]
        and record["has_letters"]
        and record["has_wm_period"]
        and (
            record["has_left_click"]
            or record["has_right_click"]
        )
    )

    records.append(record)

trial_df = pd.DataFrame(records)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

trial_df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 90)
print("TRIAL RECONSTRUCTION COMPLETE")
print("=" * 90)

print(f"Trials:       {len(trial_df):,}")
print(f"Subjects:     {trial_df['subject'].nunique()}")
print(f"Runs:         {trial_df[['subject','run']].drop_duplicates().shape[0]:,}")

print()
print("Feedback:")
print(trial_df["feedback"].value_counts(dropna=False))

print()
print("Response type:")
print(trial_df["response_type"].value_counts(dropna=False))

print()
print("Complete trials:")
print(trial_df["complete_trial"].value_counts(dropna=False))

print()
print("Saved:")
print(OUTPUT)

print("=" * 90)
