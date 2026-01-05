from pathlib import Path
import pandas as pd

# This figures out where the project lives (so paths work no matter where you run from)
ROOT = Path(__file__).resolve().parents[1]

# Put the downloaded Kaggle files here and leave them unchanged
DATA_RAW = ROOT / "data" / "raw"

# These are the three files we expect to find
TRAIN_PATH = DATA_RAW / "train.csv"
TEST_PATH = DATA_RAW / "test.csv"
SUB_PATH = DATA_RAW / "gender_submission.csv"


def main() -> None:
    # If any file isn’t in the right place, stop and tell me which one
    for p in [TRAIN_PATH, TEST_PATH, SUB_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    # Read the CSVs into DataFrames
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sub = pd.read_csv(SUB_PATH)

    # Basic sanity check: sizes should look reasonable
    print("TRAIN shape:", train.shape)
    print("TEST  shape:", test.shape)
    print("SUB   shape:", sub.shape)

    # Show the column names so we know what features exist
    print("\nTRAIN columns:\n", list(train.columns))
    print("\nTEST columns:\n", list(test.columns))

    # Show the biggest missing-value columns first
    print("\nMissing values (TRAIN):")
    print(train.isna().sum().sort_values(ascending=False).head(15))

    print("\nMissing values (TEST):")
    print(test.isna().sum().sort_values(ascending=False).head(15))


if __name__ == "__main__":
    # Only run main() when we execute this file directly
    main()
