from pathlib import Path

import pandas as pd
from joblib import load


# Project root so paths work no matter where you run from
ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data" / "raw" / "test.csv"
MODEL_PATH = ROOT / "models" / "baseline_logreg.joblib"

# Keep Kaggle submissions here so they’re easy to find and commit to GitHub
SUB_DIR = ROOT / "reports" / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Load the trained pipeline (preprocess + model together)
    clf = load(MODEL_PATH)

    # Load test data (this has no Survived column)
    test = pd.read_csv(TEST_PATH)

    # Use the same feature columns the model was trained on
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X_test = test[features]

    # Predict survival (0/1)
    preds = clf.predict(X_test)

    # Build Kaggle submission format
    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": preds.astype(int)}
    )

    out_path = SUB_DIR / "baseline_logreg_submission.csv"
    submission.to_csv(out_path, index=False)

    print(f"Saved submission to: {out_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
