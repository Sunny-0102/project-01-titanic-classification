from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Find the project root so file paths work from anywhere
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"

# Save trained models here (this folder is ignored by git)
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main() -> None:
    # Load the training data
    df = pd.read_csv(TRAIN_PATH)

    # Target we want to predict
    y = df["Survived"]

    # Start with a simple, strong feature set
    # (We skip Name/Ticket/Cabin for baseline because they need extra feature engineering)
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[features]

    # Split so we can measure performance on unseen data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Columns by type
    numeric_features = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
    categorical_features = ["Sex", "Embarked"]

    # Numeric: fill missing values with the median
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Categorical: fill missing with most common value, then one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Apply the right preprocessing to the right columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Simple, reliable baseline model
    model = LogisticRegression(max_iter=1000)

    # One pipeline: preprocess + model in a single object
    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Train
    clf.fit(X_train, y_train)

    # Validate
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation accuracy: {acc:.4f}")

    # Save the trained pipeline for reuse later
    dump(clf, MODEL_DIR / "baseline_logreg.joblib")
    print("Saved model to models/baseline_logreg.joblib")


if __name__ == "__main__":
    main()
