from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save images to files (no pop-up windows)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from joblib import load
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split


# Project paths
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
MODEL_PATH = ROOT / "models" / "baseline_logreg.joblib"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name: str) -> None:
    # Save the current figure and close it
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    sns.set_theme(style="whitegrid")

    # Load data and model
    df = pd.read_csv(TRAIN_PATH)
    clf = load(MODEL_PATH)

    # Same feature set used in training
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[features]
    y = df["Survived"].astype(int)

    # Recreate the same validation split settings
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Predict on validation set
    y_pred = clf.predict(X_val)

    # 1) Confusion matrix (seaborn heatmap)
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title("Confusion Matrix (Validation)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_fig("16_confusion_matrix.png")

    # Probability scores for curve plots
    # (LogReg supports predict_proba; pipeline passes through)
    y_proba = clf.predict_proba(X_val)[:, 1]

    # 2) ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve (Validation)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    save_fig("17_roc_curve.png")

    # 3) Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_val, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.title("Precision–Recall Curve (Validation)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    save_fig("18_precision_recall_curve.png")


if __name__ == "__main__":
    main()

