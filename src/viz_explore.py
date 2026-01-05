from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # lets plots save to files without opening windows

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Find project root so paths work no matter where you run from
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"

# Save all images here
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(filename: str) -> None:
    # Saves the current figure into reports/figures and then closes it
    out = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    # Make plots look consistent and clean
    sns.set_theme(style="whitegrid")

    # Load Kaggle training data
    df = pd.read_csv(TRAIN_PATH)

    # 1) Survival rate by Sex
    plt.figure()
    sns.barplot(data=df, x="Sex", y="Survived", errorbar=None)
    plt.title("Survival Rate by Sex")
    plt.ylabel("Survival Rate")
    save_fig("01_survival_rate_by_sex.png")

    # 2) Survival rate by Passenger Class
    plt.figure()
    sns.barplot(data=df, x="Pclass", y="Survived", errorbar=None)
    plt.title("Survival Rate by Passenger Class (Pclass)")
    plt.ylabel("Survival Rate")
    save_fig("02_survival_rate_by_pclass.png")

    # 3) Age distribution split by survival (violin)
    plt.figure()
    sns.violinplot(data=df, x="Survived", y="Age", inner="quartile")
    plt.title("Age Distribution vs Survival (Violin)")
    plt.xlabel("Survived (0=No, 1=Yes)")
    save_fig("03_age_vs_survival_violin.png")

    # 4) Fare distribution (log scale helps because fares are skewed)
    plt.figure()
    sns.histplot(data=df, x="Fare", bins=40, kde=True)
    plt.yscale("log")
    plt.title("Fare Distribution (log scale on count)")
    save_fig("04_fare_distribution.png")

    # 5) Survival rate by Embarked port
    plt.figure()
    sns.barplot(data=df, x="Embarked", y="Survived", errorbar=None)
    plt.title("Survival Rate by Embarked")
    plt.ylabel("Survival Rate")
    save_fig("05_survival_rate_by_embarked.png")

    # 6) Missing values heatmap (quick way to “see” missingness)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isna(), cbar=False)
    plt.title("Missing Values Map (Train)")
    save_fig("06_missing_values_map.png")

    # 7) Correlation heatmap (numeric columns only)
    numeric_df = df.select_dtypes(include=["number"]).copy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features)")
    save_fig("07_correlation_heatmap.png")

    # 8) “Family size” feature and survival rate (simple feature engineering viz)
    df_feat = df.copy()
    df_feat["FamilySize"] = df_feat["SibSp"].fillna(0) + df_feat["Parch"].fillna(0) + 1
    plt.figure()
    sns.barplot(data=df_feat, x="FamilySize", y="Survived", errorbar=None)
    plt.title("Survival Rate by Family Size")
    plt.ylabel("Survival Rate")
    save_fig("08_survival_rate_by_family_size.png")

    # 9) PCA scatter (uses scikit-learn) to visualize separation in numeric space
    # We impute missing values, scale, then reduce to 2D
    pca_features = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
    X = df[pca_features]
    y = df["Survived"].astype(int)

    pca_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )
    X_2d = pca_pipe.fit_transform(X)

    pca_df = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
    pca_df["Survived"] = y.values

    plt.figure()
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Survived", alpha=0.7)
    plt.title("PCA (2D) of Numeric Features (scikit-learn)")
    save_fig("09_pca_scatter.png")


if __name__ == "__main__":
    main()
