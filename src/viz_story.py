from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save images to files (no pop-up windows)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Project paths
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name: str) -> None:
    # Save the current matplotlib figure and close it
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(TRAIN_PATH)

    # 1) Survival rate by class, split by sex (easy story: class + gender mattered)
    plt.figure()
    sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex", errorbar=None)
    plt.title("Survival Rate by Class, Split by Sex")
    plt.ylabel("Survival Rate")
    save_fig("10_survival_by_pclass_and_sex.png")

    # 2) Age distribution by survival (smooth curves, more visual than histogram)
    plt.figure()
    sns.kdeplot(data=df, x="Age", hue="Survived", common_norm=False)
    plt.title("Age Distribution Split by Survival (KDE)")
    save_fig("11_age_kde_by_survival.png")

    # 3) Fare vs survival (boxplot shows median + spread clearly)
    plt.figure()
    sns.boxplot(data=df, x="Survived", y="Fare")
    plt.yscale("log")  # fare is skewed, log makes the plot readable
    plt.title("Fare vs Survival (Boxplot, log scale)")
    plt.xlabel("Survived (0=No, 1=Yes)")
    save_fig("12_fare_boxplot_by_survival.png")

    # 4) Survival rate by age bins (simple feature idea + clear chart)
    df_bins = df.copy()
    df_bins["AgeBin"] = pd.cut(df_bins["Age"], bins=[0, 16, 30, 45, 60, 80], include_lowest=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_bins, x="AgeBin", y="Survived", errorbar=None)
    plt.title("Survival Rate by Age Group")
    plt.ylabel("Survival Rate")
    plt.xticks(rotation=20)
    save_fig("13_survival_rate_by_age_bins.png")

    # 5) Count of passengers by class and survival (shows base rates + imbalance)
    plt.figure()
    sns.countplot(data=df, x="Pclass", hue="Survived")
    plt.title("Passenger Counts by Class and Survival")
    save_fig("14_counts_by_pclass_and_survival.png")

    # 6) Embarked vs class (who boarded where, by class)
    plt.figure()
    sns.countplot(data=df, x="Embarked", hue="Pclass")
    plt.title("Passenger Counts by Embarked Port and Class")
    save_fig("15_counts_by_embarked_and_class.png")


if __name__ == "__main__":
    main()
