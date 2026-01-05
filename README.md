# Project 01 — Titanic Survival Classification

## Goal
Predict passenger survival using structured tabular data.

## Tech
Python, pandas, scikit-learn, matplotlib, seaborn

## Repo Structure
- `data/` raw/interim/processed (not tracked in git)
- `notebooks/` exploration (optional)
- `src/` reusable code
- `reports/` figures + results
- `models/` saved models (not tracked in git)

## Visualizations (saved in `reports/figures/`)
A few highlights:

### Survival Rate by Sex
![Survival Rate by Sex](reports/figures/01_survival_rate_by_sex.png)

### Survival Rate by Passenger Class
![Survival Rate by Pclass](reports/figures/02_survival_rate_by_pclass.png)

### Age vs Survival (Violin)
![Age vs Survival](reports/figures/03_age_vs_survival_violin.png)

### Correlation Heatmap
![Correlation Heatmap](reports/figures/07_correlation_heatmap.png)

### PCA (2D) of Numeric Features
![PCA Scatter](reports/figures/09_pca_scatter.png)

### Survival Rate by Class, Split by Sex
![Survival by Class and Sex](reports/figures/10_survival_by_pclass_and_sex.png)

### Age Distribution Split by Survival (KDE)
![Age KDE](reports/figures/11_age_kde_by_survival.png)

### Fare vs Survival (Boxplot)
![Fare Boxplot](reports/figures/12_fare_boxplot_by_survival.png)

### Survival Rate by Age Group
![Survival by Age Bins](reports/figures/13_survival_rate_by_age_bins.png)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
