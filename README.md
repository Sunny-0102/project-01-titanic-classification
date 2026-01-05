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


## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
