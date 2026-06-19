# Predictive Modelling for Sales Forecasting

Benchmarking Lasso, Principal Component Regression (PCR), and Random Forest on a 25-firm panel dataset for one-period-ahead sales forecasting, with an emphasis on time-aware cross-validation and cross-model feature importance agreement.

## Overview

Standard k-fold cross-validation draws validation samples randomly, which allows future observations to leak into training folds. This project addresses that by implementing `PanelTimeSeriesCV`: an expanding-window CV that splits on unique dates, so all 25 firms at a given date always belong to the same fold and validation windows are strictly forward-looking. Hyperparameter selection for all models (λ for Lasso, component count for PCR, grid search for RF) uses this procedure exclusively on the training set; the held-out test set is used only for final evaluation.

## Dataset

A panel of 25 firms with 25 standardised numeric predictors (`z1`–`z25`) and company fixed effects. The target is next-period sales (`y_future`), constructed as a one-period lead of the observed sales variable. Data is split chronologically: the first 80% of unique dates form the training set and the remaining 20% form the test set.

The dataset is not included in this repository (course licence). See `data/README.md` for the expected format.

## Methodology

**Lasso** — numeric predictors standardised, company dummies one-hot encoded. `LassoCV` selects λ* minimising CV MSE across 5 time-aware folds over a 100-point log-spaced grid in [10⁻⁴, 10²].

**PCR** — PCA applied to 25 standardised numeric predictors; components retained up to 95% cumulative variance threshold. OLS fitted on [PC scores | company dummies]. Company dummies are excluded from PCA to preserve their role as fixed-effect intercept shifters.

**Random Forest** — scale-invariant; no standardisation. `GridSearchCV` over `n_estimators` × `max_depth` × `max_features` × `min_samples_leaf` (320 combinations × 5 folds). Final model refitted with `oob_score=True`.

**Lasso-PCR extension** — joint CV over (λ, k): Lasso selects z-features per fold, PCA reduces to k components, OLS fits on [PC scores | dummies]. Optimal (λ*, k*) chosen from a 25 × 15 grid.

**Feature importance agreement** — Lasso |coefficient| rankings and RF Gini importance rankings compared via Spearman rank correlation across all 25 z-features, providing model-agnostic evidence of predictive signal.

## Key Results

| Model | Train FMSE | Test FMSE | Gen. Gap |
|---|---|---|---|
| Lasso | — | 21,703,348 | — |
| PCR | — | 21,631,703 | — |
| Random Forest | — | 21,254,213 | — |
| Lasso-PCR | — | — | — |

Random Forest achieves the lowest test FMSE, outperforming PCR and Lasso by 1.75% and 2.07% respectively. The larger train-to-test generalisation gap for RF (relative to the linear models) is consistent with lower bias and higher variance. Spearman rank correlation between Lasso |coefficient| and RF Gini importance across z-features: ρ = 0.427 (p = 0.033), indicating moderate cross-model agreement in variable rankings. Winsorisation at 1%/99% confirms ranking stability across all models.

## Reproduction

```bash
# 1. Clone and enter the directory
git clone https://github.com/Yogcarmen/ml-sales-forecasting-comparison.git
cd ml-sales-forecasting-comparison

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place DS2.xlsx in data/ (see data/README.md for expected format)

# 4. Run the notebook
jupyter notebook notebooks/results.ipynb
```

## File Structure

```
ml-sales-forecasting-comparison/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md           # data format specification
├── src/
│   ├── __init__.py
│   ├── cv.py               # PanelTimeSeriesCV
│   ├── data_loader.py      # load_data, chronological_split, get_arrays
│   ├── models.py           # train_lasso, train_pcr, train_rf, train_lasso_pcr
│   └── utils.py            # fmse, constants, save_fig
├── notebooks/
│   └── results.ipynb       # end-to-end pipeline and plots
└── results/
    └── (generated PNGs)
```
