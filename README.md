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

| Model | Train FMSE | Test FMSE | Gen. Gap | Key Parameter |
|---|---|---|---|---|
| Lasso | 10,727,815.65 | 21,703,347.87 | 10,975,532.22 | λ* = 3.05 |
| PCR | 10,729,501.52 | 21,631,702.74 | 10,902,201.22 | k = 19 PCs (95.25% var) |
| Random Forest | 7,022,355.46 | 21,254,213.48 | 14,231,858.02 | depth=None, max_features=0.3 |
| Lasso-PCR | — | 21,565,839.11 | — | joint (λ*, k*) selection |

Random Forest achieves the lowest test FMSE, outperforming PCR by 1.75% and Lasso by 2.07%; PCR in turn lowers test FMSE relative to Lasso by 0.33%. The modest spread between best and worst model (2.07%) indicates most of the predictive information is recoverable through linear structure alone. Random Forest's much larger generalisation gap (14.2M vs ~10.9–11.0M for the linear models) reflects a lower-bias, higher-variance specification: its OOB R² is 0.3957, with no directly comparable in-sample statistic available for the linear models. Lasso-PCR, which pre-filters z-features by Lasso before PCA rotation, improves on standalone PCR by only 0.30%, indicating that the broad factor structure and persistent company fixed effects are already the dominant source of linear predictive power.

**Feature importance agreement.** Spearman rank correlation between Lasso |coefficient| and RF Gini importance across the 25 z-features is ρ = 0.427 (p = 0.033), a statistically significant agreement at the 5% level. Both methods converge on customer engagement and demand-side indicators (retention, satisfaction, purchasing power) as leading predictors, while diverging on variables likely driven by thresholds or interaction effects rather than smooth linear relationships.

**Residual analysis.** All three models produce right-skewed residual distributions on the test set, indicating systematic under-prediction of high-sales observations. Residual means are comparable across models (Lasso: 713.1, PCR: 708.9, Random Forest: 784.2), as are standard deviations (Lasso: 4,603.8, PCR: 4,596.6, Random Forest: 4,543.0). The persistent right skew across all specifications suggests extreme sales events are driven by factors outside the 25 observed predictors.

**Robustness — winsorisation (1%/99%, training-set bounds).**

| Model | Original FMSE | Winsorised FMSE | Δ% | Rank (orig → wins.) |
|---|---|---|---|---|
| Lasso | 21,703,347.87 | 21,697,277.04 | −0.03% | 3 → 3 |
| PCR | 21,631,702.74 | 21,633,514.82 | +0.01% | 2 → 2 |
| Random Forest | 21,254,213.48 | 21,231,663.14 | −0.11% | 1 → 1 |
| Lasso-PCR | 21,565,839.11 | 21,569,088.35 | +0.02% | — |

Model ranking is unchanged after winsorisation for all four models, confirming the comparative conclusions do not depend on extreme predictor values.

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
    └── (generated PNGs)
```
