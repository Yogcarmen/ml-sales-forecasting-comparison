"""
synthetic.py
------------
Generate a synthetic panel with the same layout as the (licensed) course
dataset, so the full pipeline can be run end to end without the original file.

The data-generating process is deliberately close in *structure* to the real
problem, not in values:

* 25 firms observed on consecutive daily dates (default 1,500 days).
* 25 numeric predictors z1..z25 on different scales, driven by a few common
  latent factors plus idiosyncratic noise (so PCA has a factor structure to
  find and Lasso has correlated predictors to choose between).
* Sales ``y`` are positive and right-skewed: a log-linear model with
  persistent firm fixed effects, a handful of truly predictive z's, one
  threshold (non-linear) effect and fat-tailed shocks.

Because the true coefficients are known, the synthetic data can also be used
to check whether each model recovers the right predictors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import FEATURE_COLS, RANDOM_STATE

# Predictors that truly drive sales in the synthetic DGP (log-scale effects).
TRUE_LINEAR_EFFECTS = {
    'z1': 0.30, 'z2': 0.22, 'z5': -0.18, 'z8': 0.15,
    'z12': 0.12, 'z17': -0.10, 'z21': 0.08,
}
# One threshold effect that a linear model can only approximate.
THRESHOLD_FEATURE = 'z9'


def generate_panel(n_firms: int = 25,
                   n_dates: int = 1500,
                   start_date: str = '2020-01-01',
                   n_factors: int = 4,
                   seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Simulate a firm x date panel with columns company_id, date, y, z1..z25.

    Parameters
    ----------
    n_firms : int
        Number of firms (cross-sectional units).
    n_dates : int
        Number of consecutive daily observations per firm.
    start_date : str
        First observation date.
    n_factors : int
        Number of common latent factors behind the z predictors.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format panel sorted by company_id and date.
    """
    rng = np.random.default_rng(seed)
    n_feat = len(FEATURE_COLS)
    dates = pd.date_range(start_date, periods=n_dates, freq='D')
    firms = [f'C{i:02d}' for i in range(1, n_firms + 1)]

    # Latent factors: persistent AR(1) processes shared by all firms.
    factors = np.zeros((n_dates, n_factors))
    for t in range(1, n_dates):
        factors[t] = 0.97 * factors[t - 1] + rng.normal(0, 0.25, n_factors)

    loadings = rng.normal(0, 1, (n_factors, n_feat))
    # Each predictor gets its own location and scale, like the raw data.
    locs = rng.uniform(0, 100, n_feat)
    scales = rng.uniform(0.1, 15, n_feat)

    firm_effect = rng.normal(6.8, 1.0, n_firms)       # log-sales level
    firm_tilt = rng.normal(0, 0.3, (n_firms, n_factors))

    beta = np.zeros(n_feat)
    for name, b in TRUE_LINEAR_EFFECTS.items():
        beta[FEATURE_COLS.index(name)] = b
    thr_idx = FEATURE_COLS.index(THRESHOLD_FEATURE)

    rows = []
    for i, firm in enumerate(firms):
        f_i = factors + firm_tilt[i]
        z_std = 0.6 * f_i @ loadings / np.sqrt(n_factors) + rng.normal(0, 1.0, (n_dates, n_feat))
        z_std = (z_std - z_std.mean(0)) / z_std.std(0)

        # Today's predictors drive *tomorrow's* sales, matching the
        # one-period-ahead forecasting target used by the pipeline.
        signal = z_std @ beta + 0.35 * (z_std[:, thr_idx] > 1.0)  # incl. threshold effect
        signal = np.concatenate([[0.0], signal[:-1]])
        log_y = (firm_effect[i]
                 + signal
                 + rng.standard_t(df=4, size=n_dates) * 0.20)  # fat-tailed shocks
        y = np.exp(log_y)

        z_raw = locs + scales * z_std
        block = pd.DataFrame(z_raw, columns=FEATURE_COLS)
        block.insert(0, 'date', dates)
        block.insert(0, 'company_id', firm)
        block['y'] = y
        rows.append(block)

    df = pd.concat(rows, ignore_index=True)
    return df[['company_id', 'date', 'y'] + FEATURE_COLS]


def true_predictors() -> list[str]:
    """Names of the z-features that carry signal in the synthetic DGP."""
    return list(TRUE_LINEAR_EFFECTS) + [THRESHOLD_FEATURE]
