"""
utils.py
--------
Shared utilities: metric, global constants, and matplotlib style.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Global constants ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
FEATURE_COLS = [f'z{i}' for i in range(1, 26)]
CAT_COLS     = ['company_id']
TARGET_COL   = 'y_future'

np.random.seed(RANDOM_STATE)


def fmse(y_true, y_pred):
    """
    Forecast Mean Squared Error: (1/n) * sum((y_true - y_pred)^2).

    Equivalent to sklearn.metrics.mean_squared_error with squared=True.

    Parameters
    ----------
    y_true : array-like of shape (n,)
    y_pred : array-like of shape (n,)

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def save_fig(fig, filename, out_dir='results'):
    """Save figure to ``out_dir`` (created if needed) at 150 dpi."""
    from pathlib import Path
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path / filename}')
