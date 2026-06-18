"""
cv.py
-----
Time-aware cross-validation for panel (entity × time) data.

Standard k-fold CV draws validation samples randomly, which allows future
observations to leak into training folds. For a chronological forecasting task
this creates look-ahead bias. PanelTimeSeriesCV uses an expanding-window design
that splits on unique dates so that all entities observed at a given date always
belong to the same fold, and validation dates are strictly after all training
dates in every fold.
"""

import numpy as np


class PanelTimeSeriesCV:
    """
    Expanding-window cross-validation for panel (entity x time) data.

    Splits are defined on unique time periods so that all entities observed
    at the same date always belong to the same fold. Validation windows are
    strictly forward-looking relative to the corresponding training window.

    Parameters
    ----------
    n_splits : int
        Number of CV folds.
    date_array : array-like
        Per-row date values of the training set (length = n_train_obs).
    gap : int, optional
        Number of date periods to skip between the end of training and the
        start of validation in each fold. Default 0.
    """

    def __init__(self, n_splits=5, date_array=None, gap=0):
        self.n_splits = n_splits
        self.date_array = np.asarray(date_array)
        self.gap = gap

    def split(self, X, y=None, groups=None):
        dates = self.date_array
        unique_dates = np.sort(np.unique(dates))
        n_d = len(unique_dates)
        fold_size = n_d // (self.n_splits + 1)

        for k in range(self.n_splits):
            cutoff_pos = (k + 1) * fold_size - 1
            val_start_p = cutoff_pos + 1 + self.gap
            val_end_p = min(val_start_p + fold_size - 1, n_d - 1)

            if val_start_p >= n_d:
                break

            cutoff_date = unique_dates[cutoff_pos]
            val_start_date = unique_dates[val_start_p]
            val_end_date = unique_dates[val_end_p]

            train_idx = np.where(dates <= cutoff_date)[0]
            val_idx = np.where(
                (dates >= val_start_date) & (dates <= val_end_date)
            )[0]
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
