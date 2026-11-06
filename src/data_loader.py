"""
data_loader.py
--------------
Data loading, feature construction, and chronological train/test split.

Expected input: an Excel file (sheet 'data') or a CSV file containing columns:
    - company_id  : firm identifier (categorical)
    - date        : observation date (parseable by pd.to_datetime)
    - y           : current-period sales
    - z1 ... z25  : 25 numeric predictors

If no file is available, ``load_data(None)`` (or ``source='synthetic'``)
builds a synthetic panel with the same layout via ``src.synthetic``.

The target variable y_future is constructed as the next-period sales for each
company (one-period-ahead forecast). Rows where y_future is not available
(the last observation per company) are dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from src.utils import FEATURE_COLS, CAT_COLS, TARGET_COL


def _read_raw(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path, sheet_name='data')
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported file type: {path.suffix}')


def load_data(file_path: str | Path | None = None,
              source: str = 'auto',
              verbose: bool = True,
              **synthetic_kwargs) -> pd.DataFrame:
    """
    Load the panel dataset and construct the one-period-ahead target.

    Parameters
    ----------
    file_path : str or Path, optional
        Path to an Excel (sheet 'data') or CSV file.
    source : {'auto', 'file', 'synthetic'}
        'auto' reads ``file_path`` if it exists, otherwise generates synthetic
        data; 'file' requires the file; 'synthetic' always generates data.
    verbose : bool
        Print a short summary of the loaded panel.
    **synthetic_kwargs
        Passed to ``src.synthetic.generate_panel`` (e.g. n_dates=400).

    Returns
    -------
    pd.DataFrame
        Sorted panel with y_future added; rows without a future observation
        are dropped. ``df.attrs['source']`` records where the data came from.
    """
    use_file = (source == 'file' or
                (source == 'auto' and file_path is not None and Path(file_path).exists()))

    if use_file:
        df = _read_raw(file_path)
        origin = str(file_path)
    else:
        from src.synthetic import generate_panel
        df = generate_panel(**synthetic_kwargs)
        origin = 'synthetic'

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['company_id', 'date']).reset_index(drop=True)

    df['y_future'] = df.groupby('company_id')['y'].shift(-1)
    df = df.dropna(subset=['y_future']).copy()
    df.attrs['source'] = origin

    if verbose:
        print(f'Source   : {origin}')
        print(f'Loaded   : {df.shape[0]} observations, {df.shape[1]} columns')
        print(f'Dates    : {df["date"].min().date()} → {df["date"].max().date()}')
        print(f'Companies: {df["company_id"].nunique()}')
    return df


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.80,
                        verbose: bool = True):
    """
    Split a panel dataset chronologically on unique dates.

    The cut-point is determined by unique date order, so all companies
    at a given date stay together on the same side of the split.

    Parameters
    ----------
    df : pd.DataFrame
    train_ratio : float
        Fraction of unique dates to use for training. Default 0.80.

    Returns
    -------
    train_df, test_df : pd.DataFrame, pd.DataFrame
    """
    unique_dates = np.sort(df['date'].unique())
    n_dates = len(unique_dates)
    train_end_idx = int(n_dates * train_ratio)

    train_dates = unique_dates[:train_end_idx]
    test_dates  = unique_dates[train_end_idx:]

    train_df = df[df['date'].isin(train_dates)].copy()
    test_df  = df[df['date'].isin(test_dates)].copy()

    if verbose:
        print(f'Training : {len(train_df):>6} obs  '
              f'({train_df["date"].min().date()} → {train_df["date"].max().date()})')
        print(f'Test     : {len(test_df):>6} obs  '
              f'({test_df["date"].min().date()} → {test_df["date"].max().date()})')
        print(f'Split    : {len(train_df)/len(df):.1%} train / '
              f'{len(test_df)/len(df):.1%} test')

    return train_df, test_df


def get_arrays(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Extract raw feature matrices and target vectors from split DataFrames.

    Returns
    -------
    X_train_raw, y_train, X_test_raw, y_test
        Raw DataFrames/arrays ready to be passed to preprocessing steps.
    """
    X_train_raw = train_df[FEATURE_COLS + CAT_COLS]
    y_train     = train_df[TARGET_COL].values

    X_test_raw  = test_df[FEATURE_COLS + CAT_COLS]
    y_test      = test_df[TARGET_COL].values

    return X_train_raw, y_train, X_test_raw, y_test
