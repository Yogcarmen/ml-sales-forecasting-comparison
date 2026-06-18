"""
data_loader.py
--------------
Data loading, feature construction, and chronological train/test split.

Expected input: an Excel file with a sheet named 'data' containing columns:
    - company_id  : firm identifier (categorical)
    - date        : observation date (parseable by pd.to_datetime)
    - y           : current-period sales
    - z1 ... z25  : 25 standardised numeric predictors

The target variable y_future is constructed as the next-period sales for each
company (one-period-ahead forecast). Rows where y_future is not available
(the last observation per company) are dropped.
"""

import numpy as np
import pandas as pd
from src.utils import FEATURE_COLS, CAT_COLS, TARGET_COL


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the panel dataset and construct the one-period-ahead target.

    Parameters
    ----------
    file_path : str
        Path to the Excel file (expects sheet name 'data').

    Returns
    -------
    pd.DataFrame
        Sorted panel with y_future added; rows without a future observation
        are dropped.
    """
    df = pd.read_excel(file_path, sheet_name='data')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['company_id', 'date']).reset_index(drop=True)

    df['y_future'] = df.groupby('company_id')['y'].shift(-1)
    df = df.dropna(subset=['y_future']).copy()

    print(f'Loaded  : {df.shape[0]} observations, {df.shape[1]} columns')
    print(f'Dates   : {df["date"].min().date()} → {df["date"].max().date()}')
    print(f'Companies: {sorted(df["company_id"].unique())}')
    return df


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.80):
    """
    Split a panel dataset chronologically on unique dates.

    The cut-point is determined by unique date order, so all 25 companies
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
