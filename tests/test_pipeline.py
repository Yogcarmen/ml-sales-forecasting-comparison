"""Basic checks: no look-ahead in CV, chronological split, synthetic data layout."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv import PanelTimeSeriesCV  # noqa: E402
from src.data_loader import load_data, chronological_split  # noqa: E402
from src.synthetic import generate_panel, true_predictors  # noqa: E402
from src.utils import FEATURE_COLS  # noqa: E402


def _small_panel():
    return load_data(source='synthetic', verbose=False, n_firms=5, n_dates=120)


def test_synthetic_layout():
    df = generate_panel(n_firms=4, n_dates=50)
    assert list(df.columns) == ['company_id', 'date', 'y'] + FEATURE_COLS
    assert len(df) == 4 * 50
    assert (df['y'] > 0).all()
    assert set(true_predictors()) <= set(FEATURE_COLS)


def test_target_is_next_period_sales():
    df = _small_panel()
    firm = df[df['company_id'] == df['company_id'].iloc[0]]
    assert np.allclose(firm['y_future'].values[:-1], firm['y'].values[1:])


def test_chronological_split_has_no_overlap():
    df = _small_panel()
    train, test = chronological_split(df, verbose=False)
    assert train['date'].max() < test['date'].min()
    # every firm is present on every date of each side
    assert train.groupby('date')['company_id'].nunique().eq(5).all()


def test_cv_validation_is_strictly_after_training():
    df = _small_panel()
    train, _ = chronological_split(df, verbose=False)
    dates = train['date'].values
    cv = PanelTimeSeriesCV(n_splits=5, date_array=dates)
    folds = list(cv.split(np.zeros(len(train))))
    assert len(folds) == 5
    for tr, va in folds:
        assert dates[tr].max() < dates[va].min()
        # a date never appears on both sides of a fold
        assert not set(pd.to_datetime(dates[tr])) & set(pd.to_datetime(dates[va]))


def test_cv_gap_skips_periods():
    df = _small_panel()
    train, _ = chronological_split(df, verbose=False)
    dates = train['date'].values
    cv = PanelTimeSeriesCV(n_splits=3, date_array=dates, gap=2)
    for tr, va in cv.split(np.zeros(len(train))):
        gap_days = (pd.Timestamp(dates[va].min()) - pd.Timestamp(dates[tr].max())).days
        assert gap_days == 3
