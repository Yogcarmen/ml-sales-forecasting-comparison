"""
run_pipeline.py
---------------
End-to-end run: load data, split, tune and evaluate all four models, then
write a results table and the figures used in the README.

Examples
--------
Run on synthetic data (no licensed file needed), quick RF grid:

    python scripts/run_pipeline.py --synthetic --fast

Run on the original dataset with the full 320-combination RF grid:

    python scripts/run_pipeline.py --data data/DS2.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.cv import PanelTimeSeriesCV  # noqa: E402
from src.data_loader import load_data, chronological_split  # noqa: E402
from src.models import (train_lasso, train_pcr, train_rf, train_lasso_pcr,  # noqa: E402
                        FAST_RF_GRID, FULL_RF_GRID)
from src.utils import FEATURE_COLS, save_fig  # noqa: E402

COLORS = {'Lasso': '#4C72B0', 'PCR': '#DD8452', 'Random Forest': '#55A868',
          'Lasso-PCR': '#8172B3'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=str(ROOT / 'data' / 'DS2.xlsx'),
                   help='Path to the panel file (xlsx sheet "data" or csv).')
    p.add_argument('--synthetic', action='store_true',
                   help='Ignore --data and generate a synthetic panel.')
    p.add_argument('--n-dates', type=int, default=1500,
                   help='Dates per firm for synthetic data (default 1500).')
    p.add_argument('--fast', action='store_true',
                   help='Use a small Random Forest grid (4 combinations).')
    p.add_argument('--out', default=str(ROOT / 'results'),
                   help='Output folder for figures and tables.')
    return p.parse_args()


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_model_comparison(summary, out):
    s = summary.sort_values('Test FMSE')
    base = s['Test FMSE'].max()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.barh(s['Model'], s['Test FMSE'],
                   color=[COLORS[m] for m in s['Model']], height=0.55)
    for bar, v in zip(bars, s['Test FMSE']):
        gain = (base - v) / base * 100
        label = f'{v:,.0f}' + (f'   (−{gain:.2f}% vs worst)' if gain > 0 else '   (worst)')
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, '  ' + label,
                va='center', fontsize=9)
    lo = s['Test FMSE'].min() * 0.97
    ax.set_xlim(lo, base * 1.025)
    ax.invert_yaxis()
    ax.set_xlabel('Out-of-sample forecast MSE (lower is better)')
    ax.set_title('Test-set error by model (axis truncated to show differences)',
                 fontsize=11, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    save_fig(fig, 'model_comparison.png', out)


def plot_cv_scheme(train_df, test_df, tscv, out):
    dates = train_df['date'].values
    folds = list(tscv.split(np.zeros(len(train_df))))
    t0 = pd.Timestamp(train_df['date'].min())
    t_end = pd.Timestamp(test_df['date'].max())
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for k, (tr, va) in enumerate(folds):
        tr_end = pd.Timestamp(dates[tr].max())
        va_s, va_e = pd.Timestamp(dates[va].min()), pd.Timestamp(dates[va].max())
        y = len(folds) - k
        ax.barh(y, (tr_end - t0).days, left=0, color='#4C72B0', height=0.6)
        ax.barh(y, (va_e - va_s).days + 1, left=(va_s - t0).days, color='#DD8452', height=0.6)
    ts = pd.Timestamp(test_df['date'].min())
    ax.barh(0, (t_end - ts).days + 1, left=(ts - t0).days, color='#999999', height=0.6)
    ax.set_yticks(range(0, len(folds) + 1))
    ax.set_yticklabels(['Test'] + [f'Fold {k}' for k in range(len(folds), 0, -1)])
    ax.set_xlabel('Days since start of sample')
    ax.set_title('Expanding-window CV on unique dates: validation is always after training',
                 fontsize=11, loc='left')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#4C72B0', label='Train'),
                       Patch(color='#DD8452', label='Validate'),
                       Patch(color='#999999', label='Held-out test')],
              loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    save_fig(fig, 'cv_scheme.png', out)


def plot_importance_agreement(lasso_res, rf_res, out, highlight=None):
    coef = pd.Series(lasso_res['model'].coef_, index=lasso_res['all_names'])
    lasso_abs = coef.reindex(FEATURE_COLS).abs()
    rf_gini = rf_res['imp_gini'].reindex(FEATURE_COLS)
    l_rank = lasso_abs.rank(ascending=False)
    r_rank = rf_gini.rank(ascending=False)
    rho, p = spearmanr(lasso_abs.values, rf_gini.values)

    fig, ax = plt.subplots(figsize=(5.2, 5))
    colors = ['#C44E52' if highlight and f in highlight else '#4C72B0' for f in FEATURE_COLS]
    ax.scatter(l_rank, r_rank, c=colors, s=40, zorder=3)
    for f in FEATURE_COLS:
        if l_rank[f] <= 6 or r_rank[f] <= 6:
            ax.annotate(f, (l_rank[f], r_rank[f]), xytext=(4, 3),
                        textcoords='offset points', fontsize=8)
    ax.plot([1, 25], [1, 25], ls='--', color='grey', lw=1)
    if highlight:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker='o', ls='', color='#C44E52',
                                  label='true predictor (synthetic DGP)'),
                           Line2D([], [], marker='o', ls='', color='#4C72B0',
                                  label='noise feature')],
                  loc='lower left', fontsize=8, frameon=False)
    ax.set_xlabel('Lasso rank (|coefficient|)')
    ax.set_ylabel('Random Forest rank (Gini importance)')
    ax.set_title(f'Do the models agree on what matters?\nSpearman ρ = {rho:.3f} (p = {p:.3f})',
                 fontsize=11, loc='left')
    ax.set_xlim(0, 26); ax.set_ylim(0, 26)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    save_fig(fig, 'importance_agreement.png', out)
    return rho, p


def plot_residuals(results, y_test, out):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    resid_all = np.concatenate([y_test - r['test_pred'] for r in results.values()])
    lo, hi = np.percentile(resid_all, [1, 99])
    bins = np.linspace(lo, hi, 60)
    for name, r in results.items():
        if name == 'Lasso-PCR':
            continue
        ax.hist(y_test - r['test_pred'], bins=bins, histtype='step', lw=1.6,
                color=COLORS[name], label=name, density=True)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Test residual (actual − forecast), central 98%')
    ax.set_ylabel('Density')
    ax.set_title('All models under-predict the largest sales days (right skew)',
                 fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    save_fig(fig, 'residuals.png', out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    source = 'synthetic' if args.synthetic else 'auto'
    df = load_data(args.data, source=source, n_dates=args.n_dates)
    is_synth = df.attrs.get('source') == 'synthetic'

    train_df, test_df = chronological_split(df, train_ratio=0.80)
    tscv = PanelTimeSeriesCV(n_splits=5, date_array=train_df['date'].values)
    y_test = test_df['y_future'].values

    print('\n--- Lasso ---');       lasso = train_lasso(train_df, test_df, tscv)
    print('\n--- PCR ---');         pcr = train_pcr(train_df, test_df, lasso)
    print('\n--- Random Forest ---')
    rf = train_rf(train_df, test_df, tscv,
                  param_grid=FAST_RF_GRID if args.fast else FULL_RF_GRID, verbose=0)
    print('\n--- Lasso-PCR ---');   lpcr = train_lasso_pcr(train_df, test_df, lasso, tscv)

    results = {'Lasso': lasso, 'PCR': pcr, 'Random Forest': rf, 'Lasso-PCR': lpcr}
    summary = pd.DataFrame({
        'Model': list(results),
        'Train FMSE': [r['train_fmse'] for r in results.values()],
        'Test FMSE': [r['test_fmse'] for r in results.values()],
    })
    summary['Gen. Gap'] = summary['Test FMSE'] - summary['Train FMSE']
    summary.to_csv(out / 'summary.csv', index=False)
    print('\n' + summary.to_string(index=False, float_format=lambda v: f'{v:,.2f}'))

    plot_model_comparison(summary, out)
    plot_cv_scheme(train_df, test_df, tscv, out)
    highlight = None
    if is_synth:
        from src.synthetic import true_predictors
        highlight = true_predictors()
    rho, p = plot_importance_agreement(lasso, rf, out, highlight)
    plot_residuals(results, y_test, out)
    print(f'\nSpearman rho (Lasso vs RF importance): {rho:.3f} (p = {p:.3f})')

    if is_synth:
        from src.synthetic import true_predictors
        true = set(true_predictors())
        kept = set(lasso['retained_z'])
        top_rf = set(rf['imp_gini'].reindex(FEATURE_COLS).nlargest(len(true)).index)
        print(f'True predictors        : {sorted(true, key=lambda s: int(s[1:]))}')
        print(f'Lasso kept, true found : {len(kept & true)}/{len(true)} '
              f'(plus {len(kept - true)} noise features kept)')
        print(f'RF top-{len(true)}, true found  : {len(top_rf & true)}/{len(true)}')


if __name__ == '__main__':
    main()
