"""
models.py
---------
Training, tuning, and evaluation for all four models:
    1. Lasso regression with time-aware CV
    2. Principal Component Regression (PCR)
    3. Random Forest with GridSearchCV
    4. Lasso-PCR extension (joint lambda x k selection)

All models use the same chronological 80/20 train/test split and the same
PanelTimeSeriesCV instance for inner-loop hyperparameter selection.
The test set is never used during tuning.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, lasso_path
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

from src.utils import FEATURE_COLS, CAT_COLS, RANDOM_STATE, fmse
from src.cv import PanelTimeSeriesCV


# ── Shared preprocessing helpers ──────────────────────────────────────────────

def _preprocess_linear(train_df, test_df):
    """
    Standardise numeric z-features and one-hot encode company_id.
    Fit exclusively on training data.

    Returns
    -------
    X_train, X_test, all_names, scaler, ohe
        Concatenated [numeric | dummy] matrices and the fitted transformers.
    """
    scaler = StandardScaler()
    ohe    = OneHotEncoder(drop='first', handle_unknown='ignore',
                           sparse_output=False)

    X_train_num = scaler.fit_transform(train_df[FEATURE_COLS])
    X_test_num  = scaler.transform(test_df[FEATURE_COLS])

    X_train_cat = ohe.fit_transform(train_df[CAT_COLS])
    X_test_cat  = ohe.transform(test_df[CAT_COLS])

    X_train = np.hstack([X_train_num, X_train_cat])
    X_test  = np.hstack([X_test_num,  X_test_cat])

    cat_names = list(ohe.get_feature_names_out(CAT_COLS))
    all_names = FEATURE_COLS + cat_names

    return X_train, X_test, X_train_num, X_test_num, X_train_cat, X_test_cat, \
           all_names, scaler, ohe


# ── Model 1: Lasso ────────────────────────────────────────────────────────────

def train_lasso(train_df, test_df, tscv,
                alphas=None, max_iter=50_000):
    """
    Fit Lasso with time-aware CV to select lambda*.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
    tscv : PanelTimeSeriesCV
    alphas : array-like, optional
        Grid of penalty values. Defaults to 100 log-spaced values in [1e-4, 1e2].
    max_iter : int

    Returns
    -------
    dict with keys:
        model         : fitted LassoCV object
        test_fmse     : float
        train_fmse    : float
        lambda_star   : float
        retained_z    : list of retained z-feature names
        zeroed_z      : list of zeroed z-feature names
        coef_df       : DataFrame of retained coefficients sorted by |coef|
        X_train, X_test, X_train_num, X_test_num, X_train_cat, X_test_cat
        all_names, scaler, ohe
        test_pred, train_pred
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 100)

    (X_train, X_test, X_train_num, X_test_num,
     X_train_cat, X_test_cat, all_names, scaler, ohe) = \
        _preprocess_linear(train_df, test_df)

    y_train = train_df['y_future'].values
    y_test  = test_df['y_future'].values

    model = LassoCV(alphas=alphas, cv=tscv, max_iter=max_iter,
                    random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    lambda_star = model.alpha_
    coef        = model.coef_

    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    retained_z = [n for n, c in zip(all_names, coef)
                  if c != 0.0 and n.startswith('z')]
    zeroed_z   = [n for n, c in zip(all_names, coef)
                  if c == 0.0 and n.startswith('z')]

    coef_df = pd.DataFrame({'Feature': all_names,
                            'Coefficient': np.round(coef, 4)}) \
                .query('Coefficient != 0') \
                .sort_values('Coefficient', key=abs, ascending=False)

    print(f'Lasso  lambda*    : {lambda_star:.6f}')
    print(f'Lasso  Train FMSE: {fmse(y_train, train_pred):.2f}')
    print(f'Lasso  Test FMSE : {fmse(y_test, test_pred):.2f}')
    print(f'Retained z-features ({len(retained_z)}/25): {retained_z}')
    print(f'Zeroed   z-features ({len(zeroed_z)}/25) : {zeroed_z}')

    return dict(model=model, test_fmse=fmse(y_test, test_pred),
                train_fmse=fmse(y_train, train_pred),
                lambda_star=lambda_star, retained_z=retained_z,
                zeroed_z=zeroed_z, coef_df=coef_df,
                X_train=X_train, X_test=X_test,
                X_train_num=X_train_num, X_test_num=X_test_num,
                X_train_cat=X_train_cat, X_test_cat=X_test_cat,
                all_names=all_names, scaler=scaler, ohe=ohe,
                test_pred=test_pred, train_pred=train_pred)


# ── Model 2: PCR ──────────────────────────────────────────────────────────────

def train_pcr(train_df, test_df, lasso_result,
              variance_threshold=0.95):
    """
    Fit PCA (95% cumulative variance threshold) + OLS on [PC scores | dummies].

    PCA is applied only to the 25 standardised numeric predictors;
    company dummies are appended directly to the OLS design matrix as
    fixed-effect regressors.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
    lasso_result : dict
        Output of train_lasso — reuses fitted scaler and OHE.
    variance_threshold : float

    Returns
    -------
    dict with keys: model, pca, test_fmse, train_fmse, n_comp, cumvar_at_threshold,
                    loadings_df, X_train_pcr, X_test_pcr, test_pred, train_pred
    """
    X_train_num = lasso_result['X_train_num']
    X_test_num  = lasso_result['X_test_num']
    X_train_cat = lasso_result['X_train_cat']
    X_test_cat  = lasso_result['X_test_cat']

    y_train = train_df['y_future'].values
    y_test  = test_df['y_future'].values

    pca_full = PCA(n_components=25, random_state=RANDOM_STATE)
    pca_full.fit(X_train_num)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, variance_threshold)) + 1
    cumvar_at_threshold = float(cumvar[n_comp - 1])

    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    Z_train = pca.fit_transform(X_train_num)
    Z_test  = pca.transform(X_test_num)

    X_train_pcr = np.hstack([Z_train, X_train_cat])
    X_test_pcr  = np.hstack([Z_test,  X_test_cat])

    model = LinearRegression()
    model.fit(X_train_pcr, y_train)

    train_pred = model.predict(X_train_pcr)
    test_pred  = model.predict(X_test_pcr)

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=FEATURE_COLS,
        columns=[f'PC{i+1}' for i in range(n_comp)]
    )

    print(f'PCR  n_components : {n_comp} ({cumvar_at_threshold*100:.2f}% variance)')
    print(f'PCR  Train FMSE   : {fmse(y_train, train_pred):.2f}')
    print(f'PCR  Test FMSE    : {fmse(y_test, test_pred):.2f}')

    return dict(model=model, pca=pca, test_fmse=fmse(y_test, test_pred),
                train_fmse=fmse(y_train, train_pred),
                n_comp=n_comp, cumvar_at_threshold=cumvar_at_threshold,
                loadings_df=loadings_df,
                X_train_pcr=X_train_pcr, X_test_pcr=X_test_pcr,
                test_pred=test_pred, train_pred=train_pred,
                pca_full=pca_full, cumvar=cumvar)


# ── Model 3: Random Forest ────────────────────────────────────────────────────

def train_rf(train_df, test_df, tscv):
    """
    Fit Random Forest via GridSearchCV (time-aware 5-fold, neg MSE scoring).

    Trees are scale-invariant so no standardisation is applied; company_id
    is one-hot encoded via a ColumnTransformer inside a Pipeline.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
    tscv : PanelTimeSeriesCV

    Returns
    -------
    dict with keys: model, grid_search, test_fmse, train_fmse, best_params,
                    imp_gini, imp_perm, X_train_rf, X_test_rf, rf_feature_names,
                    test_pred, train_pred
    """
    X_train_raw = train_df[FEATURE_COLS + CAT_COLS]
    X_test_raw  = test_df[FEATURE_COLS + CAT_COLS]
    y_train     = train_df['y_future'].values
    y_test      = test_df['y_future'].values

    preprocessor = ColumnTransformer([
        ('num', 'passthrough', FEATURE_COLS),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CAT_COLS)
    ])
    pipeline = Pipeline([('prep', preprocessor),
                         ('model', RandomForestRegressor(
                             random_state=RANDOM_STATE, n_jobs=1))])

    param_grid = {
        'model__n_estimators'    : [100, 200, 300, 500],
        'model__max_depth'       : [3, 5, 8, 10, None],
        'model__max_features'    : ['sqrt', 'log2', 0.3, 0.5],
        'model__min_samples_leaf': [1, 2, 5, 10],
    }
    print(f'Grid: {4*5*4*4} combinations × 5 folds = {4*5*4*4*5} fits')

    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, cv=tscv,
        scoring='neg_mean_squared_error', refit=True, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_raw, y_train)
    best_p = grid_search.best_params_

    # Refit with oob_score for in-bag generalisation estimate
    prep_final = ColumnTransformer([
        ('num', 'passthrough', FEATURE_COLS),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CAT_COLS)
    ])
    X_train_rf = prep_final.fit_transform(X_train_raw)
    X_test_rf  = prep_final.transform(X_test_raw)
    rf_feature_names = FEATURE_COLS + list(
        prep_final.named_transformers_['cat'].get_feature_names_out(CAT_COLS))

    model = RandomForestRegressor(
        n_estimators=best_p['model__n_estimators'],
        max_depth=best_p['model__max_depth'],
        max_features=best_p['model__max_features'],
        min_samples_leaf=best_p['model__min_samples_leaf'],
        oob_score=True, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train_rf, y_train)

    train_pred = model.predict(X_train_rf)
    test_pred  = model.predict(X_test_rf)

    imp_gini = pd.Series(model.feature_importances_, index=rf_feature_names)
    perm_res = permutation_importance(model, X_test_rf, y_test,
                                      n_repeats=10, random_state=RANDOM_STATE,
                                      n_jobs=-1)
    imp_perm = pd.Series(perm_res.importances_mean, index=rf_feature_names)

    print(f'RF   OOB R²       : {model.oob_score_:.4f}')
    print(f'RF   Train FMSE   : {fmse(y_train, train_pred):.2f}')
    print(f'RF   Test FMSE    : {fmse(y_test, test_pred):.2f}')
    print(f'Best params       : {best_p}')

    return dict(model=model, grid_search=grid_search,
                test_fmse=fmse(y_test, test_pred),
                train_fmse=fmse(y_train, train_pred),
                best_params=best_p, imp_gini=imp_gini, imp_perm=imp_perm,
                X_train_rf=X_train_rf, X_test_rf=X_test_rf,
                rf_feature_names=rf_feature_names,
                test_pred=test_pred, train_pred=train_pred)


# ── Extension: Lasso-PCR ──────────────────────────────────────────────────────

def train_lasso_pcr(train_df, test_df, lasso_result, tscv,
                    alphas=None, k_max=15):
    """
    Lasso-PCR: joint CV over (lambda, k) within the training set.

    Workflow per fold:
        1. Fit Lasso(lambda) on fold training portion -> select z-features.
        2. Fit PCA(k) on selected features -> PC scores.
        3. Fit OLS on [PC scores | company dummies].
    Company dummies are excluded from PCA and passed directly to OLS.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
    lasso_result : dict
        Reuses pre-computed standardised arrays and OHE.
    tscv : PanelTimeSeriesCV
    alphas : array-like, optional. Default: 25 log-spaced values in [1e-4, 1e2].
    k_max : int. Maximum number of PCs to consider. Default 15.

    Returns
    -------
    dict with keys: test_fmse, train_fmse, lambda_star, k_star,
                    n_retained, test_pred, train_pred, cv_grid (mean FMSE array)
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 25)

    X_train_num = lasso_result['X_train_num']
    X_test_num  = lasso_result['X_test_num']
    X_train_cat = lasso_result['X_train_cat']
    X_test_cat  = lasso_result['X_test_cat']

    y_train = train_df['y_future'].values
    y_test  = test_df['y_future'].values

    EPS    = 1e-8
    folds  = list(tscv.split(X_train_num))
    n_a    = len(alphas)
    cv_grid = [[[] for _ in range(k_max)] for _ in range(n_a)]

    for ai, lam in enumerate(alphas):
        for tr, va in folds:
            las = Lasso(alpha=lam, max_iter=50_000)
            las.fit(X_train_num[tr], y_train[tr])
            mask = np.abs(las.coef_) > EPS
            n_sel = int(mask.sum())

            if n_sel == 0:
                for ki in range(k_max):
                    cv_grid[ai][ki].append(1e12)
                continue

            kmax_f   = min(n_sel, k_max)
            pca_tmp  = PCA(n_components=kmax_f, random_state=RANDOM_STATE)
            Ztr_all  = pca_tmp.fit_transform(X_train_num[tr][:, mask])
            Zva_all  = pca_tmp.transform(X_train_num[va][:, mask])

            for ki in range(k_max):
                k = ki + 1
                if k > kmax_f:
                    cv_grid[ai][ki].append(1e12)
                    continue
                ols = LinearRegression()
                ols.fit(np.hstack([Ztr_all[:, :k], X_train_cat[tr]]),
                        y_train[tr])
                cv_grid[ai][ki].append(
                    fmse(y_train[va],
                         ols.predict(np.hstack([Zva_all[:, :k],
                                                X_train_cat[va]]))))

    cv_mean = np.array([[np.mean(v) for v in row] for row in cv_grid])
    best_ai, best_ki = divmod(int(np.argmin(cv_mean)), k_max)
    best_lam = float(alphas[best_ai])
    best_k   = best_ki + 1

    # Refit on full training set
    las_final = Lasso(alpha=best_lam, max_iter=50_000)
    las_final.fit(X_train_num, y_train)
    mask_final = np.abs(las_final.coef_) > EPS

    pca_final = PCA(n_components=best_k, random_state=RANDOM_STATE)
    Z_tr = pca_final.fit_transform(X_train_num[:, mask_final])
    Z_te = pca_final.transform(X_test_num[:, mask_final])

    ols_final = LinearRegression()
    ols_final.fit(np.hstack([Z_tr, X_train_cat]), y_train)

    train_pred = ols_final.predict(np.hstack([Z_tr, X_train_cat]))
    test_pred  = ols_final.predict(np.hstack([Z_te, X_test_cat]))

    print(f'Lasso-PCR lambda* : {best_lam:.6f}  |  k* : {best_k}')
    print(f'Retained z-features: {int(mask_final.sum())}/25')
    print(f'Lasso-PCR Train FMSE: {fmse(y_train, train_pred):.2f}')
    print(f'Lasso-PCR Test FMSE : {fmse(y_test, test_pred):.2f}')

    return dict(test_fmse=fmse(y_test, test_pred),
                train_fmse=fmse(y_train, train_pred),
                lambda_star=best_lam, k_star=best_k,
                n_retained=int(mask_final.sum()),
                test_pred=test_pred, train_pred=train_pred,
                cv_grid=cv_mean,
                best_ai=best_ai, best_ki=best_ki,
                alphas=alphas, k_max=k_max)
