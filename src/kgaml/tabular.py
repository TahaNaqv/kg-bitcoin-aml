"""Classical baselines: Logistic Regression, Random Forest, XGBoost."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

N_JOBS = 4


def make_model(name: str, seed: int, pos_weight: float, weighted: bool = True):
    if not weighted:
        pos_weight = 1.0
    if name == "LR":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(
            class_weight="balanced" if weighted else None, max_iter=3000, random_state=seed))
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=500, max_features="sqrt", class_weight="balanced_subsample" if weighted else None,
            n_jobs=N_JOBS, random_state=seed)
    if name == "XGB":
        return XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, tree_method="hist",
            eval_metric="aucpr", early_stopping_rounds=100, n_jobs=N_JOBS,
            random_state=seed)
    raise ValueError(name)


def fit_predict(name: str, seed: int, x: pd.DataFrame, y: np.ndarray,
                tr: np.ndarray, va: np.ndarray, sample_xy=None, weighted: bool = True):
    """Fit on the training window; return probabilities for all rows and fit time.

    ``sample_xy`` optionally replaces the training rows (e.g. after SMOTE).
    """
    xtr, ytr = (x[tr], y[tr]) if sample_xy is None else sample_xy
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    model = make_model(name, seed, pos_weight, weighted)
    t0 = time.perf_counter()
    if name == "XGB":
        model.fit(xtr, ytr, eval_set=[(x[va], y[va])], verbose=False)
    else:
        model.fit(xtr, ytr)
    fit_s = time.perf_counter() - t0
    return model, model.predict_proba(x)[:, 1], fit_s
