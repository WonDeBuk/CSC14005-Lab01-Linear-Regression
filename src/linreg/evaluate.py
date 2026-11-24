"""Metrics, model comparison and coefficient tables."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from . import config


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def cv_rmse_folds(model, X, y) -> np.ndarray:
    """RMSE of each fold of a shuffled K-fold on the training set."""
    folds = KFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = []
    for train, val in folds.split(X):
        fitted = clone(model).fit(X.iloc[train], y.iloc[train])
        scores.append(np.sqrt(mean_squared_error(y.iloc[val], fitted.predict(X.iloc[val]))))
    return np.array(scores)


def nadeau_bengio_test(rmse_a: np.ndarray, rmse_b: np.ndarray, n_train: int) -> dict[str, float]:
    """Corrected resampled t-test (Nadeau & Bengio, 2003) on per-fold RMSE.

    Folds share training data, so the plain paired t-test understates the variance;
    the correction inflates it by (1/k + n_val/n_fit).
    """
    diff = rmse_a - rmse_b
    k = len(diff)
    n_val = n_train / k
    var = diff.var(ddof=1) * (1 / k + n_val / (n_train - n_val))
    t = diff.mean() / np.sqrt(var)
    return {"mean_diff": float(diff.mean()), "t": float(t), "p": float(2 * stats.t.sf(abs(t), k - 1))}


def coefficients(model) -> pd.DataFrame:
    """Coefficients in g/km per original unit (standardized models are mapped back)."""
    names = model.named_steps["prep"].get_feature_names_out()
    est = model.named_steps["model"]
    coef, intercept = np.asarray(est.coef_, dtype=float), float(est.intercept_)
    if "scale" in model.named_steps:
        scaler = model.named_steps["scale"]
        coef = coef / scaler.scale_
        intercept -= float(coef @ scaler.mean_)
    return pd.DataFrame({"feature": ["(intercept)", *names], "coefficient": [intercept, *coef]})
