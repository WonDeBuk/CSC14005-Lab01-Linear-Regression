"""Metrics and coefficient tables."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from . import config


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def cv_scores(model, X, y) -> dict[str, float]:
    folds = KFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=folds, scoring={"RMSE": "neg_root_mean_squared_error", "R2": "r2"})
    return {
        "CV_RMSE_mean": float(-scores["test_RMSE"].mean()),
        "CV_RMSE_std": float(scores["test_RMSE"].std()),
        "CV_R2_mean": float(scores["test_R2"].mean()),
    }


def coefficients(model) -> pd.DataFrame:
    """Coefficients in model units (raw g/km per unit for OLS, per SD for scaled models)."""
    names = model.named_steps["prep"].get_feature_names_out()
    table = pd.DataFrame({"feature": names, "coefficient": model.named_steps["model"].coef_})
    intercept = pd.DataFrame({"feature": ["(intercept)"], "coefficient": [model.named_steps["model"].intercept_]})
    return pd.concat([intercept, table], ignore_index=True)
