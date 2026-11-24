"""Metrics and coefficient tables."""

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from . import config


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def cv_r2(model, X, y) -> tuple[float, float]:
    folds = KFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=folds, scoring="r2")
    return float(scores.mean()), float(scores.std())


def coefficients(model) -> pd.DataFrame:
    """Coefficients on the standardized feature scale, largest magnitude first."""
    pipeline = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
