"""Linear Regression models, named as in section 09 of the team moodboard.

Every model is linear in its parameters. M3h uses the corrected hybrid flag
(called M3h' in section 12). M7 (fuel consumption x fuel type) is left out on
purpose: it leaks the target (section 08, P2).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .bayes import NIGRegression
from .features import REFERENCE

M1 = ["ENGINESIZE"]
M2 = M1 + ["CYLINDERS"]
M3_NUMERIC = M2 + ["GEARS", "NOGEAR"]
CATEGORICAL = ["FUELTYPE", "BODY", "TRANS_TYPE"]
M4_NUMERIC = M3_NUMERIC + ["ENGINESIZE_SQ", "ENGINE_X_DIESEL", "ENGINE_X_E85"]
M3H_NUMERIC = M3_NUMERIC + ["HYBRID"]
M8B_NUMERIC = M3H_NUMERIC + ["ENGINE_X_PICKUP", "ENGINE_X_VAN", "ENGINE_X_HYBRID"]


class LogSmearingRegressor(RegressorMixin, BaseEstimator):
    """Fit on log(y), predict exp(pred) times Duan's smearing factor."""

    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor or LinearRegression()).fit(X, np.log(y))
        self.smearing_ = float(np.mean(np.exp(np.log(y) - self.regressor_.predict(X))))
        self.coef_, self.intercept_ = self.regressor_.coef_, self.regressor_.intercept_
        return self

    def predict(self, X):
        return np.exp(self.regressor_.predict(X)) * self.smearing_


@dataclass(frozen=True)
class ModelSpec:
    description: str
    regressor: Callable
    numeric: list[str]
    categorical: list[str] = field(default_factory=list)
    # OLS stays on raw units so coefficients read as g/km per unit; Ridge and the
    # Bayesian prior need standardized inputs (P9). Coefficients are mapped back.
    scale: bool = False
    ols: bool = True  # has a statsmodels counterpart in inference.py


MODELS: dict[str, ModelSpec] = {
    "M1": ModelSpec("Hồi quy đơn theo dung tích động cơ (mốc)", LinearRegression, M1),
    "M2": ModelSpec("M1 + số xi-lanh", LinearRegression, M2),
    "M3": ModelSpec("M2 + nhiên liệu, nhóm thân xe, hộp số", LinearRegression, M3_NUMERIC, CATEGORICAL),
    "M3h": ModelSpec("M3 + cờ HYBRID (đã sửa nhãn)", LinearRegression, M3H_NUMERIC, CATEGORICAL),
    "M4": ModelSpec(
        "M3 + dung tích², dung tích × (diesel, E85)", LinearRegression, M4_NUMERIC, CATEGORICAL
    ),
    "M4log": ModelSpec(
        "M4 trên log(CO₂), đổi ngược bằng smearing của Duan",
        LogSmearingRegressor, M4_NUMERIC, CATEGORICAL, ols=False,
    ),
    "M5": ModelSpec(
        "Ridge trên biến M4, λ = 3 (MAP với prior Gauss)",
        lambda: Ridge(alpha=3.0), M4_NUMERIC, CATEGORICAL, scale=True, ols=False,
    ),
    "M6": ModelSpec(
        "Bayesian Normal-Inverse-Gamma trên biến M3h, prior yếu (τ = 1000)",
        NIGRegression, M3H_NUMERIC, CATEGORICAL, scale=True, ols=False,
    ),
    "M8b": ModelSpec(
        "M3h + dung tích × (bán tải, van, hybrid)", LinearRegression, M8B_NUMERIC, CATEGORICAL
    ),
}


def build_model(name: str) -> Pipeline:
    spec = MODELS[name]
    encoder = OneHotEncoder(
        drop=[REFERENCE[c] for c in spec.categorical] or None,
        handle_unknown="ignore", sparse_output=False,
    )
    prep = ColumnTransformer(
        [("num", "passthrough", spec.numeric), ("cat", encoder, spec.categorical)],
        verbose_feature_names_out=False,
    )
    steps = [("prep", prep)]
    if spec.scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", spec.regressor()))
    return Pipeline(steps)
