"""Linear Regression model variants.

The lab requires at least one model per group member. Every entry in MODELS is a
linear model in its parameters; keep, drop or add entries so the count matches
the group, and justify each choice in the report (section "Thiết kế mô hình").
"""

from collections.abc import Callable

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LassoCV, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from . import config

ALPHAS = np.logspace(-3, 3, 50)


def build_preprocessor(numeric: list[str], categorical: list[str], poly_degree: int = 1) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if poly_degree > 1:
        numeric_steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ]
    return ColumnTransformer(
        [("num", Pipeline(numeric_steps), numeric), ("cat", Pipeline(categorical_steps), categorical)],
        verbose_feature_names_out=False,
    )


def _pipeline(regressor, numeric, categorical, poly_degree=1) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(numeric, categorical, poly_degree)), ("model", regressor)])


MODELS: dict[str, Callable[[list[str], list[str]], Pipeline]] = {
    # M1: plain OLS on all features, the baseline every other model is compared with.
    "ols": lambda num, cat: _pipeline(LinearRegression(), num, cat),
    # M2: L2 penalty, shrinks coefficients when features are correlated.
    "ridge": lambda num, cat: _pipeline(RidgeCV(alphas=ALPHAS), num, cat),
    # M3: L1 penalty, drives weak coefficients to exactly zero (feature selection).
    "lasso": lambda num, cat: _pipeline(
        LassoCV(alphas=ALPHAS, cv=config.CV_FOLDS, random_state=config.RANDOM_STATE, max_iter=20_000), num, cat
    ),
    # M4: mix of L1 and L2.
    "elasticnet": lambda num, cat: _pipeline(
        ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9], alphas=ALPHAS, cv=config.CV_FOLDS,
            random_state=config.RANDOM_STATE, max_iter=20_000,
        ),
        num, cat,
    ),
    # M5: degree-2 polynomial basis on numeric features + Ridge (still linear in parameters).
    "poly2_ridge": lambda num, cat: _pipeline(RidgeCV(alphas=ALPHAS), num, cat, poly_degree=2),
    # M6: OLS on log(1 + y), for a right-skewed positive target.
    "log_target_ols": lambda num, cat: TransformedTargetRegressor(
        regressor=_pipeline(LinearRegression(), num, cat), func=np.log1p, inverse_func=np.expm1
    ),
    # M7: Huber loss, less sensitive to outliers in y.
    "huber": lambda num, cat: _pipeline(HuberRegressor(max_iter=2_000), num, cat),
}


def build_model(name: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    return MODELS[name](numeric, categorical)
