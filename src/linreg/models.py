"""Linear Regression model variants, one per group member (5 members -> 5 models).

All models are linear in their parameters. Names follow the team moodboard
(M1, M3h, M8b). TODO(team): the moodboard numbers Ridge/Bayesian as M4/M5 in
section 06 but M5/M6 in section 09; pick one scheme and use it in the report.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import INTERACTIONS, REFERENCE

ALPHAS = np.logspace(-3, 3, 50)

DESIGN_NUMERIC = ["ENGINESIZE", "CYLINDERS", "HYBRID"]
DESIGN_CATEGORICAL = ["FUELTYPE", "BODY", "TRANS_TYPE"]


@dataclass(frozen=True)
class ModelSpec:
    description: str
    regressor: Callable
    numeric: list[str]
    categorical: list[str] = field(default_factory=list)
    # OLS stays on raw units so coefficients read as g/km per unit;
    # penalized/Bayesian models need a common scale for the prior/penalty.
    scale: bool = False


MODELS: dict[str, ModelSpec] = {
    "m1_engine": ModelSpec(
        "M1: OLS on engine size only, the one-variable baseline",
        LinearRegression, ["ENGINESIZE"],
    ),
    "m3h_ols": ModelSpec(
        "M3h: OLS on the design features (engine, cylinders, fuel, body, transmission, hybrid)",
        LinearRegression, DESIGN_NUMERIC, DESIGN_CATEGORICAL,
    ),
    "m3h_ridge": ModelSpec(
        "Ridge on the M3h features, alpha chosen by CV (L2 / MAP with a Gaussian prior)",
        lambda: RidgeCV(alphas=ALPHAS), DESIGN_NUMERIC, DESIGN_CATEGORICAL, scale=True,
    ),
    "m3h_bayes": ModelSpec(
        "Bayesian linear regression on the M3h features, prior precision by evidence maximization",
        BayesianRidge, DESIGN_NUMERIC, DESIGN_CATEGORICAL, scale=True,
    ),
    "m8b_interactions": ModelSpec(
        "M8b: M3h + engine size x (pickup, van, hybrid) interactions",
        LinearRegression, DESIGN_NUMERIC + INTERACTIONS, DESIGN_CATEGORICAL,
    ),
}


def build_model(name: str) -> Pipeline:
    spec = MODELS[name]
    numeric = StandardScaler() if spec.scale else "passthrough"
    encoder = OneHotEncoder(
        drop=[REFERENCE[c] for c in spec.categorical] or None,
        handle_unknown="ignore", sparse_output=False,
    )
    prep = ColumnTransformer(
        [("num", numeric, spec.numeric), ("cat", encoder, spec.categorical)],
        verbose_feature_names_out=False,
    )
    return Pipeline([("prep", prep), ("model", spec.regressor())])
