"""Conjugate Bayesian linear regression (model M6).

Normal-Inverse-Gamma prior:  sigma^2 ~ IG(a0, b0),  beta | sigma^2 ~ N(0, sigma^2 tau^2 I).
The intercept gets a flat prior. Inputs are expected to be standardized, so tau
is the prior scale of a standardized coefficient (in units of sigma): small tau
shrinks hard, large tau approaches OLS.
"""

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin


class NIGRegression(RegressorMixin, BaseEstimator):
    def __init__(self, tau: float = 1000.0, a0: float = 1e-3, b0: float = 1e-3):
        self.tau = tau
        self.a0 = a0
        self.b0 = b0

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        Z = np.column_stack([np.ones(len(X)), X])
        prior_precision = np.diag([0.0] + [1.0 / self.tau**2] * X.shape[1])
        precision = prior_precision + Z.T @ Z
        self.V_ = np.linalg.inv(precision)
        self.m_ = self.V_ @ Z.T @ y
        self.a_ = self.a0 + len(y) / 2
        self.b_ = self.b0 + 0.5 * (y @ y - self.m_ @ precision @ self.m_)
        self.intercept_, self.coef_ = self.m_[0], self.m_[1:]
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_

    def credible_intervals(self, level: float = 0.95) -> np.ndarray:
        """Marginal posterior of each coefficient is Student-t with 2a_n dof."""
        scale = np.sqrt(self.b_ / self.a_ * np.diag(self.V_))
        q = stats.t.ppf(0.5 + level / 2, 2 * self.a_)
        return np.column_stack([self.m_ - q * scale, self.m_ + q * scale])

    def predict_interval(self, X, level: float = 0.95) -> np.ndarray:
        """Posterior predictive interval for a new car."""
        Z = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        mean = Z @ self.m_
        scale = np.sqrt(self.b_ / self.a_ * (1 + np.einsum("ij,jk,ik->i", Z, self.V_, Z)))
        q = stats.t.ppf(0.5 + level / 2, 2 * self.a_)
        return np.column_stack([mean - q * scale, mean + q * scale])
