"""Frequentist inference on the OLS models: t/F tests, confidence intervals, diagnostics.

Formulas are generated from MODELS so the statsmodels fit uses exactly the same
design as the sklearn pipeline. Residuals are heteroscedastic (Breusch-Pagan),
so coefficient tests use HC3 robust standard errors by default.
"""

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

from . import config
from .features import REFERENCE
from .models import MODELS


def formula(name: str) -> str:
    spec = MODELS[name]
    if not spec.ols:
        raise ValueError(f"{name} has no OLS formula")
    terms = spec.numeric + [f"C({c}, Treatment('{REFERENCE[c]}'))" for c in spec.categorical]
    return f"{config.DATASET['target']} ~ " + " + ".join(terms)


def fit_ols(df, name: str = "M3h", cov_type: str = "HC3"):
    return smf.ols(formula(name), data=df).fit(cov_type=cov_type)


def partial_f_test(df, small: str, large: str):
    """F test of nested models (classical errors), e.g. M3 -> M3h."""
    return anova_lm(fit_ols(df, small, "nonrobust"), fit_ols(df, large, "nonrobust"))


def residual_diagnostics(result) -> dict[str, float]:
    _, bp_p, bp_f, _ = het_breuschpagan(result.resid, result.model.exog)
    jb, jb_p, _, _ = jarque_bera(result.resid)
    return {"breusch_pagan_LM_p": bp_p, "breusch_pagan_F": bp_f, "jarque_bera": jb, "jarque_bera_p": jb_p}
