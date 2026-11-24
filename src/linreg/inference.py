"""Frequentist inference on the OLS models (t/F tests, confidence intervals).

Residuals are heteroscedastic (Breusch-Pagan), so tests use HC3 robust
standard errors by default.
"""

import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan

DESIGN = (
    "ENGINESIZE + CYLINDERS + HYBRID"
    " + C(FUELTYPE, Treatment('X')) + C(BODY, Treatment('car')) + C(TRANS_TYPE, Treatment('A'))"
)
FORMULAS = {
    "m1_engine": "CO2EMISSIONS ~ ENGINESIZE",
    "m3h_ols": f"CO2EMISSIONS ~ {DESIGN}",
    "m8b_interactions": f"CO2EMISSIONS ~ {DESIGN} + ENGINE_X_PICKUP + ENGINE_X_VAN + ENGINE_X_HYBRID",
}


def fit_ols(df, name="m3h_ols", cov_type="HC3"):
    return smf.ols(FORMULAS[name], data=df).fit(cov_type=cov_type)


def breusch_pagan(result) -> dict[str, float]:
    lm, lm_p, f, f_p = het_breuschpagan(result.resid, result.model.exog)
    return {"LM": lm, "LM_p": lm_p, "F": f, "F_p": f_p}
