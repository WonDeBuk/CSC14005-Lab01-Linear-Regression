"""Export the data behind every report chart for pgfplots.

The charts themselves are drawn in LaTeX (figures/tikz/*.tex); this script only
writes the numbers they plot, so every figure stays generated from code.

Run after analysis/analyze.py:
    .venv/bin/python -m analysis.plot_data

Outputs:
    figures/data/*.dat     whitespace-separated tables read by \\addplot table
    figures/data/plotvals.tex   scalar values (\\def\\pv...) used inside the charts
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import KFold

from linreg import config
from linreg.bayes import NIGRegression
from linreg.data import load_dataset, load_frame
from linreg.inference import fit_ols
from linreg.models import build_model

ROOT = config.ROOT
DATA = ROOT / "figures" / "data"
OUT = ROOT / "analysis" / "out"
TARGET = config.DATASET["target"]
SEED = config.RANDOM_STATE

DATA.mkdir(parents=True, exist_ok=True)

df = load_frame()
raw = pd.read_csv(config.DATASET["file"])
ds = load_dataset()
train = pd.concat([ds.X_train, ds.y_train], axis=1)
results = json.loads((OUT / "results.json").read_text())
macros = results["macros"]
metrics_df = pd.DataFrame(results["metrics"]).set_index("model")
folds_df = pd.read_csv(OUT / "cv_folds.csv")

# Scalars used inside charts: \pv<Name> (plain number, dot decimal) or
# \pvt<Name> (formatted text, comma decimal) for labels.
VALS: dict[str, str] = {}


def num(name: str, x: float, d: int = 4) -> None:
    VALS[f"pv{name}"] = f"{x:.{d}f}"


def txt(name: str, x: float, d: int = 1) -> None:
    s = f"{abs(x):,.{d}f}".replace(",", " ").replace(".", ",").replace(" ", ".")
    VALS[f"pvt{name}"] = ("$-$" if x < 0 else "") + s


def macro_num(key: str) -> float:
    return float(macros[key].replace(".", "").replace(",", ".").replace("−", "-"))


def write(name: str, frame: pd.DataFrame, digits: int = 5) -> None:
    frame.to_csv(DATA / f"{name}.dat", sep=" ", index=False, float_format=f"%.{digits}f")


def box_stats(values: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Matplotlib-style box: quartiles, 1.5 IQR whiskers, outliers."""
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
    out = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
    return {"lw": inside.min(), "lq": q1, "med": med, "uq": q3, "uw": inside.max()}, out


def boxes(groups: list[tuple[str, np.ndarray]], name: str) -> None:
    rows, outs = [], []
    for i, (label, v) in enumerate(groups, start=1):
        s, out = box_stats(v)
        rows.append({"pos": i, "label": label, "n": len(v), **s})
        outs += [{"pos": i, "y": o} for o in out]
    write(name, pd.DataFrame(rows))
    write(f"{name}-out", pd.DataFrame(outs, columns=["pos", "y"]))


# =========================================================================
# Fig 02: leakage scatter and CO2 vs engine size by body group
# =========================================================================
def fig02() -> None:
    for f in ["D", "X", "Z", "E"]:
        sub = raw[raw["FUELTYPE"] == f]
        write(f"fig02-leak-{f}", sub[["FUELCONSUMPTION_COMB", TARGET]].set_axis(["fc", "co2"], axis=1), 2)
    for key in ["small", "midlarge", "suv", "pickup", "van"]:
        sub = df[df["BODY"] == key]
        write(f"fig02-body-{key}", sub[["ENGINESIZE", TARGET]].set_axis(["eng", "co2"], axis=1), 2)
    m1 = fit_ols(train, "M1", "nonrobust")
    num("MOneIntercept", m1.params["Intercept"])
    num("MOneSlope", m1.params["ENGINESIZE"])
    num("EngMin", df["ENGINESIZE"].min())
    num("EngMax", df["ENGINESIZE"].max())


# =========================================================================
# Fig 03: correlation matrix and CO2 by body group
# =========================================================================
def fig03() -> None:
    cols = ["CO2EMISSIONS", "ENGINESIZE", "CYLINDERS", "GEARS", "NOGEAR", "HYBRID"]
    corr = df[cols].corr().to_numpy()
    rows = [{"x": j, "y": i, "r": corr[i, j],
             "label": f"{corr[i, j]:.2f}".replace(".", ",").replace("-", "$-$")}
            for i in range(len(cols)) for j in range(len(cols))]
    write("fig03-corr", pd.DataFrame(rows))
    order = ["small", "midlarge", "suv", "pickup", "van"]
    boxes([(k, df.loc[df["BODY"] == k, TARGET].to_numpy(dtype=float)) for k in order], "fig03-box")


# =========================================================================
# Fig 04: hybrid flag before / after
# =========================================================================
def fig04() -> None:
    m3 = fit_ols(train, "M3", "nonrobust")
    fitted, resid = m3.fittedvalues.to_numpy(), m3.resid.to_numpy()
    named = train["HYBRID_NAME"].to_numpy() == 1
    missed = (train["HYBRID"].to_numpy() == 1) & ~named
    regular = train["HYBRID"].to_numpy() == 0
    for name, mask in [("regular", regular), ("named", named), ("missed", missed)]:
        write(f"fig04-{name}", pd.DataFrame({"fit": fitted[mask], "res": resid[mask]}), 3)

    h_old = sm.OLS.from_formula(
        f"{TARGET} ~ ENGINESIZE + CYLINDERS + GEARS + NOGEAR + HYBRID_NAME + "
        "C(FUELTYPE, Treatment('X')) + C(BODY, Treatment('small')) + C(TRANS_TYPE, Treatment('A'))",
        train,
    ).fit(cov_type="HC3")
    h_new = fit_ols(train, "M3h", "HC3")
    for tag, model, key in [("Old", h_old, "HYBRID_NAME"), ("New", h_new, "HYBRID")]:
        lo, hi = model.conf_int().loc[key]
        num(f"Hyb{tag}", model.params[key])
        num(f"Hyb{tag}Lo", lo)
        num(f"Hyb{tag}Hi", hi)


# =========================================================================
# Fig 05: model comparison (CV vs test RMSE + GP ceiling)
# =========================================================================
def fig05() -> None:
    order = ["M1", "M2", "M3", "M4", "M5", "M4log", "M3h", "M6", "M8b"]
    write("fig05-models", pd.DataFrame({
        "pos": np.arange(len(order)),
        "model": order,
        "cv": metrics_df.loc[order, "CV_RMSE_mean"].to_numpy(),
        "cvsd": metrics_df.loc[order, "CV_RMSE_std"].to_numpy(),
        "test": metrics_df.loc[order, "RMSE"].to_numpy(),
    }), 4)
    num("GpRmse", macro_num("gpRmse"))


# =========================================================================
# Fig 06: learning curve and bootstrap bias-variance decomposition
# =========================================================================
def fig06() -> None:
    bv = pd.read_csv(OUT / "bias_variance.csv")
    total = bv["bias2_noise"] + bv["variance"]
    write("fig06-bv", pd.DataFrame({
        "pos": np.arange(len(bv)),
        "model": bv["model"],
        "bias": bv["bias2_noise"],
        "var": bv["variance"],
        "total": total,
        "label": [f"{v / t * 100:.1f}".replace(".", ",") + r"\%" for v, t in zip(bv["variance"], total)],
    }), 3)

    fracs = np.linspace(0.15, 1.0, 8)
    kf = list(KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=SEED).split(ds.X_train))
    out = pd.DataFrame({"n": [int(round(len(kf[0][0]) * f)) for f in fracs]})
    for mname in ["M1", "M3h", "M8b"]:
        tr_m, va_m, va_s = [], [], []
        for frac in fracs:
            tr_f, va_f = [], []
            for fold_i, (tr_idx, va_idx) in enumerate(kf):
                n_sub = int(round(len(tr_idx) * frac))
                sub = np.random.default_rng(SEED + fold_i).choice(tr_idx, size=n_sub, replace=False)
                pipe = build_model(mname).fit(ds.X_train.iloc[sub], ds.y_train.iloc[sub])
                tr_f.append(np.sqrt(np.mean((ds.y_train.iloc[sub] - pipe.predict(ds.X_train.iloc[sub])) ** 2)))
                va_f.append(np.sqrt(np.mean((ds.y_train.iloc[va_idx] - pipe.predict(ds.X_train.iloc[va_idx])) ** 2)))
            tr_m.append(np.mean(tr_f))
            va_m.append(np.mean(va_f))
            va_s.append(np.std(va_f, ddof=1))
        va_m, va_s = np.array(va_m), np.array(va_s)
        out[f"{mname}tr"] = tr_m
        out[f"{mname}va"] = va_m
        out[f"{mname}lo"] = va_m - va_s
        out[f"{mname}hi"] = va_m + va_s
        txt(f"Lc{mname.replace('1', 'One').replace('3', 'Three').replace('8', 'Eight')}", va_m[-1], 2)
    write("fig07-learning", out, 4)


# =========================================================================
# Fig 08: residual diagnostics and PI calibration of M3h
# =========================================================================
def fig08() -> None:
    m3h = fit_ols(train, "M3h", "nonrobust")
    fitted, resid = m3h.fittedvalues.to_numpy(), m3h.resid.to_numpy()
    infl = m3h.get_influence()
    std_resid = infl.resid_studentized_internal
    hi_cook = infl.cooks_distance[0] > 4.0 / len(train)
    write("fig08-resid", pd.DataFrame({"fit": fitted[~hi_cook], "res": resid[~hi_cook]}), 3)
    write("fig08-resid-cook", pd.DataFrame({"fit": fitted[hi_cook], "res": resid[hi_cook]}), 3)
    VALS["pvtCookCount"] = str(int(hi_cook.sum()))

    bins = np.array_split(np.argsort(fitted), 9)
    bx = np.array([fitted[b].mean() for b in bins])
    bs = np.array([resid[b].std(ddof=1) for b in bins])
    write("fig08-band", pd.DataFrame({"x": bx, "lo": -1.96 * bs, "hi": 1.96 * bs}), 3)

    meta = raw.loc[ds.X_train.index, ["MAKE", "MODEL"]].reset_index(drop=True)
    for tag, i in zip(["A", "B"], np.argsort(resid)[-2:]):
        num(f"Top{tag}X", fitted[i], 2)
        num(f"Top{tag}Y", resid[i], 2)
        VALS[f"pvtTop{tag}Name"] = f"{meta.loc[i, 'MAKE'].title()} {meta.loc[i, 'MODEL']}"

    osm, osr = stats.probplot(std_resid, dist="norm", fit=False)
    band = 1.96 * np.sqrt(stats.norm.cdf(osm) * (1 - stats.norm.cdf(osm)) / len(osr)) / stats.norm.pdf(osm)
    write("fig08-qq", pd.DataFrame({"th": osm, "obs": osr, "lo": osm - band, "hi": osm + band}), 4)

    prep = build_model("M3h").named_steps["prep"].fit(ds.X_train)
    Xtr = sm.add_constant(prep.transform(ds.X_train))
    Xte = sm.add_constant(prep.transform(ds.X_test), has_constant="add")
    pred = sm.OLS(ds.y_train.to_numpy(), Xtr).fit().get_prediction(Xte)
    yhat, pi = pred.predicted_mean, pred.conf_int(obs=True, alpha=0.05)
    y_te = ds.y_test.to_numpy()
    inside = (y_te >= pi[:, 0]) & (y_te <= pi[:, 1])
    write("fig08-pi-in", pd.DataFrame({
        "yhat": yhat[inside], "y": y_te[inside],
        "em": yhat[inside] - pi[inside, 0], "ep": pi[inside, 1] - yhat[inside],
    }), 3)
    write("fig08-pi-out", pd.DataFrame({"yhat": yhat[~inside], "y": y_te[~inside]}), 3)
    VALS["pvtPiIn"] = str(int(inside.sum()))
    VALS["pvtPiOut"] = str(int((~inside).sum()))
    VALS["pvtPiN"] = str(len(y_te))

    qs = np.quantile(y_te, [0.25, 0.5, 0.75])
    for tag, q in zip(["One", "Two", "Three"], qs):
        VALS[f"pvtQ{tag}"] = str(int(round(q)))
    write("fig08-cover", pd.DataFrame({
        "pos": np.arange(4),
        "cover": [macro_num(k) for k in ["piCoverQOne", "piCoverQTwo", "piCoverQThree", "piCoverQFour"]],
    }), 2)


# =========================================================================
# Fig 09: M8b interaction slopes
# =========================================================================
def fig09() -> None:
    m8 = fit_ols(train, "M8b", "HC3")
    eng = np.linspace(1.5, 6.2, 60)
    for tag, body, hyb in [("van", "van", 0), ("small", "small", 0), ("pickup", "pickup", 0), ("hybrid", "small", 1)]:
        grid = pd.DataFrame({
            "ENGINESIZE": eng, "CYLINDERS": 6, "GEARS": 6, "NOGEAR": 0, "HYBRID": hyb,
            "FUELTYPE": "X", "BODY": body, "TRANS_TYPE": "AS",
            "ENGINE_X_PICKUP": eng * (body == "pickup"),
            "ENGINE_X_VAN": eng * (body == "van"),
            "ENGINE_X_HYBRID": eng * hyb,
        })
        sf = m8.get_prediction(grid).summary_frame(alpha=0.05)
        write(f"fig09-{tag}", pd.DataFrame({
            "eng": eng, "mean": sf["mean"].to_numpy(),
            "lo": sf["mean_ci_lower"].to_numpy(), "hi": sf["mean_ci_upper"].to_numpy(),
        }), 3)


# =========================================================================
# Fig 10: Simpson's paradox for E85
# =========================================================================
def fig10() -> None:
    fuels = ["D", "X", "Z", "E"]
    write("fig10-fuel", pd.DataFrame({
        "pos": np.arange(4),
        "fuel": fuels,
        "co2": [df.loc[df["FUELTYPE"] == f, TARGET].mean() for f in fuels],
        "eng": [df.loc[df["FUELTYPE"] == f, "ENGINESIZE"].mean() for f in fuels],
    }), 3)
    for f in fuels:
        num(f"Co{f}", df.loc[df["FUELTYPE"] == f, TARGET].mean(), 2)
        txt(f"Co{f}", df.loc[df["FUELTYPE"] == f, TARGET].mean(), 1)
        txt(f"Eng{f}", df.loc[df["FUELTYPE"] == f, "ENGINESIZE"].mean(), 2)

    m3h = fit_ols(train, "M3h", "HC3")
    for f in ["D", "Z", "E"]:
        k = f"C(FUELTYPE, Treatment('X'))[T.{f}]"
        lo, hi = m3h.conf_int().loc[k]
        num(f"Fuel{f}", m3h.params[k])
        num(f"Fuel{f}Lo", lo)
        num(f"Fuel{f}Hi", hi)
        txt(f"Fuel{f}", m3h.params[k], 1)

    keys = ["MAKE", "MODEL", "ENGINESIZE", "CYLINDERS", "TRANSMISSION"]
    e = raw[raw["FUELTYPE"] == "E"].groupby(keys)[TARGET].mean()
    g = raw[raw["FUELTYPE"] == "X"].groupby(keys)[TARGET].mean()
    pairs = pd.concat([e.rename("e85"), g.rename("x")], axis=1, join="inner").dropna()
    diff = (pairs["e85"] - pairs["x"]).to_numpy()
    counts, edges = np.histogram(diff, bins=14)
    write("fig10-pairs", pd.DataFrame({"x": edges, "n": np.append(counts, counts[-1])}), 3)
    num("PairMean", diff.mean())
    txt("PairMean", diff.mean(), 1)
    VALS["pvtPairN"] = str(len(diff))


# =========================================================================
# Fig 11: frequentist vs Bayesian intervals and prior sensitivity
# =========================================================================
def fig11() -> None:
    prep = build_model("M3h").named_steps["prep"].fit(ds.X_train)
    names = list(prep.get_feature_names_out())
    X = prep.transform(ds.X_train)
    y = ds.y_train.to_numpy()
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    hc3 = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC3")
    cri = NIGRegression(tau=1e6).fit(X, y).credible_intervals()
    show = ["BODY_van", "BODY_pickup", "BODY_suv", "ENGINESIZE", "FUELTYPE_Z",
            "CYLINDERS", "GEARS", "TRANS_TYPE_AV", "FUELTYPE_E", "HYBRID"]
    rows = []
    for pos, f in enumerate(show):
        i = names.index(f) + 1
        rows.append({
            "pos": pos, "est": ols.params[i],
            "olo": ols.conf_int()[i][0], "ohi": ols.conf_int()[i][1],
            "hlo": hc3.conf_int()[i][0], "hhi": hc3.conf_int()[i][1],
            "blo": cri[i][0], "bhi": cri[i][1],
        })
    write("fig11-ci", pd.DataFrame(rows), 4)

    sens = pd.read_csv(OUT / "prior_sensitivity.csv")
    for tag in ["full", "n40"]:
        write(f"fig11-prior-{tag}", sens[sens["data"] == tag][["tau", "mean", "lo", "hi"]], 4)


if __name__ == "__main__":
    for fn in [fig02, fig03, fig04, fig05, fig06, fig08, fig09, fig10, fig11]:
        fn()
        print(f"Exported {fn.__name__}")
    lines = ["% Generated by analysis/plot_data.py. Do not edit by hand."]
    lines += [f"\\def\\{k}{{{v}}}" for k, v in sorted(VALS.items())]
    (DATA / "plotvals.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(VALS)} values to {DATA / 'plotvals.tex'}")
