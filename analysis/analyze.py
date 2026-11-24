"""Full reproducible statistical analysis for CSC14005 Lab 01 (Linear Regression).

Run:
    .venv/bin/python analysis/analyze.py

Outputs:
    - analysis/out/results.json          (complete numerical record)
    - analysis/out/cv_folds.csv          (10-fold CV RMSE per model)
    - analysis/out/prior_sensitivity.csv (Bayesian prior sensitivity n=854 vs n=40)
    - analysis/out/ridge_path.csv        (Ridge CV RMSE and coefficients vs lambda)
    - analysis/out/bias_variance.csv     (Bootstrap B=200 Bias^2-Variance decomposition)
    - analysis/out/robustness.json       (Split ratio & Gaussian noise sensitivity)
    - tables/*.tex                       (publication-ready LaTeX tables)
    - tables/numbers.tex                 (\\newcommand macros for every reported number)

All random operations use config.RANDOM_STATE (42).
"""

from __future__ import annotations

import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

from linreg import config
from linreg.bayes import NIGRegression
from linreg.data import load_dataset, load_frame
from linreg.evaluate import cv_rmse_folds, nadeau_bengio_test
from linreg.features import BODY_GROUPS, BODY_LABELS, FUEL_LABELS, TRANS_LABELS
from linreg.inference import fit_ols, partial_f_test
from linreg.models import MODELS, build_model
from linreg.train import train_all

ROOT = config.ROOT
OUT = ROOT / "analysis" / "out"
TABLES = ROOT / "tables"
TARGET = config.DATASET["target"]
SEED = config.RANDOM_STATE

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def vn(x: float, d: int = 1) -> str:
    """Vietnamese number format: '.' thousands, ',' decimals, true minus."""
    s = f"{abs(x):,.{d}f}".replace(",", " ").replace(".", ",").replace(" ", ".")
    neg = x < 0 and float(f"{abs(x):.{d}f}") != 0
    return ("−" if neg else "") + s


def pval(p: float) -> str:
    return "< 0,001" if p < 0.001 else vn(p, 3)


def tex_text(s: str) -> str:
    replacements = [
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("CO₂", r"\COtwo{}"),
        ("λ", r"$\lambda$"),
        ("τ", r"$\tau$"),
        ("²", r"$^2$"),
        ("×", r"$\times$"),
        ("→", r"$\to$"),
        ("±", r"$\pm$"),
    ]
    out = s
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


macros: dict[str, str] = {}


def macro(name: str, value: str) -> None:
    assert name.isalpha(), f"Macro name must be alphabetic: {name}"
    macros[name] = value


# =========================================================================
# 1. Dataset & Exploratory Data Analysis
# =========================================================================
df = load_frame()
raw = pd.read_csv(config.DATASET["file"])
ds = load_dataset()
train = pd.concat([ds.X_train, ds.y_train], axis=1)
test = pd.concat([ds.X_test, ds.y_test], axis=1)
y = df[TARGET]

macro("nCars", vn(len(df), 0))
macro("nTrain", vn(len(train), 0))
macro("nTest", vn(len(test), 0))
macro("nCols", str(raw.shape[1]))
macro("nMakes", str(raw["MAKE"].nunique()))
macro("nVehicleClasses", str(raw["VEHICLECLASS"].nunique()))
macro("nTransCodes", str(raw["TRANSMISSION"].nunique()))
vc_counts = raw["VEHICLECLASS"].value_counts()
macro("nSparseClasses", str(int((vc_counts < 15).sum())))
macro("minClassCount", str(int(vc_counts.min())))
macro("maxClassCount", str(int(vc_counts.max())))
macro("coMean", vn(y.mean(), 1))
macro("coMedian", vn(y.median(), 0))
macro("coSd", vn(y.std(ddof=1), 1))
macro("coMin", vn(y.min(), 0))
macro("coMax", vn(y.max(), 0))
macro("coQOne", vn(np.quantile(y, 0.25), 0))
macro("coQThree", vn(np.quantile(y, 0.75), 0))
macro("coSkew", vn(stats.skew(y), 3))
macro("engMean", vn(df["ENGINESIZE"].mean(), 2))
macro("engSd", vn(df["ENGINESIZE"].std(ddof=1), 2))
macro("engMin", vn(df["ENGINESIZE"].min(), 1))
macro("engMax", vn(df["ENGINESIZE"].max(), 1))
macro("cylMin", vn(df["CYLINDERS"].min(), 0))
macro("cylMax", vn(df["CYLINDERS"].max(), 0))
macro("cylMedian", vn(df["CYLINDERS"].median(), 0))
geared = df.loc[df["NOGEAR"] == 0, "GEARS"]
macro("gearsMin", vn(geared.min(), 0))
macro("gearsMax", vn(geared.max(), 0))
macro("gearsMedian", vn(geared.median(), 0))
jb_raw = jarque_bera(y)
jb_log = jarque_bera(np.log(y))
macro("jbRaw", vn(jb_raw[0], 2))
macro("jbRawP", pval(jb_raw[1]))
macro("logSkew", vn(stats.skew(np.log(y)), 3))
macro("jbLog", vn(jb_log[0], 2))
macro("jbLogP", pval(jb_log[1]))
macro("nHybridOld", str(int(df["HYBRID_NAME"].sum())))
macro("nHybrid", str(int(df["HYBRID"].sum())))
macro("nHybridMissed", str(int((df["HYBRID"] == 1).sum() - (df["HYBRID_NAME"] == 1).sum())))
macro("nNoGear", str(int(df["NOGEAR"].sum())))
macro("nGeared", vn(int((df["NOGEAR"] == 0).sum()), 0))

# Table: P5 mapping of 16 VEHICLECLASS levels into 5 BODY groups
body_order = ["small", "midlarge", "suv", "pickup", "van"]
body_map_rows = []
for bkey in body_order:
    orig_classes = [c for c, g in BODY_GROUPS.items() if g == bkey]
    orig_sorted = sorted(orig_classes, key=lambda c: -int(vc_counts.get(c, 0)))
    orig_str = ", ".join(f"\\texttt{{{c}}} ({int(vc_counts[c])})" for c in orig_sorted)
    sub = df[df["BODY"] == bkey]
    ref_tag = " (tham chiếu)" if bkey == "small" else ""
    body_map_rows.append(
        f"{tex_text(BODY_LABELS[bkey])}{ref_tag} & {orig_str} & {len(sub)} & "
        f"{vn(sub[TARGET].mean(), 1)} $\\pm$ {vn(sub[TARGET].std(ddof=1), 1)} \\\\"
    )
(TABLES / "body-mapping.tex").write_text(
    "\\begin{tabular}{@{}l >{\\raggedright\\arraybackslash}p{8.4cm} r r@{}}\n\\toprule\n"
    "Nhóm \\texttt{BODY} & Các hạng \\texttt{VEHICLECLASS} gốc (số xe) & Tổng $n$ & \\COtwo{} TB $\\pm$ ĐLC \\\\\n\\midrule\n"
    + "\n".join(body_map_rows)
    + "\n\\bottomrule\n\\end{tabular}\n"
)

# Table: EDA group summary by Fuel Type, Body Group, Hybrid
eda_group_rows = []
for col, gtitle, mapping, order_keys in [
    ("FUELTYPE", "Nhiên liệu", FUEL_LABELS, ["X", "Z", "D", "E"]),
    ("BODY", "Nhóm thân xe", BODY_LABELS, ["small", "midlarge", "suv", "pickup", "van"]),
    ("HYBRID", "Hệ truyền động", {0: "Không hybrid", 1: "Hybrid"}, [0, 1]),
]:
    for idx_k, k in enumerate(order_keys):
        sub = df[df[col] == k]
        prefix = f"\\multirow{{{len(order_keys)}}}{{*}}{{{gtitle}}}" if idx_k == 0 else ""
        eda_group_rows.append(
            f"{prefix} & {tex_text(mapping[k])} & {len(sub)} & "
            f"{vn(sub[TARGET].mean(), 1)} & {vn(sub['ENGINESIZE'].mean(), 2)} \\\\"
        )
    if col != "HYBRID":
        eda_group_rows.append("\\midrule")
(TABLES / "eda-groups.tex").write_text(
    "\\begin{tabular}{@{}l l r r r@{}}\n\\toprule\n"
    "Yếu tố & Nhóm & $n$ & \\COtwo{} TB (g/km) & Dung tích TB (L) \\\\\n\\midrule\n"
    + "\n".join(eda_group_rows)
    + "\n\\bottomrule\n\\end{tabular}\n"
)

# Data leakage audit: CO2 / combined fuel consumption ratio per fuel type
ratio = raw[TARGET] / raw["FUELCONSUMPTION_COMB"]
ratios = ratio.groupby(raw["FUELTYPE"]).agg(["mean", lambda s: s.std() / s.mean() * 100])
ratios.columns = ["mean", "cv"]
for f in ["X", "Z", "D", "E"]:
    macro(f"ratio{f}", vn(ratios.loc[f, "mean"], 2))
macro("ratioCvMax", vn(ratios["cv"].max(), 2))

order = np.random.default_rng(SEED).permutation(len(raw))
raw_tr, raw_te = raw.iloc[order[: config.N_TRAIN]], raw.iloc[order[config.N_TRAIN :]]
leak = sm.OLS.from_formula(f"{TARGET} ~ FUELCONSUMPTION_COMB * C(FUELTYPE)", raw_tr).fit()
leak_pred = leak.predict(raw_te)
macro("leakRtwo", vn(r2_score(raw_te[TARGET], leak_pred), 5))
macro("leakRmse", vn(np.sqrt(mean_squared_error(raw_te[TARGET], leak_pred)), 2))

# Multicollinearity & VIF in M3 and M3h
macro("corrEngCo", vn(df["ENGINESIZE"].corr(df[TARGET]), 3))
macro("corrCylCo", vn(df["CYLINDERS"].corr(df[TARGET]), 3))
macro("corrEngCyl", vn(df["ENGINESIZE"].corr(df["CYLINDERS"]), 3))

X3_prep = build_model("M3").named_steps["prep"].fit(ds.X_train)
X3 = X3_prep.transform(ds.X_train)
names3 = list(X3_prep.get_feature_names_out())
X3c = sm.add_constant(X3)
vifs = {n: variance_inflation_factor(X3c, i + 1) for i, n in enumerate(names3)}
macro("vifEngine", vn(vifs["ENGINESIZE"], 2))
macro("vifCyl", vn(vifs["CYLINDERS"], 2))
macro("vifSeFactor", vn(np.sqrt(vifs["ENGINESIZE"]), 1))
other_vifs = {k: v for k, v in vifs.items() if k not in ("ENGINESIZE", "CYLINDERS", "GEARS", "NOGEAR", "TRANS_TYPE_AV")}
macro("vifMaxOther", vn(max(other_vifs.values()), 2))

# =========================================================================
# 2. Nine Linear Models + Timing + MedAE + Generalization Gap
# =========================================================================
with tempfile.TemporaryDirectory() as tmp:
    metrics = train_all(ds, out_dir=Path(tmp)).set_index("model")
    folds = pd.read_csv(Path(tmp) / "cv_folds.csv")

folds.to_csv(OUT / "cv_folds.csv", index=False)

ols_fit = {name: fit_ols(train, name, "nonrobust") for name, spec in MODELS.items() if spec.ols}

# Additional per-model diagnostics (train metrics, test MedAE, fit/predict time)
extra_metrics: dict[str, dict[str, float]] = {}
for name in MODELS:
    pipe = build_model(name)
    t0 = time.perf_counter()
    pipe.fit(ds.X_train, ds.y_train)
    fit_ms = (time.perf_counter() - t0) * 1000.0
    pred_tr = pipe.predict(ds.X_train)
    t1 = time.perf_counter()
    pred_te = pipe.predict(ds.X_test)
    pred_ms = (time.perf_counter() - t1) * 1000.0
    extra_metrics[name] = {
        "train_rmse": float(np.sqrt(mean_squared_error(ds.y_train, pred_tr))),
        "train_r2": float(r2_score(ds.y_train, pred_tr)),
        "test_medae": float(median_absolute_error(ds.y_test, pred_te)),
        "gap_r2": float(r2_score(ds.y_train, pred_tr) - r2_score(ds.y_test, pred_te)),
        "fit_ms": float(fit_ms),
        "pred_ms": float(pred_ms),
    }

# WLS check on M3h features (modeling log(e^2) = X gamma)
prep_h = build_model("M3h").named_steps["prep"].fit(ds.X_train)
Xh_tr = sm.add_constant(prep_h.transform(ds.X_train))
Xh_te = sm.add_constant(prep_h.transform(ds.X_test), has_constant="add")
yh_tr = ds.y_train.to_numpy()
yh_te = ds.y_test.to_numpy()
ols_h_raw = sm.OLS(yh_tr, Xh_tr).fit()
log_e2 = np.log(np.maximum(ols_h_raw.resid ** 2, 1e-6))
aux_var = sm.OLS(log_e2, Xh_tr).fit()
w_tr = 1.0 / np.exp(aux_var.fittedvalues)
wls_h = sm.WLS(yh_tr, Xh_tr, weights=w_tr).fit()
wls_pred_te = wls_h.predict(Xh_te)
macro("wlsRmse", vn(np.sqrt(mean_squared_error(yh_te, wls_pred_te)), 2))
macro("wlsMae", vn(mean_absolute_error(yh_te, wls_pred_te), 2))
macro("wlsMedae", vn(median_absolute_error(yh_te, wls_pred_te), 2))
macro("wlsRtwo", vn(r2_score(yh_te, wls_pred_te), 3))

# Non-linear reference ceiling: Gaussian Process (RBF + WhiteKernel) on M3h features
# Explicitly used ONLY as empirical non-linear benchmark (not counted among linear models).
gp_pipe = Pipeline([
    ("prep", build_model("M3h").named_steps["prep"]),
    ("scale", StandardScaler()),
    ("gp", GaussianProcessRegressor(
        kernel=ConstantKernel(100.0) * RBF(length_scale=5.0) + WhiteKernel(noise_level=100.0),
        random_state=SEED,
        normalize_y=True,
        n_restarts_optimizer=1,
    )),
])
gp_pipe.fit(ds.X_train, ds.y_train)
gp_pred_te = gp_pipe.predict(ds.X_test)
gp_rmse_val = float(np.sqrt(mean_squared_error(yh_te, gp_pred_te)))
macro("gpRmse", vn(gp_rmse_val, 2))
macro("gpMae", vn(mean_absolute_error(yh_te, gp_pred_te), 2))
macro("gpRtwo", vn(r2_score(yh_te, gp_pred_te), 3))
macro("gpGapEightB", vn(metrics.loc["M8b", "RMSE"] - gp_rmse_val, 2))

# Table: Main 9-model evaluation comparison
SHORT_DESC = {
    "M1": "Hồi quy đơn theo dung tích (mốc)",
    "M2": "M1 + số xi-lanh",
    "M3": "M2 + nhiên liệu, thân xe, hộp số",
    "M3h": "M3 + cờ hybrid (đã sửa nhãn)",
    "M4": "M3 + dung tích$^2$, dung tích $\\times$ (D, E85)",
    "M4log": "M4 trên $\\log(\\text{CO}_2)$ + Duan smearing",
    "M5": "Ridge trên biến M4 ($\\lambda = 3$)",
    "M6": "Bayesian NIG trên M3h (prior yếu)",
    "M8b": "M3h + dung tích $\\times$ (bán tải, van, hybrid)",
}
rows = []
for name, spec in MODELS.items():
    m = metrics.loc[name]
    f = ols_fit.get(name)
    ex = extra_metrics[name]
    rows.append(
        f"{name} & {SHORT_DESC[name]} & {int(m['params'])} & "
        f"{vn(f.rsquared_adj, 3) if f is not None else '--'} & "
        f"{vn(f.bic, 0) if f is not None else '--'} & "
        f"{vn(m['CV_RMSE_mean'], 2)} $\\pm$ {vn(m['CV_RMSE_std'], 2)} & "
        f"{vn(m['RMSE'], 2)} & {vn(m['MAE'], 2)} & {vn(ex['test_medae'], 2)} & {vn(m['R2'], 3)} \\\\"
    )
(TABLES / "models.tex").write_text(
    "\\begin{tabular}{@{}l >{\\raggedright\\arraybackslash}p{4.1cm} r r r c r r r r@{}}\n\\toprule\n"
    "Mã & Mô hình & $p$ & $R^2_{\\mathrm{adj}}$ & BIC & RMSE CV & RMSE & MAE & MedAE & $R^2$ \\\\\n"
    " & & & (train) & (train) & 10-fold & test & test & test & test \\\\\n\\midrule\n"
    + "\n".join(rows)
    + "\n\\bottomrule\n\\end{tabular}\n"
)

for name in MODELS:
    key = {
        "M1": "One", "M2": "Two", "M3": "Three", "M3h": "ThreeH", "M4": "Four",
        "M4log": "FourLog", "M5": "Five", "M6": "Six", "M8b": "EightB",
    }[name]
    macro(f"rmse{key}", vn(metrics.loc[name, "RMSE"], 2))
    macro(f"cv{key}", vn(metrics.loc[name, "CV_RMSE_mean"], 2))
    macro(f"cvSd{key}", vn(metrics.loc[name, "CV_RMSE_std"], 2))
    macro(f"rtwo{key}", vn(metrics.loc[name, "R2"], 3))
    macro(f"mae{key}", vn(metrics.loc[name, "MAE"], 2))
    macro(f"medae{key}", vn(extra_metrics[name]["test_medae"], 2))
    macro(f"trainRmse{key}", vn(extra_metrics[name]["train_rmse"], 2))
    macro(f"trainRtwo{key}", vn(extra_metrics[name]["train_r2"], 3))

macro("rmseGain", vn((1 - metrics.loc["M8b", "RMSE"] / metrics.loc["M1", "RMSE"]) * 100, 0))
macro("rmseGainThreeH", vn((1 - metrics.loc["M3h", "RMSE"] / metrics.loc["M1", "RMSE"]) * 100, 0))
macro("rmseDropHybrid", vn(metrics.loc["M3", "RMSE"] - metrics.loc["M3h", "RMSE"], 2))
macro("gapTestTrainThreeH", vn(metrics.loc["M3h", "RMSE"] - extra_metrics["M3h"]["train_rmse"], 2))
macro("gapTestTrainEightB", vn(metrics.loc["M8b", "RMSE"] - extra_metrics["M8b"]["train_rmse"], 2))
macro("gapCvTrainThreeH", vn(metrics.loc["M3h", "CV_RMSE_mean"] - extra_metrics["M3h"]["train_rmse"], 2))
macro("gapCvTrainEightB", vn(metrics.loc["M8b", "CV_RMSE_mean"] - extra_metrics["M8b"]["train_rmse"], 2))
macro("fitMsMin", vn(min(ex["fit_ms"] for ex in extra_metrics.values()), 0))
macro("fitMsMax", vn(max(ex["fit_ms"] for ex in extra_metrics.values()), 0))
macro("ratioNPFour", vn(len(train) / int(metrics.loc["M4", "params"]), 0))
macro("bicThree", vn(ols_fit["M3"].bic, 0))
macro("bicThreeH", vn(ols_fit["M3h"].bic, 0))
macro("bicEightB", vn(ols_fit["M8b"].bic, 0))

# =========================================================================
# 3. Partial F-tests & 10-fold CV Tests (Merged Model Comparison Table)
# =========================================================================
ftests = []
ftest_map: dict[tuple[str, str], tuple[float, float, float, float]] = {}
for small, large, label in [
    ("M1", "M2", "Thêm số xi-lanh"),
    ("M2", "M3", "Thêm nhiên liệu, thân xe, hộp số"),
    ("M3", "M3h", "Thêm cờ hybrid (đã sửa nhãn)"),
    ("M3", "M4", "Thêm dung tích², dung tích × (D, E85)"),
    ("M3h", "M8b", "Thêm dung tích × (bán tải, van, hybrid)"),
]:
    a = partial_f_test(train, small, large).iloc[1]
    ftests.append((small, large, label, a["df_diff"], a["df_resid"], a["F"], a["Pr(>F)"]))
    ftest_map[(small, large)] = (a["df_diff"], a["df_resid"], a["F"], a["Pr(>F)"])
    kname = f"{small}{large}".replace("1", "One").replace("2", "Two").replace("3", "Three").replace("4", "Four").replace("8", "Eight")
    macro(f"F{kname}", vn(a["F"], 2))
    macro(f"P{kname}", pval(a["Pr(>F)"]))
    macro(f"dfOne{kname}", str(int(a["df_diff"])))
    macro(f"dfTwo{kname}", str(int(a["df_resid"])))

nb: dict[str, dict[str, float]] = {}
nb_pairs = [
    ("M2", "M1", "Thêm số xi-lanh"),
    ("M3", "M2", "Thêm nhiên liệu, thân xe, hộp số"),
    ("M3h", "M3", "Thêm cờ hybrid (đã sửa nhãn)"),
    ("M4", "M3", "Thêm dung tích², dung tích × (D, E85)"),
    ("M5", "M4", "Chính quy hóa Ridge (λ = 3)"),
    ("M4log", "M4", "Biến đổi log(CO₂) + Duan"),
    ("M8b", "M3h", "Thêm dung tích × (bán tải, van, hybrid)"),
]
for a, b, _ in nb_pairs:
    fa, fb = folds[a].to_numpy(), folds[b].to_numpy()
    r = nadeau_bengio_test(fa, fb, len(train))
    w_stat, w_p = stats.wilcoxon(fa - fb)
    r["wilcoxon_p"] = float(w_p)
    nb[f"{a} vs {b}"] = r

merged_test_rows = []
for new_m, base_m, label in nb_pairs:
    ft = ftest_map.get((base_m, new_m))
    f_cell = f"{vn(ft[2], 2)} ({pval(ft[3])})" if ft is not None else "--"
    v = nb[f"{new_m} vs {base_m}"]
    merged_test_rows.append(
        f"{base_m} $\\to$ {new_m} & {tex_text(label)} & {f_cell} & "
        f"{vn(v['mean_diff'], 2)} & {pval(v['p'])} & {pval(v['wilcoxon_p'])} \\\\"
    )
(TABLES / "model-tests.tex").write_text(
    "\\begin{tabular}{@{}l >{\\raggedright\\arraybackslash}p{4.3cm} r r r r@{}}\n\\toprule\n"
    "Cặp mô hình & Thay đổi & $F$ ($p$) & $\\overline{\\Delta\\text{RMSE}}$ & $p$ (NB) & $p$ (Wilcox.) \\\\\n\\midrule\n"
    + "\n".join(merged_test_rows)
    + "\n\\bottomrule\n\\end{tabular}\n"
)
macro("nbThreeTwoP", pval(nb["M3 vs M2"]["p"]))
macro("nbThreeTwoDiff", vn(nb["M3 vs M2"]["mean_diff"], 2))
macro("nbHybridP", pval(nb["M3h vs M3"]["p"]))
macro("nbHybridDiff", vn(nb["M3h vs M3"]["mean_diff"], 2))
macro("nbInterP", pval(nb["M8b vs M3h"]["p"]))
macro("nbInterWilcoxP", pval(nb["M8b vs M3h"]["wilcoxon_p"]))
macro("nbInterDiff", vn(nb["M8b vs M3h"]["mean_diff"], 2))
macro("nbFiveFourP", pval(nb["M5 vs M4"]["p"]))
macro("nbFourLogFourP", pval(nb["M4log vs M4"]["p"]))

# =========================================================================
# 4. Ridge Regularization Path & Bootstrap Bias-Variance Decomposition
# =========================================================================
prep_m4 = build_model("M4").named_steps["prep"].fit(ds.X_train)
X4_tr_raw = prep_m4.transform(ds.X_train)
m4_names = list(prep_m4.get_feature_names_out())
scaler4 = StandardScaler().fit(X4_tr_raw)
X4_tr_s = scaler4.transform(X4_tr_raw)

lambdas = np.logspace(-2, 3.5, 36)
kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=SEED)
ridge_records = []
for lam in lambdas:
    fold_rmses = []
    for tr_i, va_i in kf.split(X4_tr_raw):
        sc = StandardScaler().fit(X4_tr_raw[tr_i])
        reg = Ridge(alpha=float(lam)).fit(sc.transform(X4_tr_raw[tr_i]), yh_tr[tr_i])
        pred_va = reg.predict(sc.transform(X4_tr_raw[va_i]))
        fold_rmses.append(np.sqrt(mean_squared_error(yh_tr[va_i], pred_va)))
    reg_full = Ridge(alpha=float(lam)).fit(X4_tr_s, yh_tr)
    rec = {
        "lambda": float(lam),
        "cv_rmse": float(np.mean(fold_rmses)),
        "cv_std": float(np.std(fold_rmses, ddof=1)),
        "coef_engine": float(reg_full.coef_[m4_names.index("ENGINESIZE")]),
        "coef_cyl": float(reg_full.coef_[m4_names.index("CYLINDERS")]),
        "coef_van": float(reg_full.coef_[m4_names.index("BODY_van")]),
        "coef_e85": float(reg_full.coef_[m4_names.index("FUELTYPE_E")]),
    }
    ridge_records.append(rec)
ridge_df = pd.DataFrame(ridge_records)
ridge_df.to_csv(OUT / "ridge_path.csv", index=False)
best_ridge = ridge_df.loc[ridge_df["cv_rmse"].idxmin()]
macro("ridgeBestLam", vn(best_ridge["lambda"], 2))
macro("ridgeBestCv", vn(best_ridge["cv_rmse"], 2))

# Bootstrap Bias^2 - Variance decomposition (B = 200 on fixed test set)
B_BV = 200
bv_rng = np.random.default_rng(SEED)
bv_models = ["M1", "M2", "M3", "M3h", "M4", "M5", "M8b"]
bv_rows = []
n_tr_len = len(ds.X_train)
boot_indices = [bv_rng.integers(0, n_tr_len, size=n_tr_len) for _ in range(B_BV)]
for mname in bv_models:
    preds = np.empty((B_BV, len(ds.X_test)))
    for b_i, idx in enumerate(boot_indices):
        pipe = build_model(mname)
        pipe.fit(ds.X_train.iloc[idx], ds.y_train.iloc[idx])
        preds[b_i] = pipe.predict(ds.X_test)
    mean_pred = preds.mean(axis=0)
    bias2_plus_noise = float(np.mean((yh_te - mean_pred) ** 2))
    variance = float(np.mean(np.var(preds, axis=0, ddof=1)))
    expected_mse = float(np.mean((yh_te[None, :] - preds) ** 2))
    bv_rows.append({
        "model": mname,
        "bias2_noise": bias2_plus_noise,
        "variance": variance,
        "expected_mse": expected_mse,
        "var_share_pct": variance / expected_mse * 100.0,
    })
bv_df = pd.DataFrame(bv_rows)
bv_df.to_csv(OUT / "bias_variance.csv", index=False)
bv_map = bv_df.set_index("model")
macro("bvBiasOne", vn(bv_map.loc["M1", "bias2_noise"], 1))
macro("bvVarOne", vn(bv_map.loc["M1", "variance"], 1))
macro("bvBiasThreeH", vn(bv_map.loc["M3h", "bias2_noise"], 1))
macro("bvVarThreeH", vn(bv_map.loc["M3h", "variance"], 1))
macro("bvVarShareThreeH", vn(bv_map.loc["M3h", "var_share_pct"], 1))
macro("bvBiasShareThreeH", vn(100.0 - bv_map.loc["M3h", "var_share_pct"], 1))
macro("bvBiasEightB", vn(bv_map.loc["M8b", "bias2_noise"], 1))
macro("bvVarEightB", vn(bv_map.loc["M8b", "variance"], 1))
macro("bvVarShareEightB", vn(bv_map.loc["M8b", "var_share_pct"], 1))
macro("bvBiasShareEightB", vn(100.0 - bv_map.loc["M8b", "var_share_pct"], 1))

# =========================================================================
# 5. M1, M3h, M8b Inference, Bootstrap, Residual Diagnostics & Coverage
# =========================================================================
m1 = fit_ols(train, "M1", "nonrobust")
macro("slopeOne", vn(m1.params["ENGINESIZE"], 2))
ci1 = m1.conf_int().loc["ENGINESIZE"]
macro("slopeOneLo", vn(ci1[0], 2))
macro("slopeOneHi", vn(ci1[1], 2))
macro("tOne", vn(m1.tvalues["ENGINESIZE"], 1))
macro("rtwoTrainOne", vn(m1.rsquared, 3))

names = ["(intercept)", *prep_h.get_feature_names_out()]
ols = sm.OLS(yh_tr, Xh_tr).fit()
hc3 = sm.OLS(yh_tr, Xh_tr).fit(cov_type="HC3")
bayes = NIGRegression(tau=1e6).fit(prep_h.transform(ds.X_train), yh_tr)
cri = bayes.credible_intervals()

# Nonparametric pairs bootstrap (B = 2000) for M3h
B = 2000
boot_rng = np.random.default_rng(SEED)
boot_coefs = np.empty((B, Xh_tr.shape[1]))
for b_idx in range(B):
    idx = boot_rng.integers(0, len(yh_tr), size=len(yh_tr))
    boot_coefs[b_idx] = np.linalg.lstsq(Xh_tr[idx], yh_tr[idx], rcond=None)[0]
boot_ci = np.quantile(boot_coefs, [0.025, 0.975], axis=0).T
boot_map = dict(zip(names, boot_ci.tolist()))

macro("bootEngLo", vn(boot_map["ENGINESIZE"][0], 1))
macro("bootEngHi", vn(boot_map["ENGINESIZE"][1], 1))
macro("bootHybridLo", vn(boot_map["HYBRID"][0], 1))
macro("bootHybridHi", vn(boot_map["HYBRID"][1], 1))
macro("bootEEightyFiveLo", vn(boot_map["FUELTYPE_E"][0], 1))
macro("bootEEightyFiveHi", vn(boot_map["FUELTYPE_E"][1], 1))

LABEL = {
    "(intercept)": "Hệ số chặn",
    "ENGINESIZE": "Dung tích động cơ (mỗi lít)",
    "CYLINDERS": "Số xi-lanh (mỗi xi-lanh)",
    "GEARS": "Số cấp hộp số (mỗi cấp)",
    "NOGEAR": "Không ghi số cấp (CVT)",
    "HYBRID": "Hệ truyền động hybrid",
    **{f"FUELTYPE_{k}": f"Nhiên liệu: {v}" for k, v in FUEL_LABELS.items()},
    **{f"BODY_{k}": f"Thân xe: {v}" for k, v in BODY_LABELS.items()},
    **{f"TRANS_TYPE_{k}": f"Hộp số: {k} ({TRANS_LABELS[k]})" for k in ["AM", "AS", "AV", "M"]},
}
coef_rows = []
for i, n in enumerate(names):
    lo, hi = ols.conf_int()[i]
    hlo, hhi = hc3.conf_int()[i]
    coef_rows.append(
        f"{tex_text(LABEL[n])} & {vn(ols.params[i], 2)} & [{vn(lo, 1)}; {vn(hi, 1)}] & "
        f"[{vn(hlo, 1)}; {vn(hhi, 1)}] & {pval(hc3.pvalues[i])} & [{vn(cri[i, 0], 1)}; {vn(cri[i, 1], 1)}] \\\\"
    )
(TABLES / "coef-m3h.tex").write_text(
    "\\begin{tabular}{@{}l r c c r c@{}}\n\\toprule\n"
    "Biến & $\\hat\\beta$ (g/km) & KTC 95\\% (OLS) & KTC 95\\% (HC3) & $p$ (HC3) & KĐTin 95\\% (M6) \\\\\n\\midrule\n"
    + "\n".join(coef_rows)
    + "\n\\bottomrule\n\\end{tabular}\n"
)

coef = dict(zip(names, ols.params))
ci_hc3 = dict(zip(names, hc3.conf_int().tolist()))
p_hc3 = dict(zip(names, hc3.pvalues.tolist()))
macro("bEngine", vn(coef["ENGINESIZE"], 2))
macro("bEngineLo", vn(ci_hc3["ENGINESIZE"][0], 1))
macro("bEngineHi", vn(ci_hc3["ENGINESIZE"][1], 1))
macro("bCyl", vn(coef["CYLINDERS"], 2))
macro("bCylLo", vn(ci_hc3["CYLINDERS"][0], 1))
macro("bCylHi", vn(ci_hc3["CYLINDERS"][1], 1))
macro("bGears", vn(coef["GEARS"], 2))
macro("bGearsAbs", vn(abs(coef["GEARS"]), 2))
macro("bNoGear", vn(coef["NOGEAR"], 1))
macro("bNoGearAbs", vn(abs(coef["NOGEAR"]), 1))
macro("bHybrid", vn(coef["HYBRID"], 1))
macro("bHybridAbs", vn(abs(coef["HYBRID"]), 1))
macro("bHybridLo", vn(ci_hc3["HYBRID"][0], 1))
macro("bHybridHi", vn(ci_hc3["HYBRID"][1], 1))
macro("bEEightyFive", vn(coef["FUELTYPE_E"], 1))
macro("bEEightyFiveAbs", vn(abs(coef["FUELTYPE_E"]), 1))
macro("bEEightyFiveLo", vn(ci_hc3["FUELTYPE_E"][0], 1))
macro("bEEightyFiveHi", vn(ci_hc3["FUELTYPE_E"][1], 1))
macro("bPremium", vn(coef["FUELTYPE_Z"], 1))
macro("bDiesel", vn(coef["FUELTYPE_D"], 1))
macro("pDiesel", pval(p_hc3["FUELTYPE_D"]))
macro("bMidlarge", vn(coef["BODY_midlarge"], 1))
macro("bSuv", vn(coef["BODY_suv"], 1))
macro("bVan", vn(coef["BODY_van"], 1))
macro("bPickup", vn(coef["BODY_pickup"], 1))
macro("bCvt", vn(coef["TRANS_TYPE_AV"], 1))
macro("bCvtAbs", vn(abs(coef["TRANS_TYPE_AV"]), 1))
macro("bCvtLo", vn(ci_hc3["TRANS_TYPE_AV"][0], 1))
macro("bCvtHi", vn(ci_hc3["TRANS_TYPE_AV"][1], 1))
macro("pCvt", pval(p_hc3["TRANS_TYPE_AV"]))
macro("seEngineOls", vn(ols.bse[names.index("ENGINESIZE")], 2))
macro("seEngineHc", vn(hc3.bse[names.index("ENGINESIZE")], 2))

# Old vs fixed hybrid flag comparison (P8 before/after)
h_old = sm.OLS.from_formula(
    f"{TARGET} ~ ENGINESIZE + CYLINDERS + GEARS + NOGEAR + HYBRID_NAME + "
    "C(FUELTYPE, Treatment('X')) + C(BODY, Treatment('small')) + C(TRANS_TYPE, Treatment('A'))",
    train,
).fit()
macro("bHybridOld", vn(h_old.params["HYBRID_NAME"], 1))
macro("bicThreeHOld", vn(h_old.bic, 0))
macro("bicDropFlag", vn(h_old.bic - ols_fit["M3h"].bic, 0))

# M8b interaction terms table (HC3)
m8_hc3 = fit_ols(train, "M8b", "HC3")
base_slope = m8_hc3.params["ENGINESIZE"]
m8_terms = [
    ("ENGINESIZE", "Dung tích động cơ (nhóm nền: xe thường)", base_slope),
    ("ENGINE_X_VAN", "Tương tác: Dung tích × Van & đặc dụng", base_slope + m8_hc3.params["ENGINE_X_VAN"]),
    ("ENGINE_X_PICKUP", "Tương tác: Dung tích × Bán tải", base_slope + m8_hc3.params["ENGINE_X_PICKUP"]),
    ("ENGINE_X_HYBRID", "Tương tác: Dung tích × Hybrid", base_slope + m8_hc3.params["ENGINE_X_HYBRID"]),
]
m8_rows = []
for term, label, eff_slope in m8_terms:
    est = m8_hc3.params[term]
    lo, hi = m8_hc3.conf_int().loc[term]
    pv = m8_hc3.pvalues[term]
    m8_rows.append(
        f"{tex_text(label)} & {vn(est, 2)} & [{vn(lo, 1)}; {vn(hi, 1)}] & {pval(pv)} & {vn(eff_slope, 1)} \\\\"
    )
(TABLES / "m8b-interactions.tex").write_text(
    "\\begin{tabular}{@{}l r c r r@{}}\n\\toprule\n"
    "Thành phần trong M8b & $\\hat\\beta$ (g/km/L) & KTC 95\\% (HC3) & $p$ (HC3) & Độ dốc hiệu dụng \\\\\n\\midrule\n"
    + "\n".join(m8_rows)
    + "\n\\bottomrule\n\\end{tabular}\n"
)

m8 = ols_fit["M8b"]
macro("slopeBase", vn(m8.params["ENGINESIZE"], 1))
macro("slopeVan", vn(m8.params["ENGINESIZE"] + m8.params["ENGINE_X_VAN"], 1))
macro("slopePickup", vn(m8.params["ENGINESIZE"] + m8.params["ENGINE_X_PICKUP"], 1))
macro("slopeHybrid", vn(m8.params["ENGINESIZE"] + m8.params["ENGINE_X_HYBRID"], 1))

# Residual diagnostics of M3h
r3 = ols_fit["M3h"]
bp = het_breuschpagan(r3.resid, r3.model.exog)
jb = jarque_bera(r3.resid)
macro("bpLM", vn(bp[0], 1))
macro("bpP", pval(bp[1]))
macro("jbResid", vn(jb[0], 1))
macro("jbResidP", pval(jb[1]))
infl = r3.get_influence().cooks_distance[0]
macro("nCook", str(int((infl > 4 / len(train)).sum())))

# Identify top 5 influential real cars on train for transparent reporting
train_meta = raw.loc[ds.X_train.index, ["MAKE", "MODEL", "ENGINESIZE", TARGET]].copy()
train_meta["resid"] = r3.resid.to_numpy()
train_meta["cooks_d"] = infl
top_cook = train_meta.sort_values("cooks_d", ascending=False).head(5)

for tag, make_q, model_q in [
    ("Bentley", "BENTLEY", "MULSANNE"),
    ("Audi", "AUDI", "R8"),
    ("Lancer", "MITSUBISHI", "LANCER EVOLUTION"),
]:
    row = top_cook[(top_cook["MAKE"] == make_q) & (top_cook["MODEL"] == model_q)].iloc[0]
    macro(f"cook{tag}Eng", vn(row["ENGINESIZE"], 1))
    macro(f"cook{tag}Co", vn(row[TARGET], 0))
    macro(f"cook{tag}Res", ("+" if row["resid"] > 0 else "") + vn(row["resid"], 1))

# Prediction interval 95% coverage on test set
pi = ols.get_prediction(Xh_te).conf_int(obs=True, alpha=0.05)
inside = (yh_te >= pi[:, 0]) & (yh_te <= pi[:, 1])
macro("piCover", vn(inside.mean() * 100, 1))
macro("piInCount", str(int(inside.sum())))
qs = np.quantile(yh_te, [0.25, 0.5, 0.75])
macro("piQThreeTest", vn(qs[2], 0))
q_masks = [
    yh_te < qs[0],
    (yh_te >= qs[0]) & (yh_te < qs[1]),
    (yh_te >= qs[1]) & (yh_te < qs[2]),
    yh_te >= qs[2],
]
for qi, mask in enumerate(q_masks, 1):
    macro(f"piCoverQ{['One', 'Two', 'Three', 'Four'][qi - 1]}", vn(inside[mask].mean() * 100, 1))
macro("piCoverTop", vn(inside[q_masks[3]].mean() * 100, 1))

# =========================================================================
# 6. E85 & Simpson's Paradox + Bayesian Prior Sensitivity
# =========================================================================
means = df.groupby("FUELTYPE")[TARGET].agg(["mean", "size"])
for f in ["X", "Z", "D", "E"]:
    macro(f"mean{f}", vn(means.loc[f, "mean"], 1))
    macro(f"n{f}", str(int(means.loc[f, "size"])))
macro("meanDiffEX", vn(means.loc["E", "mean"] - means.loc["X", "mean"], 1))

anova = stats.f_oneway(*[g[TARGET].to_numpy() for _, g in df.groupby("FUELTYPE")])
macro("anovaF", vn(anova.statistic, 2))
macro("anovaP", pval(anova.pvalue))
macro("anovaDfTwo", str(len(df) - 4))
eng_by_fuel = df.groupby("FUELTYPE")["ENGINESIZE"].mean()
macro("engE", vn(eng_by_fuel["E"], 2))
macro("engX", vn(eng_by_fuel["X"], 2))

keys = ["MAKE", "MODEL", "ENGINESIZE", "CYLINDERS", "TRANSMISSION"]
e_grp = raw[raw["FUELTYPE"] == "E"].groupby(keys)[TARGET].mean()
g_grp = raw[raw["FUELTYPE"] == "X"].groupby(keys)[TARGET].mean()
pairs = pd.concat([e_grp.rename("e85"), g_grp.rename("petrol")], axis=1, join="inner").dropna()
diff = pairs["e85"] - pairs["petrol"]
tt = stats.ttest_rel(pairs["e85"], pairs["petrol"])
ci_d = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
macro("nPairs", str(len(pairs)))
macro("pairDiff", vn(diff.mean(), 1))
macro("pairDiffAbs", vn(abs(diff.mean()), 1))
macro("pairLo", vn(ci_d[0], 1))
macro("pairHi", vn(ci_d[1], 1))
macro("pairT", vn(tt.statistic, 2))
macro("pairP", pval(tt.pvalue))
macro("pairDz", vn(diff.mean() / diff.std(ddof=1), 2))
macro("pairDfree", str(len(diff) - 1))

# Bayesian posterior probability and prior sensitivity
X_tr_h = prep_h.transform(ds.X_train)
scaler_mu = X_tr_h.mean(axis=0)
scaler_sd = X_tr_h.std(axis=0)
Z = (X_tr_h - scaler_mu) / scaler_sd
feat_h = list(prep_h.get_feature_names_out())
iE = feat_h.index("FUELTYPE_E")
iEng = feat_h.index("ENGINESIZE")
weak = NIGRegression(tau=1000).fit(Z, yh_tr)
scale_e = np.sqrt(weak.b_ / weak.a_ * weak.V_[iE + 1, iE + 1])
p_neg = stats.t.cdf(0, 2 * weak.a_, loc=weak.m_[iE + 1], scale=scale_e)
macro("probEneg", vn(p_neg, 4))

rng_sub = np.random.default_rng(SEED)
sub = rng_sub.choice(len(Z), size=40, replace=False)
taus = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100, 1000]
sens = []
for label, idx in [("full", np.arange(len(Z))), ("n40", sub)]:
    for tau in taus:
        model = NIGRegression(tau=tau).fit(Z[idx], yh_tr[idx])
        lo, hi = model.credible_intervals()[iEng + 1]
        sens.append({
            "data": label, "tau": tau,
            "mean": float(model.m_[iEng + 1] / scaler_sd[iEng]),
            "lo": float(lo / scaler_sd[iEng]),
            "hi": float(hi / scaler_sd[iEng]),
        })
sens_df = pd.DataFrame(sens)
sens_df.to_csv(OUT / "prior_sensitivity.csv", index=False)
for label, key in [("full", "Full"), ("n40", "Sub")]:
    s = sens_df[sens_df["data"] == label].set_index("tau")
    macro(f"engTight{key}", vn(s.loc[0.1, "mean"], 1))
    macro(f"engWeak{key}", vn(s.loc[1000, "mean"], 1))
    macro(f"engWeakLo{key}", vn(s.loc[1000, "lo"], 1))
    macro(f"engWeakHi{key}", vn(s.loc[1000, "hi"], 1))

# =========================================================================
# 7. Sensitivity, Robustness & Reproducibility (Split Ratio + Noise)
# =========================================================================
rob_models = ["M1", "M3", "M3h", "M5", "M8b"]
split_ratios = [0.60, 0.70, 0.80]
split_results: dict[str, dict[str, dict[str, float]]] = {}
X_all = df.drop(columns=[TARGET])
y_all = df[TARGET]

for ratio_tr in split_ratios:
    rkey = f"{int(ratio_tr*100)}/{int(round((1-ratio_tr)*100))}"
    perm = np.random.default_rng(SEED).permutation(len(df))
    n_s = int(round(len(df) * ratio_tr))
    Xtr_s, Xte_s = X_all.iloc[perm[:n_s]], X_all.iloc[perm[n_s:]]
    ytr_s, yte_s = y_all.iloc[perm[:n_s]], y_all.iloc[perm[n_s:]]
    split_results[rkey] = {}
    for mname in rob_models:
        p = build_model(mname).fit(Xtr_s, ytr_s)
        pr = p.predict(Xte_s)
        split_results[rkey][mname] = {
            "rmse": float(np.sqrt(mean_squared_error(yte_s, pr))),
            "r2": float(r2_score(yte_s, pr)),
        }

for mkey, mname in [("EightB", "M8b"), ("ThreeH", "M3h")]:
    r2_vals = [split_results[r][mname]["r2"] for r in ["60/40", "70/30", "80/20"]]
    macro(f"splitRtwo{mkey}Min", vn(min(r2_vals), 3))
    macro(f"splitRtwo{mkey}Max", vn(max(r2_vals), 3))

# Gaussian noise injection on continuous test predictors (ENGINESIZE, CYLINDERS, GEARS)
sigmas = [0.0, 0.1, 0.25, 0.5]
noise_models = ["M3h", "M5", "M8b"]
fitted_clean = {m: build_model(m).fit(ds.X_train, ds.y_train) for m in noise_models}
cont_cols = ["ENGINESIZE", "CYLINDERS", "GEARS"]
col_stds = {c: float(ds.X_train[c].std(ddof=1)) for c in cont_cols}
noise_results: dict[str, dict[str, dict[str, float]]] = {m: {} for m in noise_models}

for sig in sigmas:
    skey = f"{sig:.2f}"
    # Average over 20 Monte Carlo draws for smooth noise curve
    for m in noise_models:
        rmses_mc, r2s_mc = [], []
        n_rng = np.random.default_rng(SEED + int(sig * 1000))
        n_draws = 1 if sig == 0.0 else 20
        for _ in range(n_draws):
            Xn = ds.X_test.copy()
            if sig > 0:
                for c in cont_cols:
                    Xn[c] = np.clip(Xn[c] + n_rng.normal(0, sig * col_stds[c], size=len(Xn)), 0.0, None)
                # Recompute derived terms consistently from perturbed design inputs
                Xn["ENGINESIZE_SQ"] = Xn["ENGINESIZE"] ** 2
                Xn["ENGINE_X_DIESEL"] = Xn["ENGINESIZE"] * (Xn["FUELTYPE"] == "D")
                Xn["ENGINE_X_E85"] = Xn["ENGINESIZE"] * (Xn["FUELTYPE"] == "E")
                Xn["ENGINE_X_PICKUP"] = Xn["ENGINESIZE"] * (Xn["BODY"] == "pickup")
                Xn["ENGINE_X_VAN"] = Xn["ENGINESIZE"] * (Xn["BODY"] == "van")
                Xn["ENGINE_X_HYBRID"] = Xn["ENGINESIZE"] * Xn["HYBRID"]
            pr = fitted_clean[m].predict(Xn)
            rmses_mc.append(np.sqrt(mean_squared_error(ds.y_test, pr)))
            r2s_mc.append(r2_score(ds.y_test, pr))
        noise_results[m][skey] = {
            "rmse": float(np.mean(rmses_mc)),
            "r2": float(np.mean(r2s_mc)),
        }

macro("noiseThreeHHalf", vn(noise_results["M3h"]["0.50"]["rmse"], 2))
macro("noiseEightBHalf", vn(noise_results["M8b"]["0.50"]["rmse"], 2))
macro("noiseFiveHalf", vn(noise_results["M5"]["0.50"]["rmse"], 2))

(OUT / "robustness.json").write_text(
    json.dumps({"split": split_results, "noise": noise_results}, indent=2, ensure_ascii=False)
)

# Deterministic reproducibility sanity check
p_check1 = build_model("M3h").fit(ds.X_train, ds.y_train).predict(ds.X_test)
p_check2 = build_model("M3h").fit(ds.X_train, ds.y_train).predict(ds.X_test)
assert np.allclose(p_check1, p_check2, atol=1e-12)

# =========================================================================
# 8. Write Macros & Full JSON Record
# =========================================================================
macro("nMacros", str(len(macros) + 1))
(TABLES / "numbers.tex").write_text(
    "% Generated by analysis/analyze.py. Do not edit by hand.\n"
    + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items()))
    + "\n"
)

results = {
    "macros": macros,
    "metrics": metrics.drop(columns="description").round(4).reset_index().to_dict(orient="records"),
    "extra_metrics": extra_metrics,
    "nadeau_bengio": nb,
    "ftests": [dict(zip(["small", "large", "label", "df1", "df2", "F", "p"], t)) for t in ftests],
    "bootstrap_m3h_95ci": {
        k: [round(v[0], 3), round(v[1], 3)]
        for k, v in boot_map.items()
        if k in ("ENGINESIZE", "CYLINDERS", "HYBRID", "FUELTYPE_E", "BODY_van", "BODY_suv", "TRANS_TYPE_AV")
    },
    "top_cook": top_cook.round(4).to_dict(orient="records"),
    "env": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
    },
}
(OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))

print(f"Done: {len(macros)} macros, tables: {sorted(p.name for p in TABLES.glob('*.tex'))}")
print("Top 5 Cook's D cars on train:")
print(top_cook.to_string())
print(f"WLS test RMSE={macros['wlsRmse']}, MAE={macros['wlsMae']}, MedAE={macros['wlsMedae']}, R2={macros['wlsRtwo']}")
print(f"GP ceiling test RMSE={macros['gpRmse']}, R2={macros['gpRtwo']}")
