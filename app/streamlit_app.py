"""CO2 prediction app for the Lab 01 models.

    streamlit run app/streamlit_app.py

Requires `python -m linreg.train` to have written models/ first.
"""

import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from linreg import config
from linreg.data import load_frame
from linreg.features import FUEL_LABELS, add_interactions

LABELS = {
    "ENGINESIZE": "Dung tích động cơ (L)",
    "CYLINDERS": "Số xi-lanh",
    "FUELTYPE": "Nhiên liệu",
    "BODY": "Nhóm thân xe",
    "TRANS_TYPE": "Loại hộp số",
    "HYBRID": "Hybrid",
}

st.set_page_config(page_title="Dự đoán CO₂ — Lab 01", layout="wide")

meta_path = config.MODELS_DIR / "meta.json"
if not meta_path.exists():
    st.error("Chưa có mô hình. Chạy `python -m linreg.train` trước.")
    st.stop()

meta = json.loads(meta_path.read_text())
metrics = pd.read_csv(config.MODELS_DIR / "metrics.csv")
target = meta["target"]

model_name = st.sidebar.selectbox(
    "Mô hình", metrics["model"], index=int(metrics.index[metrics["model"] == meta["best_model"]][0])
)
model = joblib.load(config.MODELS_DIR / f"{model_name}.joblib")

st.sidebar.header("Thông số xe")
inputs = {}
for col, (lo, hi, median) in meta["numeric"].items():
    step = 1.0 if col == "CYLINDERS" else 0.1
    inputs[col] = st.sidebar.slider(LABELS[col], lo, hi, median, step=step)
for col, options in meta["categorical"].items():
    fmt = (lambda v: f"{v} — {FUEL_LABELS[v]}") if col == "FUELTYPE" else str
    inputs[col] = st.sidebar.selectbox(LABELS[col], options, format_func=fmt)
for col in meta["binary"]:
    inputs[col] = int(st.sidebar.checkbox(LABELS[col]))
row = add_interactions(pd.DataFrame([inputs]))

st.title("Dự đoán lượng phát thải CO₂ (g/km)")
left, right = st.columns([1, 2])
left.metric("CO₂ dự đoán", f"{model.predict(row)[0]:,.1f} g/km")
selected = metrics.set_index("model").loc[model_name]
left.caption(selected["description"])
left.caption(
    f"Tập test: R² = {selected['R2']:.3f}, RMSE = {selected['RMSE']:.1f} g/km. "
    f"CV {config.CV_FOLDS}-fold: RMSE = {selected['CV_RMSE_mean']:.1f} ± {selected['CV_RMSE_std']:.1f} g/km."
)

lo, hi, _ = meta["numeric"]["ENGINESIZE"]
sweep = pd.concat([pd.DataFrame([inputs])] * 50, ignore_index=True)
sweep["ENGINESIZE"] = [lo + i * (hi - lo) / 49 for i in range(50)]
sweep = add_interactions(sweep)
sweep[target] = model.predict(sweep)
right.plotly_chart(
    px.line(sweep, x="ENGINESIZE", y=target, labels={"ENGINESIZE": LABELS["ENGINESIZE"], target: "CO₂ (g/km)"},
            title="CO₂ dự đoán theo dung tích động cơ (các thông số khác giữ nguyên)"),
    width="stretch",
)

tab_metrics, tab_coef, tab_data = st.tabs(["So sánh mô hình", "Hệ số", "Dữ liệu"])
tab_metrics.dataframe(metrics.style.format(precision=3), width="stretch")
tab_metrics.plotly_chart(
    px.bar(metrics, x="model", y="CV_RMSE_mean", error_y="CV_RMSE_std", title="RMSE cross-validation (g/km)"),
    width="stretch",
)

coef = pd.read_csv(config.MODELS_DIR / f"{model_name}_coefficients.csv")
tab_coef.caption("Mô hình OLS: g/km trên một đơn vị. Ridge và Bayesian: g/km trên một độ lệch chuẩn của biến số.")
tab_coef.plotly_chart(
    px.bar(coef[coef["feature"] != "(intercept)"], x="coefficient", y="feature", orientation="h"),
    width="stretch",
)

df = load_frame()
tab_data.plotly_chart(
    px.scatter(df, x="ENGINESIZE", y=target, color="BODY", opacity=0.6,
               labels={"ENGINESIZE": LABELS["ENGINESIZE"], target: "CO₂ (g/km)", "BODY": LABELS["BODY"]}),
    width="stretch",
)
