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
from linreg.features import BODY_LABELS, FUEL_LABELS, TRANS_LABELS, add_derived

LABELS = {
    "ENGINESIZE": "Dung tích động cơ (L)",
    "CYLINDERS": "Số xi-lanh",
    "GEARS": "Số cấp hộp số (0 = vô cấp)",
    "BODY": "Nhóm thân xe",
    "FUELTYPE": "Nhiên liệu",
    "TRANS_TYPE": "Kiểu hộp số",
    "HYBRID": "Hybrid",
}
OPTION_LABELS = {"BODY": BODY_LABELS, "FUELTYPE": FUEL_LABELS, "TRANS_TYPE": TRANS_LABELS}
STEPS = {"ENGINESIZE": 0.1, "CYLINDERS": 1.0, "GEARS": 1.0}

st.set_page_config(page_title="Dự đoán CO₂ — Lab 01", layout="wide")

meta_path = config.MODELS_DIR / "meta.json"
if not meta_path.exists():
    st.error("Chưa có mô hình. Chạy `python -m linreg.train` trước.")
    st.stop()

meta = json.loads(meta_path.read_text())
metrics = pd.read_csv(config.MODELS_DIR / "metrics.csv")
target = meta["target"]

model_name = st.sidebar.selectbox(
    "Mô hình", metrics["model"], index=int(metrics.index[metrics["model"] == meta["app_model"]][0]),
    format_func=lambda m: f"{m}: {metrics.set_index('model').loc[m, 'description']}",
)
model = joblib.load(config.MODELS_DIR / f"{model_name}.joblib")

st.sidebar.header("Thông số thiết kế")
inputs = {}
for col, options in meta["categorical"].items():
    inputs[col] = st.sidebar.selectbox(
        LABELS[col], options, format_func=lambda v, c=col: f"{v} — {OPTION_LABELS[c][v]}"
    )
for col, (lo, hi, median) in meta["numeric"].items():
    inputs[col] = st.sidebar.slider(LABELS[col], lo, hi, median, step=STEPS[col])
for col in meta["binary"]:
    inputs[col] = int(st.sidebar.checkbox(LABELS[col]))
row = add_derived(pd.DataFrame([inputs]))

st.title("Dự đoán lượng phát thải CO₂ (g/km)")
left, right = st.columns([1, 2])
prediction = model.predict(row)[0]
half_width = 1.96 * meta["residual_sd"]
left.metric("CO₂ dự đoán", f"{prediction:,.1f} g/km")
left.caption(
    f"Khoảng dự đoán 95% xấp xỉ (±1,96·σ̂ của {meta['app_model']}): "
    f"{prediction - half_width:,.0f} – {prediction + half_width:,.0f} g/km. "
    "Khoảng này hẹp hơn thực tế ở nhóm xe thải nhiều."
)
selected = metrics.set_index("model").loc[model_name]
left.caption(
    f"Tập test: R² = {selected['R2']:.3f}, RMSE = {selected['RMSE']:.1f} g/km. "
    f"CV {config.CV_FOLDS}-fold: RMSE = {selected['CV_RMSE_mean']:.1f} ± {selected['CV_RMSE_std']:.1f} g/km."
)

lo, hi, _ = meta["numeric"]["ENGINESIZE"]
sweep = pd.concat([pd.DataFrame([inputs])] * 50, ignore_index=True)
sweep["ENGINESIZE"] = [lo + i * (hi - lo) / 49 for i in range(50)]
sweep = add_derived(sweep)
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
tab_coef.caption("Hệ số theo đơn vị gốc: g/km trên một đơn vị của biến (M4log: thang log).")
tab_coef.plotly_chart(
    px.bar(coef[coef["feature"] != "(intercept)"], x="coefficient", y="feature", orientation="h"),
    width="stretch",
)

df = load_frame()
tab_data.plotly_chart(
    px.scatter(df.assign(BODY=df["BODY"].map(BODY_LABELS)), x="ENGINESIZE", y=target, color="BODY", opacity=0.6,
               labels={"ENGINESIZE": LABELS["ENGINESIZE"], target: "CO₂ (g/km)", "BODY": LABELS["BODY"]}),
    width="stretch",
)
