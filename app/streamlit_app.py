"""Prediction app for the Lab 01 models.

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

st.set_page_config(page_title="Linear Regression — Lab 01", layout="wide")

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

st.sidebar.header("Giá trị đặc trưng")
inputs = {}
for col, (lo, hi, median) in meta["numeric_features"].items():
    inputs[col] = st.sidebar.slider(col, lo, hi, median) if lo < hi else lo
for col, options in meta["categorical_features"].items():
    inputs[col] = st.sidebar.selectbox(col, options)
row = pd.DataFrame([inputs])

st.title(f"Dự đoán {target}")
left, right = st.columns([1, 2])
left.metric(f"{target} dự đoán", f"{model.predict(row)[0]:,.3f}")
selected = metrics.set_index("model").loc[model_name]
left.caption(f"Tập test — R² = {selected['R2']:.3f}, RMSE = {selected['RMSE']:,.3f}, MAE = {selected['MAE']:,.3f}")

if meta["numeric_features"]:
    feature = right.selectbox("Ảnh hưởng của một đặc trưng (giữ nguyên các đặc trưng khác)", list(meta["numeric_features"]))
    lo, hi, _ = meta["numeric_features"][feature]
    sweep = pd.concat([row] * 50, ignore_index=True)
    sweep[feature] = pd.Series(range(50)) * (hi - lo) / 49 + lo
    sweep[target] = model.predict(sweep)
    right.plotly_chart(px.line(sweep, x=feature, y=target), use_container_width=True)

tab_metrics, tab_coef, tab_data = st.tabs(["So sánh mô hình", "Hệ số", "Dữ liệu"])
tab_metrics.dataframe(metrics.style.format(precision=4), use_container_width=True)
tab_metrics.plotly_chart(px.bar(metrics, x="model", y="R2", title="R² trên tập test"), use_container_width=True)

coef = pd.read_csv(config.MODELS_DIR / f"{model_name}_coefficients.csv").head(20)
tab_coef.caption("Hệ số trên thang đã chuẩn hoá (StandardScaler), 20 hệ số có độ lớn lớn nhất.")
tab_coef.plotly_chart(
    px.bar(coef, x="coefficient", y="feature", orientation="h").update_yaxes(autorange="reversed"),
    use_container_width=True,
)

try:
    df = load_frame()
    x_col = tab_data.selectbox("Trục x", list(meta["numeric_features"]))
    tab_data.plotly_chart(px.scatter(df, x=x_col, y=target, opacity=0.4, trendline="ols"), use_container_width=True)
except FileNotFoundError:
    tab_data.info("Không tìm thấy file dữ liệu gốc để vẽ biểu đồ.")
