import streamlit as st
import pandas as pd
import plotly.express as px
from src.models import build_models
from src.preprocessing import load_data, preprocess_data
from src.evaluation import compare_models

st.set_page_config(page_title="Disease Prediction Toolkit", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.main {background: linear-gradient(135deg,#06111f,#0b1728)}
.block-container {max-width:1200px;padding-top:2rem}
[data-testid="stMetric"] {background:#101f33;border:1px solid #243b53;padding:14px;border-radius:14px}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Disease Prediction Toolkit")
st.caption("Interactive classical-ML experimentation workspace — educational use only, not a medical diagnostic tool.")

@st.cache_data
def prepare():
    df = load_data("heart.csv")
    split, scaler, features = preprocess_data(df, "target")
    return df, split, scaler, features

try:
    df, split, scaler, features = prepare()
    (X_train, X_test, y_train, y_test) = split
except Exception as exc:
    st.error(f"Could not prepare the dataset: {exc}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", len(df))
m2.metric("Features", len(features))
m3.metric("Train", len(y_train))
m4.metric("Test", len(y_test))

with st.sidebar:
    st.header("Controls")
    run = st.button("🚀 Train & Compare", type="primary", use_container_width=True)
    show_data = st.checkbox("Show dataset", value=False)
    st.info("The toolkit compares multiple classical classifiers using a reproducible preprocessing split.")

if run:
    with st.spinner("Training and evaluating models..."):
        results = compare_models(build_models(), X_train, X_test, y_train, y_test)
    st.session_state["results"] = results

results = st.session_state.get("results")
if results is not None:
    st.subheader("📊 Model Leaderboard")
    display = results.sort_values("F1-Score", ascending=False).reset_index(drop=True)
    st.dataframe(display.style.format({c:"{:.3f}" for c in display.columns if c != "Model"}), use_container_width=True)
    best = display.iloc[0]
    a,b,c = st.columns(3)
    a.metric("Best model", best["Model"])
    b.metric("Best F1", f'{best["F1-Score"]:.3f}')
    c.metric("ROC-AUC", f'{best["ROC-AUC"]:.3f}' if pd.notna(best["ROC-AUC"]) else "N/A")
    chart = px.bar(display, x="Model", y="F1-Score", color="F1-Score", color_continuous_scale="Tealgrn", title="F1-Score comparison")
    chart.update_layout(template="plotly_dark", xaxis_title=None)
    st.plotly_chart(chart, use_container_width=True)
else:
    st.info("Click **Train & Compare** to run the complete model benchmark.")

if show_data:
    st.subheader("🔎 Dataset preview")
    st.dataframe(df.head(50), use_container_width=True)
    st.caption(f"Columns: {', '.join(df.columns)}")

st.divider()
st.caption("Educational ML project • Results depend on the dataset and evaluation split.")
