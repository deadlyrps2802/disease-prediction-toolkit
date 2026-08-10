import streamlit as st
from src.models import build_models
from src.preprocessing import load_data, preprocess_data
from src.evaluation import compare_models

st.set_page_config(page_title="Disease Prediction Toolkit", page_icon="🧠", layout="wide")
st.title("🧠 Disease Prediction Toolkit")
st.caption("Educational ML model comparison — not a medical diagnostic tool.")

@st.cache_data
def prepare():
    df = load_data("heart.csv")
    (X_train, X_test, y_train, y_test), scaler, features = preprocess_data(df, "target")
    return X_train, X_test, y_train, y_test, features

X_train, X_test, y_train, y_test, features = prepare()
st.write(f"Dataset: **{len(y_train) + len(y_test)} rows** • Features: **{len(features)}**")

if st.button("Train & Compare Models", type="primary"):
    with st.spinner("Training models..."):
        results = compare_models(build_models(), X_train, X_test, y_train, y_test)
    st.subheader("Model comparison")
    st.dataframe(results.style.format({c: "{:.3f}" for c in results.columns if c != "Model"}), use_container_width=True)
    best = results.sort_values("F1-Score", ascending=False).iloc[0]
    st.success(f"Best F1-Score: {best['Model']} ({best['F1-Score']:.3f})")

with st.expander("Dataset preview"):
    st.dataframe(load_data("heart.csv").head(20), use_container_width=True)
