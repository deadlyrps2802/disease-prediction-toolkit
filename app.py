import streamlit as st
import pandas as pd
import plotly.express as px
from src.models import build_models
from src.preprocessing import load_data, preprocess_data
from src.evaluation import compare_models

st.set_page_config(page_title="Disease Prediction Toolkit", page_icon="🧠", layout="wide")
st.markdown("""<style>.main{background:linear-gradient(135deg,#06111f,#0b1728)}.block-container{max-width:1200px;padding-top:2rem}[data-testid="stMetric"]{background:#101f33;border:1px solid #243b53;padding:14px;border-radius:14px}.hero{padding:24px;border-radius:22px;background:linear-gradient(135deg,#0e749055,#312e8155);border:1px solid #22d3ee33}</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🧠 Disease Prediction Toolkit</h1><p>Interactive ML experimentation workspace — educational use only.</p></div>',unsafe_allow_html=True)
st.warning("Not a medical diagnostic tool. Results are for ML experimentation only.")
@st.cache_data
def prepare():
    df=load_data("heart.csv"); split,scaler,features=preprocess_data(df,"target"); return df,split,features
try:
    df,(X_train,X_test,y_train,y_test),features=prepare()
except Exception as exc:
    st.error(f"Could not prepare dataset: {exc}"); st.stop()
m1,m2,m3,m4=st.columns(4); m1.metric("Rows",len(df)); m2.metric("Features",len(features)); m3.metric("Train",len(y_train)); m4.metric("Test",len(y_test))
with st.sidebar:
    st.header("Experiment Controls"); run=st.button("🚀 Train & Compare",type="primary",use_container_width=True); show_data=st.checkbox("Show dataset",False); st.caption("Five reproducible classical classifiers are benchmarked on the same split.")
if run:
    with st.spinner("Training models and evaluating metrics..."):
        results=compare_models(build_models(),X_train,X_test,y_train,y_test)
    st.session_state.results=results
results=st.session_state.get("results")
if results is not None:
    display=results.sort_values("F1-Score",ascending=False).reset_index(drop=True)
    st.subheader("📊 Model Leaderboard"); st.dataframe(display.style.format({c:"{:.3f}" for c in display.columns if c!="Model"}),use_container_width=True)
    best=display.iloc[0]; a,b,c=st.columns(3); a.metric("🏆 Best model",best["Model"]); b.metric("Best F1",f'{best["F1-Score"]:.3f}'); c.metric("ROC-AUC",f'{best["ROC-AUC"]:.3f}' if pd.notna(best["ROC-AUC"]) else "N/A")
    left,right=st.columns(2)
    with left:
        fig=px.bar(display,x="Model",y="F1-Score",color="F1-Score",color_continuous_scale="Tealgrn",title="F1-Score comparison"); fig.update_layout(template="plotly_dark",xaxis_title=None); st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=px.bar(display,x="Model",y="Accuracy",color="Accuracy",color_continuous_scale="Blues",title="Accuracy comparison"); fig.update_layout(template="plotly_dark",xaxis_title=None); st.plotly_chart(fig,use_container_width=True)
else: st.info("Click **Train & Compare** to run the complete model benchmark.")
if show_data:
    st.subheader("🔎 Dataset preview"); st.dataframe(df.head(50),use_container_width=True); st.caption(f"Columns: {', '.join(df.columns)}")
st.divider(); st.caption("Pipeline: Load → Encode → Impute → Scale → Stratified Split → Train → Evaluate → Compare → Visualize")
