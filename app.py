import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.models import build_models
from src.preprocessing import load_data, preprocess_data
from src.evaluation import compare_models

st.set_page_config(page_title="Disease Prediction Toolkit", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp{background:radial-gradient(circle at 10% 5%,#064e3b44,transparent 30%),radial-gradient(circle at 90% 10%,#0f766e33,transparent 28%),#04100d;color:#e7fff4}.block-container{max-width:1280px;padding-top:2rem}.hero{padding:34px;border-radius:28px;background:linear-gradient(135deg,#0f513244,#10b98118 50%,#22d3ee14);border:1px solid #34d39933;box-shadow:0 25px 90px #0006}.hero h1{font-size:clamp(38px,5vw,68px);letter-spacing:-.05em;margin:0;background:linear-gradient(100deg,#fff,#6ee7b7,#67e8f9);color:transparent;background-clip:text}.hero p{color:#9cc5b3;font-size:17px;max-width:760px;line-height:1.7}.metric-card{background:#071b15;border:1px solid #34d39922;border-radius:18px;padding:16px}.stButton>button{border-radius:12px;border:1px solid #34d39955;background:linear-gradient(100deg,#059669,#0d9488);color:white;font-weight:800}.section{border:1px solid #34d39918;border-radius:22px;padding:18px;background:#06161199}.stDataFrame{border-radius:16px}.notice{padding:12px 16px;border-radius:14px;background:#451a031f;border:1px solid #f59e0b33;color:#fcd34d}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div style="color:#6ee7b7;font-weight:900;letter-spacing:.18em;font-size:11px">AI / ML EXPERIMENTATION LAB</div><h1>🧬 Disease Prediction Toolkit</h1><p>Train, compare, inspect and visualize classical machine-learning models through an interactive experimentation workspace.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="notice">⚠️ <b>Educational use only.</b> This application is a machine-learning experiment and is not a medical diagnostic tool.</div>', unsafe_allow_html=True)

@st.cache_data
def prepare():
    df=load_data("heart.csv")
    split,scaler,features=preprocess_data(df,"target")
    return df,split,features

try:
    df,(X_train,X_test,y_train,y_test),features=prepare()
except Exception as exc:
    st.error(f"Could not prepare dataset: {exc}")
    st.stop()

c1,c2,c3,c4=st.columns(4)
for col,label,value in [(c1,"DATASET ROWS",len(df)),(c2,"FEATURES",len(features)),(c3,"TRAIN SAMPLES",len(y_train)),(c4,"TEST SAMPLES",len(y_test))]:
    with col:
        st.markdown(f'<div class="metric-card"><div style="color:#6b8f80;font-size:11px">{label}</div><div style="font-size:30px;font-weight:900">{value}</div></div>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧪 Experiment Lab")
    run=st.button("🚀 Run Full Benchmark",use_container_width=True)
    show_data=st.checkbox("Explore dataset",False)
    show_details=st.checkbox("Show model details",True)
    st.caption("Models use the same reproducible preprocessing and stratified split.")

if run:
    with st.spinner("Training models and calculating evaluation metrics..."):
        st.session_state.results=compare_models(build_models(),X_train,X_test,y_train,y_test)
results=st.session_state.get("results")

if results is None:
    st.markdown("## 🚀 Ready for an experiment")
    st.info("Use **Run Full Benchmark** in the sidebar to train and compare the available classifiers.")
else:
    display=results.sort_values("F1-Score",ascending=False).reset_index(drop=True)
    best=display.iloc[0]
    st.markdown("## 🏆 Model Intelligence")
    a,b,c,d=st.columns(4)
    a.metric("Best model",best["Model"]); b.metric("F1 score",f'{best["F1-Score"]:.3f}'); c.metric("Accuracy",f'{best["Accuracy"]:.3f}'); d.metric("ROC-AUC",f'{best["ROC-AUC"]:.3f}' if pd.notna(best["ROC-AUC"]) else "N/A")
    left,right=st.columns(2)
    with left:
        fig=px.bar(display,x="Model",y="F1-Score",color="F1-Score",color_continuous_scale=["#064e3b","#10b981","#67e8f9"],title="F1-score leaderboard")
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis_title=None,margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=go.Figure()
        fig.add_trace(go.Scatterpolar(r=display["Accuracy"].tolist(),theta=display["Model"].tolist(),fill="toself",line_color="#34d399",fillcolor="rgba(52,211,153,.15)"))
        fig.update_layout(title="Accuracy profile",template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",polar=dict(radialaxis=dict(visible=True,range=[0,1])),margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("## 📋 Full benchmark")
    st.dataframe(display.style.format({c:"{:.3f}" for c in display.columns if c!="Model"}),use_container_width=True,hide_index=True)
    if show_details:
        st.markdown("## 🔬 Pipeline")
        st.code("Load → Encode → Impute → Scale → Stratified Split → Train → Evaluate → Rank → Visualize",language="text")

if show_data:
    st.markdown("## 🔎 Dataset Explorer")
    st.dataframe(df.head(100),use_container_width=True,hide_index=True)
    st.caption(f"{len(df):,} rows · {len(df.columns)} columns · Features: {', '.join(features)}")

st.divider(); st.caption("Disease Prediction Toolkit · Python · Streamlit · scikit-learn · Plotly · Educational ML experimentation")
