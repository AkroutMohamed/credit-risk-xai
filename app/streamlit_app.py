"""End-to-end interactive explainable credit scoring dashboard."""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve
from src.data import load_uci, normalize, audit, TARGET

from src.ml import MODELS, split_data, fit_models, compare, metrics
from src.explain import global_importance, shap_local, lime_local, human_summary

st.set_page_config(page_title="Credit Risk | Explainable AI", page_icon="🏦", layout="wide")
st.title("🏦 Explainable Credit Risk Lab")
st.caption("Academic decision-support prototype • Default prediction is not a final credit decision")

@st.cache_data(show_spinner="Downloading UCI data…")
def fetch():
    return load_uci()

with st.sidebar:
    st.header("Dataset & settings")
    source = st.radio("Source", ["UCI Taiwan Credit Default", "Upload CSV"])
    uploaded = st.file_uploader("CSV", type=["csv"]) if source == "Upload CSV" else None
    if source == "Upload CSV" and uploaded is not None:
        raw = pd.read_csv(uploaded)
        target_col = st.selectbox("Target", list(raw.columns), index=list(raw.columns).index("default") if "default" in raw else 0)
        try:
            df = normalize(raw, target_col)
        except ValueError as e:
            st.error(str(e)); st.stop()
    elif source == "Upload CSV":
        st.info("Upload a binary-labeled CSV to begin."); st.stop()
    else:
        try:
            df = fetch()
        except Exception as e:
            st.error(f"Download failed: {e}. Check network access or upload your own CSV."); st.stop()
    features = st.multiselect("Predictor columns", [c for c in df if c != TARGET],
                               default=[c for c in df if c != TARGET])
    test_size = st.slider("Test fraction", 0.15, 0.4, 0.2, step=0.05)
    seed = st.number_input("Random seed", 0, 99999, 42)
    chosen = st.multiselect("Models", MODELS, default=MODELS)
    tune = st.checkbox("Hyperparameter search (slower)", value=False)
    iterations = st.slider("Search iterations/model", 2, 15, 6) if tune else 6
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    api_url = st.text_input("FastAPI URL (optional)", value="http://127.0.0.1:8000")
    train = st.button("Train / retrain models", type="primary", use_container_width=True)

config = (tuple(features), test_size, int(seed), tuple(chosen), tune, iterations, source,
          (uploaded.getvalue().__hash__() if uploaded else "uci"))
if "config" in st.session_state and st.session_state.config != config:
    for key in ["trained_models", "train_split"]:
        st.session_state.pop(key, None)

page1, page2, page3, page4, page5, page6 = st.tabs([
    "1 · Data", "2 · Preparation", "3 · Models", "4 · Evaluation", "5 · Explainability", "6 · API"])
with page1:
    a,b,c = st.columns(3)
    a.metric("Observations", f"{len(df):,}")
    b.metric("Predictor columns", len(df.columns)-1)
    c.metric("Default class (1)", f"{100*df[TARGET].mean():.1f}%")
    st.dataframe(df.head(60), use_container_width=True)
    st.subheader("Data quality")
    st.dataframe(audit(df), use_container_width=True)
    st.plotly_chart(px.histogram(df, x=TARGET, title="Class distribution"), use_container_width=True)
    numeric = [c for c in df if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]
    if numeric:
        col = st.selectbox("Explore variable", numeric)
        st.plotly_chart(px.histogram(df, x=col, color=df[TARGET].astype(str), barmode="overlay", nbins=30,
                                       labels={"color": "Default"}), use_container_width=True)
        st.plotly_chart(px.bar(df.groupby(col, dropna=False)[TARGET].mean().reset_index().head(50),
                               x=col, y=TARGET, title="Default rate by variable (first 50 values)"), use_container_width=True)
        corrcols = [c for c in numeric if df[c].nunique() > 1][:25]
        if corrcols:
            st.plotly_chart(px.imshow(df[corrcols+[TARGET]].corr(numeric_only=True),
                                      title="Correlations (up to 25 predictors)", aspect="auto"), use_container_width=True)

with page2:
    st.write("The training-only pipeline imputes missing values, standardizes numeric fields and one-hot encodes categoricals. The split is stratified; the test set is held out from fitting and tuning.")
    st.write(f"**Test fraction:** {test_size:.0%} | **Seed:** {seed} | **Selected predictors:** {len(features)}")
    st.write("Check and exclude identifiers, post-event fields, proxies, and features unavailable at decision time.")

if train:
    if not features or not chosen:
        st.error("Select at least one feature and model.")
    else:
        try:
            with st.spinner("Fitting models; XGBoost and cross-validation can take several minutes…"):
                split = split_data(df, features, test_size=test_size, seed=int(seed))
                models = fit_models(split, chosen, tune=tune, n_iter=iterations, seed=int(seed))
                st.session_state.trained_models = models
                st.session_state.train_split = split
                st.session_state.config = config
            st.success("Models trained. Visit the evaluation and explainability tabs.")
        except Exception as e:
            st.exception(e)

ready = "trained_models" in st.session_state and st.session_state.get("config") == config
with page3:
    if not ready: st.info("Click Train / retrain models in the sidebar.")
    else:
        models = st.session_state.trained_models
        split = st.session_state.train_split
        st.dataframe(compare(models, split, threshold).style.format(precision=3), use_container_width=True)
        st.caption("Ranking uses held-out ROC-AUC; model selection should ideally use validation/CV, not repeated inspection of the test set.")

with page4:
    if not ready: st.info("Train models first.")
    else:
        models, split = st.session_state.trained_models, st.session_state.train_split
        ranking = compare(models, split, threshold)
        active = st.selectbox("Model to inspect", list(ranking.Model), key="evaluate_model")
        score = models[active].predict_proba(split.x_test)[:,1]
        st.dataframe(pd.DataFrame([metrics(split.y_test, score, threshold)]).style.format(precision=3), use_container_width=True)
        fig = go.Figure()
        for name, model in models.items():
            fpr, tpr, _ = roc_curve(split.y_test, model.predict_proba(split.x_test)[:,1])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Chance",line=dict(dash="dash")))
        fig.update_layout(title="Held-out ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, use_container_width=True)
        precision, recall, _ = precision_recall_curve(split.y_test, score)
        st.plotly_chart(px.area(x=recall, y=precision, title="Precision–recall curve", labels={"x":"Recall", "y":"Precision"}), use_container_width=True)
        m = metrics(split.y_test, score, threshold)
        st.plotly_chart(px.imshow([[m["TN"],m["FP"]],[m["FN"],m["TP"]]], text_auto=True,
                                  x=["Predicted 0","Predicted 1"], y=["Actual 0","Actual 1"], title="Confusion matrix"), use_container_width=True)
        st.write("**Threshold trade-off:** Lower cutoffs generally catch more defaults but generate more false alerts; the correct cutoff depends on financial costs and calibration.")

with page5:
    if not ready: st.info("Train models first.")
    else:
        models, split = st.session_state.trained_models, st.session_state.train_split
        model_name = st.selectbox("Explain model", list(models), key="explain_model")
        model = models[model_name]
        st.subheader("Global: permutation importance")
        if st.button("Compute global importance"):
            with st.spinner("Computing permutation importance..."):
                imp = global_importance(model, split.x_test, split.y_test)
                st.session_state["importance"] = (model_name, imp)
        cached = st.session_state.get("importance")
        if cached and cached[0] == model_name:
            st.plotly_chart(px.bar(cached[1].head(20), x="Importance", y="Feature", orientation="h", title="Held-out permutation importance"), use_container_width=True)
        st.subheader("Local: customer decision")
        ix = st.number_input("Test row index", min_value=0, max_value=len(split.x_test)-1, value=0)
        row = split.x_test.iloc[[ix]].copy()
        st.dataframe(row.T.rename(columns={row.index[0]: "Value"}), use_container_width=True)
        p = float(model.predict_proba(row)[0,1])
        st.metric("Estimated probability of default", f"{p:.1%}")
        st.write("**Threshold classification:**", "Flag for analyst review" if p >= threshold else "No default flag")
        if st.button("Ask FastAPI for prediction"):
            try:
                response = requests.post(api_url.rstrip("/")+"/predict", json={"features":row.iloc[0].where(pd.notna(row.iloc[0]),None).to_dict(),"threshold":threshold}, timeout=20)
                response.raise_for_status()
                st.json(response.json())
            except requests.RequestException as exc:
                st.error(f"API unavailable: {exc}. Start the API and export its trained bundle first.")
        if st.button("Explain selected customer (SHAP + LIME)"):
            with st.spinner("Computing SHAP and LIME; this may take time..."):
                try:
                    shap_result = shap_local(model, split.x_train, row)
                    lime_result = lime_local(model, split.x_train.sample(n=min(500,len(split.x_train)),random_state=42), row)
                    st.session_state["local_expl"] = (model_name, int(ix), shap_result, lime_result)
                except Exception as e:
                    st.exception(e)
        explained = st.session_state.get("local_expl")
        if explained and explained[:2] == (model_name, int(ix)):
            sr, lr = explained[2:]
            shap_frame = pd.DataFrame({"Feature": sr["features"], "SHAP": sr["contributions"]}).sort_values("SHAP")
            st.plotly_chart(px.bar(shap_frame.tail(15), x="SHAP", y="Feature", orientation="h", title="SHAP local contributions (probability scale)"), use_container_width=True)
            import shap, matplotlib.pyplot as plt
            from src.explain import transformed
            values, names = transformed(model,row)
            explanation = shap.Explanation(values=np.array(sr["contributions"]),base_values=sr["base_value"],data=values[0],feature_names=names)
            shap.plots.waterfall(explanation, max_display=12, show=False)
            st.pyplot(plt.gcf()); plt.close("all")
            st.write("SHAP base probability:", round(sr["base_value"],4), "→ prediction:", round(sr["prediction"],4))
            st.plotly_chart(px.bar(pd.DataFrame(lr["rules"]), x="weight",y="condition",orientation="h", title="LIME local explanation"), use_container_width=True)
            st.write(human_summary(sr))
            st.caption("Compare direction and strongest variables across SHAP and LIME. They need not agree exactly; neither establishes causality.")
        st.subheader("Global SHAP summary and dependence")
        if st.button("Compute global SHAP sample"):
            import shap
            sample = split.x_test.sample(n=min(15,len(split.x_test)),random_state=42)
            try:
                from src.explain import transformed
                bg, names = transformed(model,split.x_train.sample(n=min(40,len(split.x_train)),random_state=42))
                vals, _ = transformed(model, sample)
                clf = model.named_steps["model"]
                explainer = shap.Explainer(lambda a: clf.predict_proba(np.asarray(a))[:,1],bg,algorithm="permutation")
                sv = explainer(vals,max_evals=max(2*len(names)+1,51))
                fig = shap.plots.beeswarm(shap.Explanation(values=sv.values,base_values=sv.base_values,data=vals,feature_names=names),show=False,max_display=15)
                import matplotlib.pyplot as plt
                st.pyplot(plt.gcf()); plt.close("all")
                feature = st.session_state.get("shap_feature", names[0])
                st.session_state["global_shap"] = (model_name, vals, sv.values, names)
            except Exception as e: st.exception(e)
        gs = st.session_state.get("global_shap")
        if gs and gs[0] == model_name:
            feature_name = st.selectbox("SHAP dependence feature", gs[3])
            j = gs[3].index(feature_name)
            st.plotly_chart(px.scatter(x=gs[1][:,j], y=gs[2][:,j], labels={"x":feature_name,"y":"SHAP effect"}, title="SHAP dependence (sample)"), use_container_width=True)
        st.caption("Sensitive attributes and proxies must be reviewed for unfair effects. Explanations support analyst review, not automatic approval/denial.")

with page6:
    st.write("FastAPI service exposes `GET /health`, `POST /predict`, `POST /explain/shap` and `POST /explain/lime`. View interactive documentation at `/docs` on your API server.")
    if ready:
        name = st.selectbox("Model to export", list(st.session_state.trained_models), key="export_model")
        buffer = io.BytesIO()
        joblib.dump({"model":st.session_state.trained_models[name],
                     "background":st.session_state.train_split.x_train.sample(n=min(500,len(st.session_state.train_split.x_train)),random_state=42)}, buffer)
        st.download_button("Download trained API bundle", buffer.getvalue(), "bundle.joblib", mime="application/octet-stream")
        st.write("Place the downloaded bundle at `models/bundle.joblib` in your trusted local project directory, then start FastAPI. Never load unknown `.joblib` files.")
    st.code('uvicorn api.main:app --reload', language="bash")
