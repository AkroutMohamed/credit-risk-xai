# Explainable Credit Default Risk Lab

A finance-oriented, educational credit-default decision-support prototype. Built for **Projet Challenge 1: Application Web Interactive de Classification Explicable (ENSI, 2026–2027)**.

## Assignment coverage

| Required module | Implementation |
|---|---|
| Data | Automatic download (UCI id 350), CSV upload, audit, descriptive stats, class balance, distributions, correlation, default rate |
| Preparation | Target/predictor selection, train-only imputation, encoding and scaling, configurable stratified train/test split |
| Modeling | Logistic Regression, Decision Tree, KNN, Random Forest, XGBoost, optional randomized CV hyperparameter search |
| Evaluation | ROC-AUC, PR-AUC, precision, recall, F1, Gini, KS, ROC comparison, PR curve, confusion matrix, interactive threshold |
| XAI | Permutation feature importance, SHAP beeswarm/dependence, SHAP local waterfall, LIME individual explanation, natural-language summary |
| API | FastAPI `/predict`, `/explain/shap`, `/explain/lime`, `/health`, `/docs`; optional invocation from dashboard |

## Quick start (Python 3.11 recommended)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Click **Train / retrain models** in the sidebar. The default UCI dataset is downloaded over the network on first use; if the source is unavailable, upload your own labeled CSV. Model training can take minutes on slower hardware. SHAP is expensive, so global and local explanations are button-triggered and sampled.

## Start the API

1. Train a model in the dashboard.
2. Open tab **6 · API** and download `bundle.joblib`.
3. Save the file as `models/bundle.joblib` locally. **Only load model files you trained yourself**; joblib uses pickle serialization and untrusted files may execute arbitrary code.
4. In a second terminal, from repository root:

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`. In the dashboard, open **5 · Explainability** and click **Ask FastAPI for prediction**. The standalone API needs the bundle from the same training run to return corresponding results.

Example using a full case from your dataset (the API requires all training features):

```python
import requests
import pandas as pd
customer = pd.read_csv('your_data.csv').drop(columns=['default']).iloc[0].to_dict()
print(requests.post('http://127.0.0.1:8000/predict',json={'features':customer,'threshold':0.5}).json())
```

## Test

```bash
python -m pytest -q
```

## Architecture

```text
UCI (ucimlrepo) / custom CSV → Data & EDA → Stratified split
                                 ↓
                          sklearn ColumnTransformer
                          (fit on train only)
                                 ↓
              5 candidate models + randomized CV (optional)
                                 ↓
            Evaluation/threshold and financial metrics
                                 ↓
                   SHAP + LIME explanations
                                 ↓
Streamlit dashboard ← HTTP optional → FastAPI + trusted model bundle
```

## Deployment

- **Streamlit Community Cloud:** Push the repository to GitHub. Set the entrypoint to `app/streamlit_app.py` and Python to 3.11 if configurable. The app needs network access for its first UCI download. Dashboard training is ephemeral and restarts when the service sleeps; do not expect models to persist across sessions.
- **Hugging Face Spaces:** A Docker Space can serve both API and Streamlit behind a proxy; simpler is separate services: Streamlit frontend and API deployed separately. Set the dashboard API URL to the publicly hosted FastAPI endpoint. Both deployments must use the same approved model artifact for identical scores.
- Run `pytest` and inspect logs before sharing public links. No actual deployment is included in this repository.

## Financial interpretation

The positive class is **default payment next month** (1). ROC-AUC indicates ranking discrimination; Gini = 2·ROC-AUC − 1; KS = max(TPR − FPR). PR-AUC is average precision. Threshold controls the trade-off of missed defaults vs false positives. None is a profitability metric, and predicted probabilities are **not guaranteed calibrated**. If deploying in an actual lending setting, independently validate temporal stability, calibration, bias, data provenance, governance, adverse action reason quality and local law compliance.

## Key limitations

- This is a public historical **Taiwan credit-card default** dataset; it does not directly represent loan origination or current Tunisian borrowers. It is not a production loan approval tool.
- Training/evaluation uses a random stratified split for this course; real risk validation generally needs out-of-time holdouts and leakage reviews.
- SHAP/LIME explain associations within a fitted model, not why someone genuinely defaulted. Permutation importance, SHAP and LIME may disagree.
- Sensitive attributes/proxies may produce discriminatory effects; their predictive importance does not justify using them in lending decisions.
- Hyperparameter search runs CV on the training set; test scores should only be used for final evaluation, not repeated model selection.
- Explanations use transformed numeric/one-hot feature names; a production system should aggregate categorical groups and offer human-readable legally reviewed reasons.

## Files

`src/data.py` data access, `src/ml.py` ML/metrics, `src/explain.py` explainers, `src/artifacts.py` model bundles, `api/main.py` endpoints, `app/streamlit_app.py` dashboard, `tests/` tests, `notebooks/` starter exploration.

## Final submission checklist

- [ ] Upload repo to GitHub; include screenshots of each module.
- [ ] Deploy the app and a documented reachable FastAPI service; verify frontend calls the public API.
- [ ] Train on UCI data and save actual evaluation numbers and images.
- [ ] Complete `ONE_PAGER_TEMPLATE.md` using **measured** values; export to one-page PDF.
- [ ] Discuss logistic regression vs XGBoost, SHAP vs LIME, fairness and risk-policy implications.

> Never use real identifiable customer data or share secrets in the application or screenshots.
