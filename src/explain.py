
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def transformed(model, x: pd.DataFrame):
    prep = model.named_steps["prep"]
    return np.asarray(prep.transform(x), dtype=float), list(prep.get_feature_names_out())


def shap_local(model, background: pd.DataFrame, customer: pd.DataFrame, max_background=40):
    import shap
    bg, names = transformed(model, background.sample(n=min(len(background), max_background), random_state=42))
    row, _ = transformed(model, customer)
    clf = model.named_steps["model"]
    def probability(a):
        return clf.predict_proba(np.asarray(a))[:, 1]
    explainer = shap.Explainer(probability, bg, algorithm="permutation")
    explanation = explainer(row, max_evals=max(2 * len(names) + 1, 51))
    return {"base_value": float(np.asarray(explanation.base_values).ravel()[0]),
            "prediction": float(probability(row)[0]), "features": names,
            "contributions": np.asarray(explanation.values)[0].tolist()}


def lime_local(model, background: pd.DataFrame, customer: pd.DataFrame, num_features=10):
    from lime.lime_tabular import LimeTabularExplainer
    bg, names = transformed(model, background)
    row, _ = transformed(model, customer)
    explainer = LimeTabularExplainer(bg, feature_names=names, class_names=["No default", "Default"],
                                     mode="classification", random_state=42, discretize_continuous=True)
    explanation = explainer.explain_instance(row[0], model.named_steps["model"].predict_proba,
                                             num_features=min(num_features, len(names)), num_samples=600)
    return {"prediction": float(model.predict_proba(customer)[0, 1]),
            "rules": [{"condition": key, "weight": float(weight)} for key, weight in explanation.as_list(label=1)]}


def global_importance(model, x: pd.DataFrame, y: pd.Series, max_rows=300):
    if len(x) > max_rows:
        subset = x.sample(n=max_rows, random_state=42)
        y = y.loc[subset.index]
        x = subset
    result = permutation_importance(model, x, y, n_repeats=3, scoring="roc_auc", random_state=42, n_jobs=1)
    return pd.DataFrame({"Feature": x.columns, "Importance": result.importances_mean}).sort_values("Importance", ascending=False)


def human_summary(shap_result, top=3):
    weights = np.asarray(shap_result["contributions"])
    names = shap_result["features"]
    order = np.argsort(-np.abs(weights))[:top]
    factors = [f"{names[i]} ({'increases' if weights[i] > 0 else 'decreases'} estimated risk)" for i in order]
    return "Main model drivers: " + ", ".join(factors) + ". These are model explanations, not causal claims."
