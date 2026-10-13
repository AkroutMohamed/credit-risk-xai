
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_curve)
from xgboost import XGBClassifier

MODELS = ["Logistic Regression", "Decision Tree", "KNN", "Random Forest", "XGBoost"]

@dataclass
class Split:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_data(df: pd.DataFrame, features: list[str], target="default", test_size=0.2, seed=42) -> Split:
    if target in features or not features:
        raise ValueError("Choose non-target features.")
    if df[target].nunique() != 2:
        raise ValueError("Target must be binary.")
    xtr, xte, ytr, yte = train_test_split(df[features], df[target], test_size=test_size,
                                           random_state=seed, stratify=df[target])
    return Split(xtr, xte, ytr, yte)


def preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in x if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
    ], remainder="drop", sparse_threshold=0)


def build_model(name: str, x_train: pd.DataFrame, seed=42) -> Pipeline:
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=seed),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=160, max_depth=4, learning_rate=0.06,
                                  eval_metric="logloss", random_state=seed, n_jobs=2),
    }
    if name not in estimators:
        raise ValueError(f"Unsupported model: {name}")
    return Pipeline([("prep", preprocessor(x_train)), ("model", estimators[name])])

SEARCH = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
    "Decision Tree": {"model__max_depth": [3, 5, 8, 12, None], "model__min_samples_leaf": [1, 5, 20]},
    "KNN": {"model__n_neighbors": [3, 5, 9, 15], "model__weights": ["uniform", "distance"]},
    "Random Forest": {"model__n_estimators": [100, 160], "model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5]},
    "XGBoost": {"model__n_estimators": [100, 160], "model__max_depth": [3, 5], "model__learning_rate": [0.03, 0.1]},
}


def fit_models(split: Split, chosen: list[str], tune=False, n_iter=6, seed=42):
    if not chosen:
        raise ValueError("Select at least one model.")
    result = {}
    for name in chosen:
        estimator = build_model(name, split.x_train, seed)
        if tune:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            estimator = RandomizedSearchCV(estimator, SEARCH[name], n_iter=min(n_iter, np.prod([len(v) for v in SEARCH[name].values()])),
                                          scoring="roc_auc", cv=cv, random_state=seed, n_jobs=1, refit=True)
        estimator.fit(split.x_train, split.y_train)
        result[name] = estimator.best_estimator_ if tune else estimator
    return result


def ks_stat(y_true, probabilities) -> float:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    return float(np.max(tpr - fpr))


def metrics(y_true, probabilities, threshold=0.5) -> dict:
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    auc = roc_auc_score(y_true, probabilities)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"ROC-AUC": float(auc), "PR-AUC": float(average_precision_score(y_true, probabilities)),
            "Precision": float(precision_score(y_true, pred, zero_division=0)),
            "Recall": float(recall_score(y_true, pred, zero_division=0)),
            "F1": float(f1_score(y_true, pred, zero_division=0)),
            "Gini": float(2 * auc - 1), "KS": ks_stat(y_true, probabilities),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def compare(models: dict, split: Split, threshold=0.5) -> pd.DataFrame:
    out = []
    for name, model in models.items():
        p = model.predict_proba(split.x_test)[:, 1]
        out.append({"Model": name, **metrics(split.y_test, p, threshold)})
    return pd.DataFrame(out).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
