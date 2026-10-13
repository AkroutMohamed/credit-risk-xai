"""Documented inference and interpretation service: /docs."""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.artifacts import load_bundle
from src.explain import shap_local, lime_local, human_summary

app = FastAPI(title="Explainable Credit Risk API", version="1.0.0",
              description="Educational credit-default prediction and SHAP/LIME explanation API.")

class Customer(BaseModel):
    features: dict[str, int | float | str | None] = Field(..., description="Single customer input with training feature names")
    threshold: float = Field(0.5, ge=0, le=1)


def bundle():
    try:
        return load_bundle(Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models/bundle.joblib"))))
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(503, str(exc)) from exc


def customer_frame(payload: Customer, model):
    columns = list(model.feature_names_in_)
    missing = [c for c in columns if c not in payload.features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")
    return pd.DataFrame([{c: payload.features[c] for c in columns}])

@app.get("/health")
def health():
    try:
        bundle()
        return {"status": "ready"}
    except HTTPException:
        return {"status": "model_not_loaded"}

@app.post("/predict")
def predict(payload: Customer):
    b = bundle()
    x = customer_frame(payload, b["model"])
    try:
        p = float(b["model"].predict_proba(x)[0, 1])
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"probability_of_default": p, "threshold": payload.threshold,
            "predicted_default": int(p >= payload.threshold)}

@app.post("/explain/shap")
def explain_shap(payload: Customer):
    b = bundle()
    x = customer_frame(payload, b["model"])
    try:
        result = shap_local(b["model"], b["background"], x)
        return {**result, "summary": human_summary(result)}
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc

@app.post("/explain/lime")
def explain_lime(payload: Customer):
    b = bundle()
    x = customer_frame(payload, b["model"])
    try:
        return lime_local(b["model"], b["background"], x)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
