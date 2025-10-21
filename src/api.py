from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Union, Optional, Literal
import joblib
import glob
import os
from sklearn.datasets import load_breast_cancer
import numpy as np


app = FastAPI(title="Breast Cancer Classifier API")


def _latest_calibrated_model_path() -> Optional[str]:
    pattern = os.path.join("models", "model_*_svc_calibrated.joblib")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    # Sort by timestamp embedded in filename (lex order works with YYYYMMDDHHMMSS)
    candidates.sort()
    return candidates[-1]


def _load_model(path: str):
    return joblib.load(path)


BC = load_breast_cancer()
FEATURE_NAMES = list(BC.feature_names)


class PredictBody(BaseModel):
    features: Union[List[float], Dict[str, float]]


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Service running. See /docs for Swagger UI.", "docs": "/docs", "health": "/health"}


@app.get("/feature_names")
def feature_names():
    return {"feature_names": FEATURE_NAMES}


@app.post("/predict")
def predict(body: PredictBody):
    model_path = _latest_calibrated_model_path()
    if model_path is None:
        raise HTTPException(status_code=503, detail="No calibrated model found in models/.")

    model = _load_model(model_path)

    # Build input row
    if isinstance(body.features, list):
        if len(body.features) != len(FEATURE_NAMES):
            raise HTTPException(status_code=400, detail=f"Expected {len(FEATURE_NAMES)} features, got {len(body.features)}.")
        X = np.array(body.features, dtype=float).reshape(1, -1)
    elif isinstance(body.features, dict):
        try:
            X = np.array([body.features[name] for name in FEATURE_NAMES], dtype=float).reshape(1, -1)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Missing feature: {e.args[0]}")
    else:
        raise HTTPException(status_code=400, detail="Invalid features format.")

    # Predict
    pred = int(model.predict(X)[0])
    proba: Optional[float] = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])

    return {
        "model_path": model_path,
        "prediction": pred,
        "probability_positive": proba,
    }


@app.get("/samples")
def get_samples(
    format: Literal["dict", "list"] = Query("dict", description="Return features as dict or list"),
    label: Literal["benign", "malignant", "any"] = Query("any", description="Filter by true label"),
    n: int = Query(2, ge=1, le=10, description="Number of samples to return"),
    seed: Optional[int] = Query(None, description="Random seed for reproducibility"),
):
    rng = np.random.default_rng(seed)

    X = BC.data
    y = BC.target  # 0=malignant, 1=benign
    label_names = BC.target_names  # ["malignant", "benign"]

    if label == "any":
        idx_pool = np.arange(X.shape[0])
    else:
        desired = 1 if label == "benign" else 0
        idx_pool = np.where(y == desired)[0]

    if idx_pool.size == 0:
        raise HTTPException(status_code=404, detail=f"No samples found for label '{label}'.")

    k = int(min(max(1, n), idx_pool.size))
    chosen = rng.choice(idx_pool, size=k, replace=False)

    samples = []
    for i in chosen.tolist():
        row = X[i]
        true_idx = int(y[i])
        true_name = str(label_names[true_idx])

        if format == "dict":
            feat = {name: float(val) for name, val in zip(FEATURE_NAMES, row)}
        else:
            feat = [float(v) for v in row.tolist()]

        samples.append({
            "source_index": int(i),
            "true_label_index": true_idx,
            "true_label_name": true_name,
            "features": feat,
            "payload": {"features": feat},
        })

    return {
        "format": format,
        "label": label,
        "count": len(samples),
        "feature_names": FEATURE_NAMES,
        "samples": samples,
    }
