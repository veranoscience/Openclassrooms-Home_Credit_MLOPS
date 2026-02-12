from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
import numpy as np

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Response

from app.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    MetadataResponse,
)

import hashlib

app_name = "Home Scredit Scoring API"
art_dir = Path(__file__).resolve().parent / "artifacts"

model_path = art_dir / "model.pkl"
feature_cols_path = art_dir / "feature_cols.json"
threshold_path = art_dir /  "threshold_config.json"

LOG_DIR = Path(os.getenv("LOG_DIR", "prod_logs"))
PRED_LOG = LOG_DIR / "predictions.jsonl"
ERR_LOG = LOG_DIR / "errors.jsonl"

# Globals chargés au startup
model = None
feature_cols: List[str] = []
feature_set = set()
threshold: float = 0.5
fn_cost: Optional[float] = None
fp_cost: Optional[float] = None

model_name = os.getenv("MODEL_NAME", "XGBoost_Home_Credit_Scoring")
model_version = os.getenv("MODEL_VERSION", None)

def load_json (path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return json.loads(path.read_text())

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def make_row_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    - bloque les features inconnues
    - complète les features manquantes par None
    - aligne l'ordre des colonnes exactement comme au training
    """
    unknown = sorted(set(payload.keys()) - feature_set)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Unknown feature(s) provided",
                "count": len(unknown),
                "examples": unknown[:20],
            },
        )
    
    # on remplit les features attendues (missing -> None)
    row = {c: payload.get(c, None) for c in feature_cols}
    df = pd.DataFrame([row], columns=feature_cols)

    #None->NaN, inf->Nan
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan: charge le modèle + configs une seule fois au démarrage.
    """
    global model, feature_cols, feature_set, threshold, fn_cost, fp_cost

    feature_cols = load_json(feature_cols_path)
    if not isinstance (feature_cols, list) or len(feature_cols) == 0:
        raise RuntimeError("feature_cols.json invalide (liste vide ou format invalide)")
    feature_set = set(feature_cols)

    #load threshold config
    cfg = load_json(threshold_path)
    threshold = float(cfg.get("threshold", cfg.get("best_threshold", 0.5)))
    fn_cost = cfg.get("fn_cost", None)
    fp_cost = cfg.get("fp_cost", None)

    # load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model introuvable: {model_path}")
    model = joblib.load(model_path)

    #startup done
    yield

    #shutdown
    model = None
app = FastAPI(
    title=app_name,
    version = "1.0.0",
    description=(
        "API de prédiction de risque de défaut de paiement"
    ),
    lifespan=lifespan,
)

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        n_features_expected=len(feature_cols) if feature_cols else None,
    )

@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MetadataResponse(
        model_name=model_name,
        model_version=model_version,
        threshold=float(threshold),
        fn_cost=float(fn_cost) if fn_cost is not None else None,
        fp_cost=float(fp_cost) if fp_cost is not None else None,
        n_features_expected=len(feature_cols),
    )

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Prédit le risque de défaut pour un client
    
    Client: Données du client validées par Pydantic
        
    Returns:
        PredictionOutput avec la probabilité et la décision
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    timestamp = datetime.now(timezone.utc)

    payload_hash = hashlib.sha256(
        json.dumps(req.features, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    # utile pour monitoring “qualité d’entrée”
    n_payload_features = len(req.features) if req.features else 0

    try:
        X_one = make_row_df(req.features)

        # stats missing sur la ligne alignée
        missing_count = int(pd.isna(X_one.iloc[0]).sum())
        missing_rate = float(missing_count / len(feature_cols)) if feature_cols else None

        proba = float(model.predict_proba(X_one)[:, 1][0])
        pred = int(proba >= threshold)
        decision = "Refusé" if pred == 1 else "Accepté"

        latency_ms = float((time.perf_counter() - t0) * 1000)


        # log succès
        append_jsonl(PRED_LOG, {
            "request_id": request_id,
            "timestamp": timestamp.isoformat(),
            "client_id": req.client_id,
            "model_name": model_name,
            "model_version": model_version,
            "threshold": float(threshold),
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "latency_ms": latency_ms,
            "input": {
                "payload_hash": payload_hash,
                "n_features_sent": n_payload_features,
                "missing_count_aligned": missing_count,
                "missing_rate_aligned": missing_rate,
            },
            "output": {
                "probability_default": proba,
                "prediction": pred,
                "decision": decision,
            },
        })

        return PredictResponse(
            request_id=request_id,
            timestamp=timestamp,
            latency_ms=float(latency_ms),
            probability_default=proba,
            threshold=float(threshold),
            prediction=pred,
            decision=decision,
        )

    except HTTPException as e:
        latency_ms = float((time.perf_counter() - t0) * 1000)
        append_jsonl(ERR_LOG, {
            "request_id": request_id,
            "timestamp": timestamp.isoformat(),
            "client_id": req.client_id,
            "model_name": model_name,
            "model_version": model_version,
            "latency_ms": latency_ms,
            "status_code": e.status_code,
            "error": e.detail,
            "input": {"payload_hash": payload_hash},
        })
        raise

    except Exception as e:
        latency_ms = float((time.perf_counter() - t0) * 1000)
        append_jsonl(ERR_LOG, {
            "request_id": request_id,
            "timestamp": timestamp.isoformat(),
            "client_id": req.client_id,
            "model_name": model_name,
            "model_version": model_version,
            "latency_ms": latency_ms,
            "status_code": 500,
            "error": str(e),
            "input": {"payload_hash": payload_hash},
        })
        raise HTTPException(status_code=422, detail={"error": "Prediction failed", "message": str(e)})

@app.exception_handler(Exception)
def global_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)},
    )

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)