
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from app.schemas import PredictRequest, PredictResponse, HealthResponse, MetadataResponse


# -----------------------
# Paths / config
# -----------------------
APP_NAME = "Home Credit Scoring API"

ART_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ART_DIR / "model.pkl"
FEATURE_COLS_PATH = ART_DIR / "feature_cols.json"
THRESHOLD_PATH = ART_DIR / "threshold_config.json"
TOP_FEATURES_PATH = ART_DIR / "top_features.json"

LOG_DIR = Path(os.getenv("LOG_DIR", "prod_logs"))
PRED_LOG = LOG_DIR / "predictions.jsonl"
ERR_LOG = LOG_DIR / "errors.jsonl"

MODEL_NAME = os.getenv("MODEL_NAME", "XGBoost_Home_Credit_Scoring")
MODEL_VERSION = os.getenv("MODEL_VERSION", None)


# -----------------------
# Globals (loaded at startup)
# -----------------------
model = None
feature_cols: List[str] = []
feature_set: set[str] = set()
feature_index: Dict [str, int] = {}
top_feature_indices: Dict[str, int] = {}

top_features: List[str] = []

threshold: float = 0.5
fn_cost: Optional[float] = None
fp_cost: Optional[float] = None

# -----------------------
# Utils
# -----------------------
def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def to_jsonable(v: Any) -> Any:
    """Convertit NaN / numpy scalars -> JSON-safe."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, np.generic):
        return v.item()
    return v


def make_row_df(payload: Dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, int]:
    """
    - refuse les features inconnues
    - construit un vecteur numpy array float32 de taille (1, n_features)
    - valeurs manquantes -> NaN
    - reture (df, x, missiong_count)
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

    x = np.full((1, len(feature_cols)), np.nan, dtype=np.float32)

    for k, v in payload.items():
        i = feature_index.get(k)
        if i is None or v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            x[0,i] = fv

    missing_count = int(np.isnan(x).sum())
    df = pd.DataFrame(x, columns=feature_cols)
    return df, x, missing_count


def compute_payload_hash(features: Dict[str, Any]) -> str:
    payload_str = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


# -----------------------
# Lifespan: load once
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_cols, feature_set, threshold, fn_cost, fp_cost, top_features, feature_index, top_feature_indices

    # features attendues
    feature_cols = load_json(FEATURE_COLS_PATH)
    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise RuntimeError("feature_cols.json invalide (liste vide ou format invalide)")
    feature_set = set(feature_cols)

    # threshold config
    cfg = load_json(THRESHOLD_PATH)
    threshold = float(cfg.get("threshold", cfg.get("best_threshold", 0.5)))
    fn_cost = cfg.get("fn_cost", None)
    fp_cost = cfg.get("fp_cost", None)

    # top features 
    if TOP_FEATURES_PATH.exists():
        tf = load_json(TOP_FEATURES_PATH)
        if isinstance(tf, list):
            top_features = [f for f in tf if f in feature_set]
        else:
            top_features = []
    else:
        top_features = []

    feature_index = {c: i for i, c in enumerate(feature_cols)}
    top_feature_indices = {f: feature_index[f] for f in top_features if f in feature_index}

    # model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model introuvable: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    yield

    model = None


app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="API de prédiction de risque de défaut de paiement",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Home Credit Scoring API",
        "docs": "/docs",
        "health": "/health",
        "metadata": "/metadata",
        "predict": "/predict"
    }

# -----------------------
# Endpoints
# -----------------------
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
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        threshold=float(threshold),
        fn_cost=float(fn_cost) if fn_cost is not None else None,
        fp_cost=float(fp_cost) if fp_cost is not None else None,
        n_features_expected=len(feature_cols),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = str(uuid.uuid4())
    t_start = time.perf_counter()
    ts = datetime.now(timezone.utc)

    features = req.features or {}

    # Etap 1: hash + comptage 
    t0 = time.perf_counter()
    payload_hash = compute_payload_hash(features)
    n_features_sent = len(features)
    timing_hash_ms = (time.perf_counter() - t0) * 1000

    try:
        # Etape 2: Validation + construction DataFrame
        t0 = time.perf_counter()
        df, x_arr, missing_count = make_row_df(features)

        timing_preprocessing_ms = (time.perf_counter() - t0) * 1000

        # Etape 3: Stats missing + top features
        t0 = time.perf_counter()
        missing_rate = float(missing_count / len(feature_cols)) if feature_cols else None

        # top features values
        top_features_values = {
            f: to_jsonable(x_arr[0, idx]) for f, idx in top_feature_indices.items()
            }

        timing_stats_ms = (time.perf_counter() - t0) * 1000

        # Etape 4: Inférence modèle
        t0 = time.perf_counter()
        proba = float(model.predict_proba(df)[:, 1][0])
        pred = int(proba >= threshold)
        decision = "Refusé" if pred == 1 else "Accepté"
        timing_inference_ms = (time.perf_counter() - t0) * 1000

        latency_ms = float((time.perf_counter() - t_start) * 1000)

        # Etape 5: Logging JSONL
        t0 = time.perf_counter()

        append_jsonl(
            PRED_LOG,
            {
                "request_id": request_id,
                "timestamp": ts.isoformat(),
                "client_id": req.client_id,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "threshold": float(threshold),
                "fn_cost": fn_cost,
                "fp_cost": fp_cost,
                "latency_ms": latency_ms,
                "timings": {                          
                    "hash_ms":          round(timing_hash_ms, 3),
                    "preprocessing_ms": round(timing_preprocessing_ms, 3),
                    "stats_ms":         round(timing_stats_ms, 3),
                    "inference_ms":     round(timing_inference_ms, 3),
                },
                "input": {
                    "payload_hash": payload_hash,
                    "n_features_sent": n_features_sent,
                    "missing_count_aligned": missing_count,
                    "missing_rate_aligned": missing_rate,
                    "top_features": top_features_values,
                },
                "output": {
                    "probability_default": proba,
                    "prediction": pred,
                    "decision": decision,
                },
            },
        )
        timing_logging_ms = (time.perf_counter() - t0) * 1000

        print(
            f"[TIMINGS] hash={timing_hash_ms:.2f}ms | "
            f"preprocessing={timing_preprocessing_ms:.2f}ms | "
            f"stats={timing_stats_ms:.2f}ms | "
            f"inference={timing_inference_ms:.2f}ms | "
            f"logging={timing_logging_ms:.2f}ms | "
            f"TOTAL={latency_ms:.2f}ms"
        )

        return PredictResponse(
            request_id=request_id,
            timestamp=ts,
            latency_ms=latency_ms,
            probability_default=proba,
            threshold=float(threshold),
            prediction=pred,
            decision=decision,
        )

    except HTTPException as e:
        latency_ms = float((time.perf_counter() - t_start) * 1000)
        append_jsonl(
            ERR_LOG,
            {
                "request_id": request_id,
                "timestamp": ts.isoformat(),
                "client_id": req.client_id,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "latency_ms": latency_ms,
                "status_code": e.status_code,
                "error": e.detail,
                "input": {"payload_hash": payload_hash},
            },
        )
        raise

    except Exception as e:
        latency_ms = float((time.perf_counter() - t_start) * 1000)
        append_jsonl(
            ERR_LOG,
            {
                "request_id": request_id,
                "timestamp": ts.isoformat(),
                "client_id": req.client_id,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "latency_ms": latency_ms,
                "status_code": 500,
                "error": str(e),
                "input": {"payload_hash": payload_hash},
            },
        )
        raise HTTPException(
            status_code=422,
            detail={"error": "Prediction failed", "message": str(e)},
        )


@app.exception_handler(Exception)
def global_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)},
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)
