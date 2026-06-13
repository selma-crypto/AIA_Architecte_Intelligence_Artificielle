"""
API FastAPI — Détecteur de Fraude MLOps
========================================
Endpoints:
  GET  /health          → statut du service et du modèle
  POST /predict         → prédiction unitaire (JSON)
  POST /predict/batch   → prédictions en masse (JSON array)
  GET  /metrics         → métriques opérationnelles en temps réel
  GET  /model/info      → informations sur le modèle chargé
  POST /model/reload    → rechargement à chaud du modèle
"""

import os
import time
import pickle
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud-api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "API de détection de fraude e-commerce basée sur un modèle XGBoost "
        "entraîné sur 299 695 transactions. Intégration MLflow pour le versioning "
        "et le suivi des expériences.\n\n"
        "**Seuil de décision par défaut : 0.30** (optimisé Optuna — 150 trials)\n\n"
        "| Métrique | Valeur |\n|---|---|\n"
        "| Recall | 0.869 |\n| Precision | 0.614 |\n| F1 | 0.720 |\n| ROC-AUC | 0.981 |"
    ),
    version="1.0.0",
    contact={"name": "Groupe 4 — CDSD", "email": "k.rochet92@gmail.com"},
    openapi_tags=[
        {"name": "health", "description": "Vérification de l'état du service"},
        {"name": "predictions", "description": "Inférence du modèle XGBoost"},
        {"name": "monitoring", "description": "Métriques opérationnelles et gestion du modèle"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Configuration via variables d'environnement
# ---------------------------------------------------------------------------
MODEL_PATH              = os.getenv("MODEL_PATH", "fraud_xgb_model.pkl")
MLFLOW_TRACKING_URI     = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_MODEL_NAME       = os.getenv("MLFLOW_MODEL_NAME", "fraud-xgboost")
DECISION_THRESHOLD      = float(os.getenv("DECISION_THRESHOLD", "0.30"))

# ---------------------------------------------------------------------------
# État global du service
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {
    "model": None,
    "model_features": None,
    "model_loaded_at": None,
    "model_version": None,
    "prediction_count": 0,
    "fraud_count": 0,
    "error_count": 0,
    "latencies_ms": [],
}

_start_time = time.time()

# ---------------------------------------------------------------------------
# Schémas Pydantic
# ---------------------------------------------------------------------------
class TransactionFeatures(BaseModel):
    account_age_days: float        = Field(..., description="Ancienneté du compte (jours)",            example=365.0)
    hour_local: int                = Field(..., ge=0, le=23, description="Heure locale (0-23)",         example=14)
    dayofweek_local: int           = Field(..., ge=0, le=6,  description="Jour semaine (0=Dim)",        example=3)
    day_local: int                 = Field(..., ge=1, le=31, description="Jour du mois",                example=15)
    is_night: int                  = Field(..., ge=0, le=1,  description="Nocturne (0/1)",              example=0)
    time_since_last: float         = Field(..., description="Secondes depuis dernière transaction",     example=86400.0)
    amount: float                  = Field(..., gt=0, description="Montant en USD",                     example=149.99)
    avg_amount_user: float         = Field(..., description="Montant moyen habituel",                   example=80.0)
    avg_amount_user_past: float    = Field(..., description="Moyenne historique des montants",          example=75.0)
    amount_diff_user_avg: float    = Field(..., description="Écart vs moyenne habituelle",              example=70.0)
    amount_delta_prev: float       = Field(..., description="Écart vs transaction précédente",          example=50.0)
    total_transactions_user: int   = Field(..., description="Nb total de transactions",                 example=120)
    transaction_count_cum: int     = Field(..., description="Compteur cumulé",                          example=120)
    tx_last_24h: int               = Field(..., description="Transactions dernières 24h",               example=2)
    tx_last_7d: int                = Field(..., description="Transactions derniers 7j",                 example=5)
    tx_last_30d: int               = Field(..., description="Transactions derniers 30j",                example=18)
    user_tx_count: int             = Field(..., description="Compteur transactions utilisateur",        example=120)
    user_fraud_rate: float         = Field(..., ge=0, le=1, description="Taux fraude historique",       example=0.0)
    user_fraud_count: int          = Field(..., description="Nb fraudes passées",                       example=0)
    user_has_fraud_history: int    = Field(..., ge=0, le=1, description="A déjà fraudé (0/1)",          example=0)
    is_new_account: int            = Field(..., ge=0, le=1, description="Nouveau compte <30j (0/1)",    example=0)
    shipping_distance_km: float    = Field(..., description="Distance livraison vs IP (km)",            example=5.0)
    distance_amount_ratio: float   = Field(..., description="Ratio distance/montant",                   example=0.03)
    country_bin_mismatch: int      = Field(..., ge=0, le=1, description="Pays carte≠pays déclaré (0/1)",example=0)
    avs_match: int                 = Field(..., ge=0, le=1, description="AVS OK (0/1)",                 example=1)
    cvv_result: int                = Field(..., ge=0, le=1, description="CVV OK (0/1)",                 example=1)
    three_ds_flag: int             = Field(..., ge=0, le=1, description="3DS activé (0/1)",             example=1)
    security_mismatch_score: float = Field(..., ge=0, description="Score incohérence sécurité",         example=0.0)
    # OHE facultatifs (country_*, bin_country_*, merchant_category_*, channel_*)
    extra_features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Variables OHE supplémentaires (country_FR, bin_country_FR, merchant_category_electronics, channel_app, ...)",
        example={"country_FR": 1, "bin_country_FR": 1, "merchant_category_electronics": 1, "channel_app": 0},
    )


class PredictionRequest(BaseModel):
    transaction_id: Optional[str] = Field(None, description="Identifiant optionnel", example="TX-001")
    features: TransactionFeatures
    threshold: Optional[float]    = Field(None, ge=0.0, le=1.0, description="Seuil (défaut 0.30)", example=0.30)


class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float
    is_fraud: bool
    risk_level: str  = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    threshold_used: float
    model_version: Optional[str]
    inference_time_ms: float
    timestamp: str


class BatchPredictionRequest(BaseModel):
    transactions: List[PredictionRequest] = Field(..., min_items=1, max_items=1000)


class BatchPredictionResponse(BaseModel):
    count: int
    fraud_detected: int
    fraud_rate: float
    predictions: List[PredictionResponse]
    total_inference_time_ms: float


class ModelInfo(BaseModel):
    model_path: str
    model_version: Optional[str]
    loaded_at: Optional[str]
    feature_count: Optional[int]
    decision_threshold: float
    mlflow_tracking_uri: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    timestamp: str


class MetricsResponse(BaseModel):
    total_predictions: int
    fraud_predictions: int
    fraud_rate: float
    error_count: int
    avg_latency_ms: float
    p95_latency_ms: float
    model_info: ModelInfo


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def _risk_level(proba: float) -> str:
    if proba < 0.15:   return "LOW"
    elif proba < 0.30: return "MEDIUM"
    elif proba < 0.60: return "HIGH"
    return "CRITICAL"


def _features_to_df(tx: TransactionFeatures, model_features) -> pd.DataFrame:
    data = {k: v for k, v in tx.dict(exclude={"extra_features"}).items()}
    if tx.extra_features:
        data.update(tx.extra_features)
    df = pd.DataFrame([data])
    if model_features:
        df = df.reindex(columns=model_features, fill_value=0)
    return df


# ---------------------------------------------------------------------------
# Chargement modèle
# ---------------------------------------------------------------------------
def load_model(path: str = MODEL_PATH) -> bool:
    global _state
    if Path(path).exists():
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            model    = data["model"] if isinstance(data, dict) and "model" in data else data
            features = data.get("final_columns") if isinstance(data, dict) else None
            if features is None:
                features = getattr(model, "feature_names_in_", None)
            _state.update({
                "model": model,
                "model_features": list(features) if features is not None else None,
                "model_loaded_at": datetime.utcnow().isoformat(),
                "model_version": hashlib.md5(Path(path).read_bytes()).hexdigest()[:8],
            })
            logger.info(f"Modèle chargé: {path} | version: {_state['model_version']}")
            return True
        except Exception as e:
            logger.error(f"Erreur chargement pickle: {e}")

    # Fallback: MLflow Model Registry
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}/Production")
        _state.update({
            "model": model,
            "model_features": list(getattr(model, "feature_names_in_", None) or []),
            "model_loaded_at": datetime.utcnow().isoformat(),
            "model_version": "mlflow-production",
        })
        logger.info("Modèle chargé depuis MLflow Registry")
        return True
    except Exception as e:
        logger.warning(f"MLflow non disponible: {e}")

    logger.error("Aucun modèle disponible.")
    return False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    logger.info("Démarrage API Fraud Detection v1.0.0")
    load_model()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["health"], summary="Statut du service")
async def health():
    """Retourne 200 si le service est opérationnel, 503 si le modèle est absent."""
    ok = _state["model"] is not None
    resp = HealthResponse(
        status="ok" if ok else "degraded",
        model_loaded=ok,
        uptime_seconds=round(time.time() - _start_time, 2),
        timestamp=datetime.utcnow().isoformat(),
    )
    return JSONResponse(status_code=200 if ok else 503, content=resp.dict())


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"],
          summary="Prédiction unitaire")
async def predict(request: PredictionRequest):
    """
    Soumet **une transaction** et reçoit :
    - `fraud_probability` : score de risque (0–1)
    - `is_fraud` : décision binaire
    - `risk_level` : LOW / MEDIUM / HIGH / CRITICAL
    - `inference_time_ms` : latence de l'inférence

    Le seuil par défaut est **0.30** — surcharger avec `threshold` si nécessaire.
    """
    if _state["model"] is None:
        raise HTTPException(503, "Modèle non disponible.")
    t0 = time.time()
    try:
        df    = _features_to_df(request.features, _state["model_features"])
        proba = float(_state["model"].predict_proba(df)[0][1])
        thr   = request.threshold if request.threshold is not None else DECISION_THRESHOLD
        fraud = proba >= thr
        ms    = round((time.time() - t0) * 1000, 2)

        _state["prediction_count"] += 1
        _state["fraud_count"]      += int(fraud)
        _state["latencies_ms"].append(ms)
        if len(_state["latencies_ms"]) > 10_000:
            _state["latencies_ms"] = _state["latencies_ms"][-5_000:]

        logger.info(f"predict | tx={request.transaction_id or 'anon'} | p={proba:.4f} | fraud={fraud} | {ms}ms")
        return PredictionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=round(proba, 6),
            is_fraud=fraud,
            risk_level=_risk_level(proba),
            threshold_used=thr,
            model_version=_state["model_version"],
            inference_time_ms=ms,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        _state["error_count"] += 1
        logger.error(f"Erreur inférence: {e}", exc_info=True)
        raise HTTPException(500, f"Erreur d'inférence: {e}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["predictions"],
          summary="Prédiction en masse (max 1000 transactions)")
async def predict_batch(request: BatchPredictionRequest):
    """
    Soumet jusqu'à **1 000 transactions** en une seule requête.

    Retourne les prédictions individuelles + résumé (nb fraudes, taux de fraude).
    """
    if _state["model"] is None:
        raise HTTPException(503, "Modèle non disponible.")
    t0      = time.time()
    results = []
    for tx in request.transactions:
        try:
            df    = _features_to_df(tx.features, _state["model_features"])
            proba = float(_state["model"].predict_proba(df)[0][1])
            thr   = tx.threshold if tx.threshold is not None else DECISION_THRESHOLD
            fraud = proba >= thr
            results.append(PredictionResponse(
                transaction_id=tx.transaction_id,
                fraud_probability=round(proba, 6),
                is_fraud=fraud,
                risk_level=_risk_level(proba),
                threshold_used=thr,
                model_version=_state["model_version"],
                inference_time_ms=0.0,
                timestamp=datetime.utcnow().isoformat(),
            ))
        except Exception as e:
            _state["error_count"] += 1
            logger.error(f"Erreur batch tx={tx.transaction_id}: {e}")

    total_ms    = round((time.time() - t0) * 1000, 2)
    fraud_count = sum(1 for r in results if r.is_fraud)
    _state["prediction_count"] += len(results)
    _state["fraud_count"]      += fraud_count
    return BatchPredictionResponse(
        count=len(results),
        fraud_detected=fraud_count,
        fraud_rate=round(fraud_count / len(results), 4) if results else 0.0,
        predictions=results,
        total_inference_time_ms=total_ms,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["monitoring"],
         summary="Métriques opérationnelles en temps réel")
async def metrics():
    """Volume de prédictions, taux de fraude, latences P50/P95, infos modèle."""
    lats = _state["latencies_ms"]
    total = _state["prediction_count"]
    fraud = _state["fraud_count"]
    return MetricsResponse(
        total_predictions=total,
        fraud_predictions=fraud,
        fraud_rate=round(fraud / total, 4) if total else 0.0,
        error_count=_state["error_count"],
        avg_latency_ms=round(float(np.mean(lats)), 2) if lats else 0.0,
        p95_latency_ms=round(float(np.percentile(lats, 95)), 2) if lats else 0.0,
        model_info=ModelInfo(
            model_path=MODEL_PATH,
            model_version=_state["model_version"],
            loaded_at=_state["model_loaded_at"],
            feature_count=len(_state["model_features"]) if _state["model_features"] else None,
            decision_threshold=DECISION_THRESHOLD,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        ),
    )


@app.get("/model/info", response_model=ModelInfo, tags=["monitoring"],
         summary="Informations sur le modèle actif")
async def model_info():
    return ModelInfo(
        model_path=MODEL_PATH,
        model_version=_state["model_version"],
        loaded_at=_state["model_loaded_at"],
        feature_count=len(_state["model_features"]) if _state["model_features"] else None,
        decision_threshold=DECISION_THRESHOLD,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    )


@app.post("/model/reload", tags=["monitoring"], summary="Rechargement à chaud du modèle")
async def reload_model(background_tasks: BackgroundTasks):
    """Déclenche un rechargement sans redémarrer le service (utile après réentraînement)."""
    background_tasks.add_task(load_model)
    return {"status": "reload_scheduled", "message": "Rechargement planifié en arrière-plan."}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
