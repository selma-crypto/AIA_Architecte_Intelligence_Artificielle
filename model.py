"""
model.py — Chargement et inférence du modèle en production.

Le pipeline sklearn (préprocesseur + classifieur) est chargé une seule fois
au démarrage. Les prédictions s'appuient sur les mêmes features
que celles créées dans train.py.
"""

import logging
import pickle
from datetime import date, datetime
from functools import lru_cache

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_pipeline():
    """Charge le pipeline depuis le disque (une seule fois)."""
    with open(config.MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    logger.info("Modèle chargé depuis %s", config.MODEL_PATH)
    return pipeline


def _build_feature_row(tx: dict) -> pd.DataFrame:
    """
    Reconstruit les features à partir d'une transaction normalisée.
    Doit être cohérent avec train.py::add_features().
    """
    trans_dt: datetime = tx.get("trans_datetime") or datetime.now()
    dob: date          = tx.get("dob") or date(1990, 1, 1)

    dob_dt  = datetime.combine(dob, datetime.min.time())
    age     = (trans_dt - dob_dt).days // 365
    hour    = trans_dt.hour

    lat, long          = float(tx.get("lat", 0)),       float(tx.get("long", 0))
    merch_lat, merch_long = float(tx.get("merch_lat", 0)), float(tx.get("merch_long", 0))
    distance_km = np.sqrt((lat - merch_lat) ** 2 + (long - merch_long) ** 2) * 111

    return pd.DataFrame([{
        # Numériques
        "amt":         float(tx.get("amt", 0)),
        "city_pop":    int(tx.get("city_pop", 0)),
        "hour":        hour,
        "age":         age,
        "distance_km": distance_km,
        "lat":         lat,
        "long":        long,
        # Catégorielles
        "category":    str(tx.get("category", "")),
        "gender":      str(tx.get("gender", "")),
        "state":       str(tx.get("state", "")),
    }])


def predict(tx: dict) -> tuple[float, bool]:
    """
    Retourne (probabilité_fraude, est_fraude).
    est_fraude = True si proba >= FRAUD_THRESHOLD.
    """
    pipeline = _load_pipeline()
    X = _build_feature_row(tx)

    fraud_proba: float = float(pipeline.predict_proba(X)[0, 1])
    is_fraud: bool     = fraud_proba >= config.FRAUD_THRESHOLD

    return fraud_proba, is_fraud
