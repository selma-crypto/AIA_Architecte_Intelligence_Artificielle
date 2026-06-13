"""
fix_model_pkl.py
================
Repackage xgboost_fraud_optuna.pkl → fraud_xgb_model.pkl
dans le format attendu par app.py ET api/main.py.

Corrections appliquées :
  1. Ajout de la clé 'final_columns' (requise par app.py)
  2. Ajout des 3 features manquantes avec valeur 0 par défaut
     (hour_local, dayofweek_local, day_local)

Usage : python fix_model_pkl.py
"""

import pickle
from pathlib import Path

SRC  = Path("xgboost_fraud_optuna.pkl")
DEST = Path("fraud_xgb_model.pkl")

# ── Chargement ──────────────────────────────────────────────────────────────
print(f"Chargement : {SRC}")
with open(SRC, "rb") as f:
    data = pickle.load(f)

print(f"  Type     : {type(data)}")
print(f"  Clés     : {list(data.keys()) if isinstance(data, dict) else 'objet direct'}")

# ── Extraire modèle et features ─────────────────────────────────────────────
def _safe_features(obj):
    """Extrait les noms de features depuis un modèle, sans planter sur numpy array."""
    attr = getattr(obj, "feature_names_in_", None)
    if attr is None:
        return []
    try:
        return list(attr)
    except Exception:
        return []

if isinstance(data, dict):
    model = data.get("model") or data.get("xgb_model")
    feature_names = (
        data.get("feature_names")
        or data.get("final_columns")
        or _safe_features(model)
    )
    # feature_names peut lui aussi être un numpy array dans le dict
    if not isinstance(feature_names, list):
        try:
            feature_names = list(feature_names)
        except Exception:
            feature_names = []
    existing_metrics = data.get("metrics", {})
else:
    # Objet direct (XGBClassifier, Pipeline, etc.)
    model            = data
    feature_names    = _safe_features(data)
    existing_metrics = {}

print(f"  Modèle   : {type(model).__name__}")

# ── Utiliser les features internes du booster XGBoost (source de vérité) ────
# Les feature_names du booster sont exactement ce que le modèle attend.
# Ne pas ajouter de features qui n'étaient pas dans les données d'entraînement.
try:
    booster_features = model.get_booster().feature_names
    if booster_features:
        feature_names = list(booster_features)
        print(f"  Features (booster): {len(feature_names)} ← source de vérité")
    else:
        print(f"  Features (pkl)    : {len(feature_names)}")
except Exception:
    print(f"  Features (pkl)    : {len(feature_names)}")

print(f"  5 premières : {feature_names[:5]}")

# ── Construire l'artefact final ──────────────────────────────────────────────
artifact = {
    "model":         model,
    "final_columns": feature_names,   # clé attendue par app.py
    "feature_names": feature_names,   # clé attendue par api/main.py
    "metrics":       existing_metrics,
}

# ── Sauvegarde ───────────────────────────────────────────────────────────────
with open(DEST, "wb") as f:
    pickle.dump(artifact, f)

size_kb = DEST.stat().st_size / 1024
print(f"\n✅ Sauvegardé : {DEST}  ({size_kb:.0f} KB)")
print(f"   final_columns : {len(feature_names)} features")

# ── Vérification rapide ───────────────────────────────────────────────────────
print("\nVérification de rechargement...")
with open(DEST, "rb") as f:
    check = pickle.load(f)

assert "model"         in check, "Clé 'model' manquante !"
assert "final_columns" in check, "Clé 'final_columns' manquante !"
assert len(check["final_columns"]) > 0, "final_columns vide !"

import numpy as np
import pandas as pd
test_df = pd.DataFrame(
    [[0.0] * len(check["final_columns"])],
    columns=check["final_columns"]
)
proba = check["model"].predict_proba(test_df)[0][1]
print(f"  Test inférence  : proba = {proba:.4f}  ✅")
print("\nfraud_xgb_model.pkl est prêt. Placez-le à la racine du repo.")
