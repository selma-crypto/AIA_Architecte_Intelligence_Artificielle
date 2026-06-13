"""
generate_test_model.py
======================
Génère fraud_xgb_model.pkl à partir des fichiers CSV sample.
Usage : python generate_test_model.py

Le modèle produit est fonctionnel pour tester l'API et le DAG.
Il est moins performant que le modèle entraîné sur 300k lignes,
mais le format .pkl est identique et 100% compatible.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Chargement des données ───────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
X_PATH   = DATA_DIR / "X_test_app_sample.csv"
Y_PATH   = DATA_DIR / "y_test_app_sample.csv"
OUT_PATH = Path(__file__).parent / "fraud_xgb_model.pkl"

print("Chargement des données...")
X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH).iloc[:, 0].astype(int)

print(f"  X: {X.shape} | fraudes: {y.sum()}/{len(y)} ({y.mean():.1%})")

# ── Préparation des features ─────────────────────────────────────────────────
# Supprimer user_id (identifiant, pas une feature prédictive)
if "user_id" in X.columns:
    X = X.drop(columns=["user_id"])

# Convertir les booléens en int (XGBoost ne les accepte pas en bool)
bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

# Remplir les NaN résiduels
X = X.fillna(X.median(numeric_only=True))

final_columns = list(X.columns)
print(f"  Features finales: {len(final_columns)}")

# ── Entraînement XGBoost ─────────────────────────────────────────────────────
print("\nEntraînement XGBoost...")
try:
    from xgboost import XGBClassifier
except ImportError:
    print("Installation de xgboost...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    from xgboost import XGBClassifier

# scale_pos_weight compense le déséquilibre de classes
neg, pos = (y == 0).sum(), (y == 1).sum()
spw = neg / pos if pos > 0 else 5.0

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=spw,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
model.fit(X, y, verbose=False)

# ── Évaluation rapide ────────────────────────────────────────────────────────
from sklearn.metrics import classification_report, roc_auc_score

probas  = model.predict_proba(X)[:, 1]
y_pred  = (probas >= 0.30).astype(int)
auc     = roc_auc_score(y, probas)

print("\nMétriques sur les données d'entraînement (indicatif) :")
print(classification_report(y, y_pred, target_names=["Légitime", "Fraude"], zero_division=0))
print(f"ROC-AUC : {auc:.4f}")

# ── Sauvegarde ────────────────────────────────────────────────────────────────
artifact = {
    "model":         model,
    "final_columns": final_columns,
    "metrics": {
        "roc_auc":    round(auc, 4),
        "n_features": len(final_columns),
        "trained_on": "X_test_app_sample.csv (500 rows — test model)",
    },
}

with open(OUT_PATH, "wb") as f:
    pickle.dump(artifact, f)

size_kb = OUT_PATH.stat().st_size / 1024
print(f"\n✅ Modèle sauvegardé : {OUT_PATH}  ({size_kb:.0f} KB)")
print("   Placez ce fichier à la racine de votre repo avant docker compose up.")
