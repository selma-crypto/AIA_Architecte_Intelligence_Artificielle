"""
train.py — Entraînement du modèle de détection de fraude.

Usage :
    python train.py --data fraudTest.csv --output model/fraud_model.pkl

Le pipeline sklearn inclut :
  1. Feature engineering (âge, heure, distance géographique)
  2. Encodage des variables catégorielles
  3. Standardisation
  4. Random Forest avec class_weight='balanced' (gère le déséquilibre)

Le fichier .pkl produit contient le pipeline COMPLET (préprocesseur + modèle),
ce qui garantit que les mêmes transformations sont appliquées en production.
"""

import argparse
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Feature engineering ──────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features dérivées pertinentes pour la fraude."""
    df = df.copy()

    # Heure de la transaction (les fraudes sont plus fréquentes la nuit)
    df["trans_datetime"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_datetime"].dt.hour

    # Âge du porteur de carte
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_datetime"] - df["dob"]).dt.days // 365

    # Distance entre le commerçant et le lieu de résidence (approximation euclidienne)
    df["distance_km"] = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 +
        (df["long"] - df["merch_long"]) ** 2
    ) * 111  # 1° ≈ 111 km

    return df


NUMERIC_FEATURES  = ["amt", "city_pop", "hour", "age", "distance_km", "lat", "long"]
CATEGORIC_FEATURES = ["category", "gender", "state"]

TARGET = "is_fraud"


# ── Pipeline sklearn ─────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORIC_FEATURES),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",   # compense le déséquilibre fraude/légitime
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])


# ── Entraînement ─────────────────────────────────────────────────────────────

def train(data_path: str, output_path: str) -> None:
    logger.info("Chargement des données : %s", data_path)
    df = pd.read_csv(data_path)

    df = add_features(df)

    X = df[NUMERIC_FEATURES + CATEGORIC_FEATURES]
    y = df[TARGET]

    logger.info("Distribution de la cible : %s", y.value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline()

    logger.info("Entraînement en cours...")
    pipeline.fit(X_train, y_train)

    # Évaluation
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    logger.info("\n%s", classification_report(y_test, y_pred))
    logger.info("ROC-AUC : %.4f", roc_auc_score(y_test, y_proba))

    # Sauvegarde du pipeline complet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info("Modèle sauvegardé : %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne le modèle de détection de fraude.")
    parser.add_argument("--data",   default="fraudTest.csv",        help="Chemin vers le CSV d'entraînement")
    parser.add_argument("--output", default="model/fraud_model.pkl", help="Chemin de sortie du modèle")
    args = parser.parse_args()

    train(args.data, args.output)
