"""
Script de validation des seuils de performance du modèle.
Utilisé dans le CI/CD GitHub Actions pour bloquer le déploiement si les
métriques sont sous les seuils configurés.
"""
import os
import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score

logger = logging.getLogger("validate-model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MIN_RECALL    = float(os.getenv("MIN_RECALL",    "0.80"))
MIN_PRECISION = float(os.getenv("MIN_PRECISION", "0.55"))
MIN_F1        = float(os.getenv("MIN_F1",        "0.65"))
MODEL_PATH    = os.getenv("MODEL_PATH", "fraud_xgb_model.pkl")
TEST_X_PATH   = os.getenv("TEST_X_PATH", "data/X_test_app_sample.csv")
TEST_Y_PATH   = os.getenv("TEST_Y_PATH", "data/y_test_app_sample.csv")
THRESHOLD     = float(os.getenv("DECISION_THRESHOLD", "0.30"))


def main():
    if not Path(MODEL_PATH).exists():
        logger.warning(f"Modèle absent ({MODEL_PATH}) — validation ignorée (OK en CI sans artefact).")
        sys.exit(0)
    if not Path(TEST_X_PATH).exists() or not Path(TEST_Y_PATH).exists():
        logger.warning("Données de test absentes — validation ignorée.")
        sys.exit(0)

    logger.info(f"Validation du modèle: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    model    = data["model"] if isinstance(data, dict) and "model" in data else data
    features = data.get("final_columns") if isinstance(data, dict) else None

    X = pd.read_csv(TEST_X_PATH)
    y = pd.read_csv(TEST_Y_PATH).iloc[:, -1]

    if features:
        X = X.reindex(columns=features, fill_value=0)

    probas = model.predict_proba(X)[:, 1]
    y_pred = (probas >= THRESHOLD).astype(int)

    recall    = recall_score(y, y_pred,    zero_division=0)
    precision = precision_score(y, y_pred, zero_division=0)
    f1        = f1_score(y, y_pred,        zero_division=0)

    logger.info(f"Recall    : {recall:.4f}  (min {MIN_RECALL})")
    logger.info(f"Precision : {precision:.4f}  (min {MIN_PRECISION})")
    logger.info(f"F1        : {f1:.4f}  (min {MIN_F1})")

    failed = []
    if recall    < MIN_RECALL:    failed.append(f"recall={recall:.4f} < {MIN_RECALL}")
    if precision < MIN_PRECISION: failed.append(f"precision={precision:.4f} < {MIN_PRECISION}")
    if f1        < MIN_F1:        failed.append(f"f1={f1:.4f} < {MIN_F1}")

    if failed:
        logger.error(f"❌ Quality gate échoué: {', '.join(failed)}")
        sys.exit(1)
    else:
        logger.info("✅ Quality gate OK — modèle validé pour déploiement.")
        sys.exit(0)


if __name__ == "__main__":
    main()
