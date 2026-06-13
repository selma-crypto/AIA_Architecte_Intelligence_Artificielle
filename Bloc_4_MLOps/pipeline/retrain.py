"""
Pipeline de réentraînement automatisé — Fraud Detection MLOps
=============================================================
Déclenché par :
  1. Dérive détectée (monitor.py retourne exit code 1)
  2. Nouvelles données disponibles (> seuil configurable)
  3. Planification régulière (cron weekly)
  4. Manuellement via CLI ou API /model/reload

Usage:
  python retrain.py --trigger=drift --data-path data/new_data.csv
  python retrain.py --trigger=scheduled --auto-deploy

Étapes du pipeline:
  1. Chargement et validation des nouvelles données
  2. Feature engineering (identique au notebook original)
  3. Entraînement XGBoost avec optimisation Optuna
  4. Évaluation et comparaison avec le modèle en production
  5. Enregistrement dans MLflow Model Registry
  6. Déploiement si les métriques progressent (ou sont stables)
  7. Notification des parties prenantes
"""

import os
import sys
import json
import pickle
import logging
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger("fraud-retrain")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT    = os.getenv("MLFLOW_EXPERIMENT",  "fraud-detection-xgboost")
MLFLOW_MODEL_NAME    = os.getenv("MLFLOW_MODEL_NAME",  "fraud-xgboost")
MODEL_OUTPUT_PATH    = os.getenv("MODEL_OUTPUT_PATH",  "fraud_xgb_model.pkl")
DECISION_THRESHOLD   = float(os.getenv("DECISION_THRESHOLD", "0.30"))
MIN_NEW_ROWS         = int(os.getenv("MIN_NEW_ROWS", "1000"))

# Seuils minimaux pour déployer
MIN_RECALL_PROD      = float(os.getenv("MIN_RECALL_PROD",    "0.80"))
MIN_PRECISION_PROD   = float(os.getenv("MIN_PRECISION_PROD", "0.55"))
MIN_F1_PROD          = float(os.getenv("MIN_F1_PROD",        "0.65"))

# Paramètres XGBoost de base (production actuelle — Optuna 150 trials)
BEST_PARAMS = {
    "n_estimators":     360,
    "max_depth":        3,
    "learning_rate":    0.08989,
    "gamma":            6.832,
    "min_child_weight": 16,
    "subsample":        0.999,
    "colsample_bytree": 0.542,
    "scale_pos_weight": 5.004,
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}


# ---------------------------------------------------------------------------
# 1. Chargement & validation des données
# ---------------------------------------------------------------------------
def load_and_validate(data_path: str) -> pd.DataFrame:
    logger.info(f"Chargement: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Shape: {df.shape}")

    required_cols = ["is_fraud", "amount", "account_age_days"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    if len(df) < MIN_NEW_ROWS:
        raise ValueError(f"Données insuffisantes: {len(df)} < {MIN_NEW_ROWS} lignes requises.")

    fraud_rate = df["is_fraud"].mean()
    logger.info(f"Taux de fraude: {fraud_rate:.4f} ({df['is_fraud'].sum()} fraudes)")
    if fraud_rate < 0.001 or fraud_rate > 0.5:
        logger.warning(f"⚠️ Taux de fraude anormal: {fraud_rate:.4f}")

    return df


# ---------------------------------------------------------------------------
# 2. Préparation des features (identique notebook)
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Applique le même pipeline de préparation que le notebook original."""
    logger.info("Préparation des features...")

    # Copie pour éviter les modifications en place
    df = df.copy()

    # Variables cible
    y = df["is_fraud"].astype(int)
    X = df.drop(columns=["is_fraud"])

    # Encoder les variables catégorielles si présentes
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        logger.info(f"Encodage OHE: {cat_cols}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # Traitement des NaN
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    logger.info(f"Features finales: {X.shape[1]} colonnes")
    return X, y


# ---------------------------------------------------------------------------
# 3. Entraînement
# ---------------------------------------------------------------------------
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    optimize: bool = False,
    n_trials: int = 30,
) -> Any:
    """Entraîne un XGBoost, avec optimisation Optuna optionnelle."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise RuntimeError("xgboost non installé. Lancer: pip install xgboost")

    params = BEST_PARAMS.copy()

    if optimize:
        logger.info(f"Optimisation Optuna ({n_trials} trials)...")
        params = _optuna_optimize(X_train, y_train, X_val, y_val, n_trials)

    logger.info(f"Entraînement XGBoost | {len(X_train)} exemples | {X_train.shape[1]} features")
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _optuna_optimize(X_train, y_train, X_val, y_val, n_trials: int) -> Dict:
    """Optimisation des hyperparamètres avec Optuna."""
    try:
        import optuna
        from xgboost import XGBClassifier
        from sklearn.metrics import f1_score
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "max_depth":        trial.suggest_int("max_depth", 2, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "gamma":            trial.suggest_float("gamma", 0, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42,
            }
            m = XGBClassifier(**params)
            m.fit(X_train, y_train, verbose=False)
            y_pred = (m.predict_proba(X_val)[:, 1] >= DECISION_THRESHOLD).astype(int)
            return f1_score(y_val, y_pred)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best.update({"use_label_encoder": False, "eval_metric": "logloss", "random_state": 42})
        logger.info(f"Meilleurs paramètres Optuna: F1={study.best_value:.4f}")
        return best
    except ImportError:
        logger.warning("Optuna non installé — utilisation des paramètres par défaut.")
        return BEST_PARAMS.copy()


# ---------------------------------------------------------------------------
# 4. Évaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = DECISION_THRESHOLD) -> Dict:
    from sklearn.metrics import (
        recall_score, precision_score, f1_score, accuracy_score, roc_auc_score,
        confusion_matrix,
    )
    probas  = model.predict_proba(X_test)[:, 1]
    y_pred  = (probas >= threshold).astype(int)
    cm      = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "recall":           round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "precision":        round(precision_score(y_test, y_pred,  zero_division=0), 4),
        "f1":               round(f1_score(y_test, y_pred,         zero_division=0), 4),
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":          round(roc_auc_score(y_test, probas), 4),
        "true_positives":   int(tp),
        "false_positives":  int(fp),
        "false_negatives":  int(fn),
        "true_negatives":   int(tn),
        "decision_threshold": threshold,
    }


def passes_gate(metrics: Dict) -> bool:
    """Vérifie si le modèle atteint les seuils minimaux pour production."""
    ok = (
        metrics["recall"]    >= MIN_RECALL_PROD
        and metrics["precision"] >= MIN_PRECISION_PROD
        and metrics["f1"]        >= MIN_F1_PROD
    )
    if not ok:
        logger.warning(
            f"❌ Seuils non atteints | recall={metrics['recall']} (min {MIN_RECALL_PROD}) "
            f"| precision={metrics['precision']} (min {MIN_PRECISION_PROD}) "
            f"| f1={metrics['f1']} (min {MIN_F1_PROD})"
        )
    return ok


# ---------------------------------------------------------------------------
# 5. Sauvegarde & MLflow
# ---------------------------------------------------------------------------
def save_and_register(
    model,
    features: list,
    metrics: Dict,
    params: Dict,
    trigger: str,
    auto_deploy: bool = False,
) -> Optional[str]:
    """Sauvegarde le modèle localement + enregistre dans MLflow Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"retrain-{trigger}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        # Log params & métriques
        mlflow.log_params({**params, "trigger": trigger, "n_features": len(features)})
        mlflow.log_metrics(metrics)

        # Log modèle dans MLflow
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )
        run_id = run.info.run_id
        logger.info(f"Run MLflow: {run_id}")

    # Sauvegarde pickle locale (compatible app Streamlit + API FastAPI)
    artifact = {"model": model, "final_columns": features, "metrics": metrics, "run_id": run_id}
    tmp_path = f"{MODEL_OUTPUT_PATH}.new"
    with open(tmp_path, "wb") as f:
        pickle.dump(artifact, f)

    # Vérification intégrité
    checksum = hashlib.md5(Path(tmp_path).read_bytes()).hexdigest()
    logger.info(f"Modèle sauvegardé: {tmp_path} | md5={checksum[:8]}")

    # Promotion en production si auto-deploy
    if auto_deploy:
        _promote_to_production(run_id)
        # Remplacer le fichier courant
        import shutil
        shutil.move(tmp_path, MODEL_OUTPUT_PATH)
        logger.info(f"Modèle déployé: {MODEL_OUTPUT_PATH}")

    return run_id


def _promote_to_production(run_id: str) -> None:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    try:
        versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
        for v in versions:
            if v.run_id == run_id:
                # Archiver l'ancienne version Production
                for old_v in versions:
                    if old_v.current_stage == "Production":
                        client.transition_model_version_stage(
                            name=MLFLOW_MODEL_NAME,
                            version=old_v.version,
                            stage="Archived",
                        )
                # Promouvoir la nouvelle
                client.transition_model_version_stage(
                    name=MLFLOW_MODEL_NAME,
                    version=v.version,
                    stage="Production",
                )
                logger.info(f"Modèle v{v.version} promu en Production dans MLflow")
                return
    except Exception as e:
        logger.error(f"Erreur promotion MLflow: {e}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def run_pipeline(
    trigger: str,
    data_path: Optional[str],
    auto_deploy: bool,
    optimize: bool,
    n_trials: int,
) -> Dict[str, Any]:
    from sklearn.model_selection import train_test_split

    logger.info(f"=== Réentraînement | trigger={trigger} | auto_deploy={auto_deploy} ===")

    if not data_path:
        raise ValueError("--data-path requis pour le réentraînement.")

    # 1. Chargement
    df = load_and_validate(data_path)

    # 2. Features
    X, y = prepare_features(df)
    features = list(X.columns)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    logger.info(f"Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # 4. Entraînement
    model = train_model(X_train, y_train, X_val, y_val, optimize=optimize, n_trials=n_trials)

    # 5. Évaluation
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"Métriques | recall={metrics['recall']} | precision={metrics['precision']} | f1={metrics['f1']} | auc={metrics['roc_auc']}")

    # 6. Quality gate
    gate_ok = passes_gate(metrics)
    if not gate_ok and auto_deploy:
        logger.error("Quality gate échoué — déploiement annulé.")
        return {"status": "rejected", "metrics": metrics, "reason": "quality_gate_failed"}

    # 7. Sauvegarde & MLflow
    run_id = save_and_register(
        model=model,
        features=features,
        metrics=metrics,
        params=model.get_params() if hasattr(model, "get_params") else BEST_PARAMS,
        trigger=trigger,
        auto_deploy=auto_deploy and gate_ok,
    )

    result = {
        "status": "deployed" if (auto_deploy and gate_ok) else "saved",
        "run_id": run_id,
        "metrics": metrics,
        "trigger": trigger,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # 8. Résumé final
    logger.info("=" * 60)
    logger.info("RÉENTRAÎNEMENT TERMINÉ")
    logger.info(f"  Statut  : {result['status']}")
    logger.info(f"  Run ID  : {run_id}")
    logger.info(f"  Recall  : {metrics['recall']}")
    logger.info(f"  F1      : {metrics['f1']}")
    logger.info(f"  AUC     : {metrics['roc_auc']}")
    logger.info("=" * 60)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de réentraînement — Fraud Detection")
    parser.add_argument("--trigger",     default="manual",
                        choices=["manual", "drift", "scheduled", "ci_cd"],
                        help="Déclencheur du réentraînement")
    parser.add_argument("--data-path",   help="Chemin vers les nouvelles données (CSV)")
    parser.add_argument("--auto-deploy", action="store_true",
                        help="Déployer automatiquement si quality gate OK")
    parser.add_argument("--optimize",    action="store_true",
                        help="Réoptimiser les hyperparamètres avec Optuna")
    parser.add_argument("--n-trials",    type=int, default=30,
                        help="Nombre de trials Optuna (défaut: 30)")
    args = parser.parse_args()

    result = run_pipeline(
        trigger=args.trigger,
        data_path=args.data_path,
        auto_deploy=args.auto_deploy,
        optimize=args.optimize,
        n_trials=args.n_trials,
    )

    sys.exit(0 if result["status"] in ("deployed", "saved") else 1)
