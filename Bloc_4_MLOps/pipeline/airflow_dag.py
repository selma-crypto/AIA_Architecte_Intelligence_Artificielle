"""
DAG Apache Airflow — Pipeline MLOps Fraud Detection
====================================================
Ce DAG orchestre le cycle de vie complet du modèle de détection de fraude :

  1. check_new_data        → Vérifie si de nouvelles données sont disponibles
  2. run_monitoring        → Calcule la dérive PSI avec Evidently
  3. decide_retrain        → BranchPythonOperator : retrain si dérive ou nouvelles données
  4. retrain_model         → Réentraîne XGBoost + Optuna + quality gate
  5. validate_model        → Valide les métriques (recall ≥ 0.80, F1 ≥ 0.65)
  6. register_model        → Enregistre dans MLflow Model Registry
  7. deploy_model          → Déploie via API /model/reload (zero-downtime)
  8. notify_success        → Notification Slack/email
  9. skip_retrain          → Branche alternative si pas de dérive

Planification :
  - Quotidien à 02:00 UTC (données fraîches disponibles chaque nuit)
  - Déclenchable manuellement via Airflow UI ou API

Prérequis :
  pip install apache-airflow apache-airflow-providers-http apache-airflow-providers-slack

Variables Airflow à configurer (Admin → Variables) :
  MLFLOW_TRACKING_URI     URI du serveur MLflow
  API_BASE_URL            URL de l'API FastAPI (ex: http://fraud-api:8000)
  REFERENCE_DATA_PATH     Chemin CSV de référence
  NEW_DATA_PATH           Chemin CSV nouvelles données
  ALERT_WEBHOOK_URL       Webhook Slack/Teams

Connexions Airflow à configurer (Admin → Connections) :
  fraud_api_http          HTTP connection vers l'API FastAPI
  slack_default           Connexion Slack (pour notifications)
"""

from __future__ import annotations

import os
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration depuis les Variables Airflow
# ---------------------------------------------------------------------------
def _var(key: str, default: str = "") -> str:
    try:
        return Variable.get(key)
    except Exception:
        return os.getenv(key, default)


MLFLOW_TRACKING_URI  = _var("MLFLOW_TRACKING_URI",  "sqlite:///mlflow.db")
API_BASE_URL         = _var("API_BASE_URL",          "http://fraud-api:8000")
REFERENCE_DATA_PATH  = _var("REFERENCE_DATA_PATH",   "data/X_test_app_sample.csv")
NEW_DATA_PATH        = _var("NEW_DATA_PATH",          "data/new_data.csv")
ALERT_WEBHOOK_URL    = _var("ALERT_WEBHOOK_URL",      "")
DRIFT_THRESHOLD      = float(_var("DRIFT_THRESHOLD", "0.10"))
MIN_NEW_ROWS         = int(_var("MIN_NEW_ROWS",       "1000"))
PROJECT_ROOT         = _var("PROJECT_ROOT",           "/opt/fraud-detection")

# ---------------------------------------------------------------------------
# Arguments par défaut du DAG
# ---------------------------------------------------------------------------
default_args = {
    "owner":            "mlops-team",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
with DAG(
    dag_id="fraud_detection_mlops_pipeline",
    description="Pipeline MLOps complet : monitoring → réentraînement → déploiement",
    schedule_interval="0 2 * * *",   # Tous les jours à 02:00 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "fraud-detection", "xgboost"],
    doc_md="""
## Pipeline MLOps — Fraud Detection

Ce DAG gère le cycle de vie automatisé du modèle XGBoost de détection de fraude.

### Flux principal
```
check_new_data → run_monitoring → decide_retrain
                                        ├── [drift] → retrain_model → validate_model
                                        │                                    ├── [ok]   → register_model → deploy_model → notify_success
                                        │                                    └── [fail] → notify_failure
                                        └── [no_drift] → skip_retrain → notify_success
```

### Déclenchement manuel
```bash
airflow dags trigger fraud_detection_mlops_pipeline
```

### Paramètres configurables (Airflow Variables)
- `DRIFT_THRESHOLD` : seuil PSI pour déclencher le réentraînement (défaut: 0.10)
- `MIN_NEW_ROWS` : minimum de nouvelles lignes pour lancer un retrain (défaut: 1000)
""",
) as dag:

    # ────────────────────────────────────────────────────────────────────────
    # TASK 1 : Vérification des nouvelles données
    # ────────────────────────────────────────────────────────────────────────
    def check_new_data_fn(**context) -> dict:
        """Vérifie si de nouvelles données sont disponibles et suffisantes."""
        new_data = Path(NEW_DATA_PATH)
        if not new_data.exists():
            logger.info(f"Pas de nouvelles données: {new_data}")
            context["ti"].xcom_push(key="has_new_data", value=False)
            context["ti"].xcom_push(key="new_rows", value=0)
            return {"has_new_data": False, "new_rows": 0}

        import pandas as pd
        df = pd.read_csv(new_data)
        n  = len(df)
        ok = n >= MIN_NEW_ROWS
        logger.info(f"Nouvelles données: {n} lignes (min: {MIN_NEW_ROWS}) → {'OK' if ok else 'insuffisant'}")
        context["ti"].xcom_push(key="has_new_data", value=ok)
        context["ti"].xcom_push(key="new_rows",     value=n)
        return {"has_new_data": ok, "new_rows": n}

    check_new_data = PythonOperator(
        task_id="check_new_data",
        python_callable=check_new_data_fn,
    )

    # ────────────────────────────────────────────────────────────────────────
    # TASK 2 : Monitoring de dérive (Evidently + PSI)
    # ────────────────────────────────────────────────────────────────────────
    def run_monitoring_fn(**context) -> dict:
        """Lance monitor.py et stocke les résultats dans XCom."""
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from monitoring.monitor import run_monitoring

        results = run_monitoring(
            reference_path=REFERENCE_DATA_PATH,
            current_path=NEW_DATA_PATH if Path(NEW_DATA_PATH).exists() else REFERENCE_DATA_PATH,
            output_dir=os.path.join(PROJECT_ROOT, "reports"),
            drift_threshold=DRIFT_THRESHOLD,
            log_to_mlflow=True,
        )

        context["ti"].xcom_push(key="drift_results",       value=results)
        context["ti"].xcom_push(key="requires_retraining", value=results["requires_retraining"])
        context["ti"].xcom_push(key="max_psi",             value=results["max_psi"])
        context["ti"].xcom_push(key="n_drifted",           value=results["n_drifted_features"])

        logger.info(
            f"Monitoring | max_psi={results['max_psi']:.4f} | "
            f"drifted={results['n_drifted_features']} features | "
            f"retrain={results['requires_retraining']}"
        )
        return results

    run_monitoring_task = PythonOperator(
        task_id="run_monitoring",
        python_callable=run_monitoring_fn,
    )

    # ────────────────────────────────────────────────────────────────────────
    # TASK 3 : Décision de réentraînement (BranchOperator)
    # ────────────────────────────────────────────────────────────────────────
    def decide_retrain_fn(**context) -> str:
        """
        Retourne l'ID de la tâche suivante selon la décision :
          - 'retrain_model'  si dérive détectée OU nouvelles données suffisantes
          - 'skip_retrain'   sinon
        """
        ti = context["ti"]
        requires_retraining = ti.xcom_pull(task_ids="run_monitoring",  key="requires_retraining")
        has_new_data        = ti.xcom_pull(task_ids="check_new_data",  key="has_new_data")
        max_psi             = ti.xcom_pull(task_ids="run_monitoring",  key="max_psi") or 0

        should_retrain = requires_retraining or has_new_data
        logger.info(
            f"Décision | requires_retraining={requires_retraining} | "
            f"has_new_data={has_new_data} | max_psi={max_psi:.4f} "
            f"→ {'RETRAIN' if should_retrain else 'SKIP'}"
        )
        return "retrain_model" if should_retrain else "skip_retrain"

    decide_retrain = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=decide_retrain_fn,
    )

    # ────────────────────────────────────────────────────────────────────────
    # TASK 4a : Skip (pas de dérive)
    # ────────────────────────────────────────────────────────────────────────
    skip_retrain = EmptyOperator(task_id="skip_retrain")

    # ────────────────────────────────────────────────────────────────────────
    # TASK 4b : Réentraînement XGBoost
    # ────────────────────────────────────────────────────────────────────────
    def retrain_model_fn(**context) -> dict:
        """Lance le pipeline de réentraînement complet."""
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from pipeline.retrain import run_pipeline

        ti          = context["ti"]
        has_new     = ti.xcom_pull(task_ids="check_new_data", key="has_new_data")
        trigger_src = "drift" if ti.xcom_pull(task_ids="run_monitoring", key="requires_retraining") else "new_data"

        result = run_pipeline(
            trigger=trigger_src,
            data_path=NEW_DATA_PATH if has_new else REFERENCE_DATA_PATH,
            auto_deploy=False,   # Le déploiement se fait dans deploy_model
            optimize=False,      # Optuna désactivé par défaut (trop lent pour daily)
            n_trials=30,
        )

        if result["status"] == "rejected":
            raise ValueError(f"Quality gate échoué: {result.get('reason', 'unknown')}")

        context["ti"].xcom_push(key="retrain_result", value=result)
        context["ti"].xcom_push(key="run_id",         value=result.get("run_id"))
        context["ti"].xcom_push(key="new_f1",         value=result["metrics"].get("f1", 0))
        logger.info(f"Réentraînement OK | run_id={result.get('run_id')} | f1={result['metrics'].get('f1')}")
        return result

    retrain_model = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model_fn,
        execution_timeout=timedelta(hours=1),
    )

    # ────────────────────────────────────────────────────────────────────────
    # TASK 5 : Validation post-entraînement
    # ────────────────────────────────────────────────────────────────────────
    def validate_model_fn(**context) -> str:
        """Valide que le nouveau modèle passe le quality gate avant déploiement."""
        import sys
        sys.path.insert(0, PROJECT_ROOT)

        result = subprocess.run(
            ["python", os.path.join(PROJECT_ROOT, "pipeline/validate_model.py")],
            capture_output=True, text=True,
            env={**os.environ, "MODEL_PATH": f"{PROJECT_ROOT}/fraud_xgb_model.pkl.new"},
        )
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error(result.stderr)
            raise ValueError(f"Validation échouée:\n{result.stdout}\n{result.stderr}")
        return "passed"

    validate_model = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model_fn,
    )

    # ────────────────────────────────────────────────────────────────────────
    # TASK 6 : Enregistrement dans MLflow Model Registry
    # ────────────────────────────────────────────────────────────────────────
    def register_model_fn(**context) -> None:
        """Promeut le modèle en Production dans le MLflow Model Registry."""
        import mlflow
        from mlflow.tracking import MlflowClient

        ti     = context["ti"]
        run_id = ti.xcom_pull(task_ids="retrain_model", key="run_id")
        if not run_id:
            raise ValueError("run_id manquant — réentraînement non terminé ?")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        # Archiver l'ancienne version Production
        for v in client.search_model_versions("name='fraud-xgboost'"):
            if v.current_stage == "Production":
                client.transition_model_version_stage(
                    name="fraud-xgboost", version=v.version, stage="Archived"
                )
                logger.info(f"Version {v.version} archivée")

        # Promouvoir la nouvelle version
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if versions:
            new_v = versions[0]
            client.transition_model_version_stage(
                name="fraud-xgboost", version=new_v.version, stage="Production"
            )
            logger.info(f"Version {new_v.version} promue en Production")
        else:
            raise ValueError(f"Aucune version trouvée pour run_id={run_id}")

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_model_fn,
    )

    # ────────────────────────────────────────────────────────────────────────
    # TASK 7 : Déploiement zero-downtime via API /model/reload
    # ────────────────────────────────────────────────────────────────────────
    def deploy_model_fn(**context) -> None:
        """
        Déploie le nouveau modèle sans redémarrage :
          1. Copie fraud_xgb_model.pkl.new → fraud_xgb_model.pkl
          2. Appelle POST /model/reload sur l'API FastAPI (hot-reload)
          3. Vérifie la santé avec GET /health
        """
        import shutil
        import requests
        import time

        new_path = Path(f"{PROJECT_ROOT}/fraud_xgb_model.pkl.new")
        cur_path = Path(f"{PROJECT_ROOT}/fraud_xgb_model.pkl")
        bak_path = Path(f"{PROJECT_ROOT}/fraud_xgb_model.bak.{int(time.time())}.pkl")

        if not new_path.exists():
            raise FileNotFoundError(f"Modèle entraîné introuvable: {new_path}")

        # Backup
        if cur_path.exists():
            shutil.copy2(cur_path, bak_path)
            logger.info(f"Backup: {bak_path}")

        # Remplacement atomique
        shutil.move(str(new_path), str(cur_path))
        logger.info(f"Modèle déployé: {cur_path}")

        # Hot-reload de l'API
        try:
            r = requests.post(f"{API_BASE_URL}/model/reload", timeout=10)
            logger.info(f"API reload: HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"Hot-reload API non disponible: {e} — redémarrage manuel requis")

        # Health check post-déploiement
        time.sleep(5)
        for attempt in range(5):
            try:
                r = requests.get(f"{API_BASE_URL}/health", timeout=10)
                if r.status_code == 200 and r.json().get("model_loaded"):
                    logger.info(f"✅ Health check OK après déploiement (tentative {attempt+1})")
                    return
            except Exception:
                pass
            time.sleep(10)

        raise RuntimeError("Health check échoué après déploiement — rollback recommandé")

    deploy_model = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model_fn,
    )

    # ────────────────────────────────────────────────────────────────────────
    # TASK 8 : Notifications
    # ────────────────────────────────────────────────────────────────────────
    def notify_success_fn(**context) -> None:
        """Envoie une notification de succès (Slack/webhook)."""
        if not ALERT_WEBHOOK_URL:
            logger.info("Pas de webhook configuré — notification ignorée.")
            return
        import requests
        ti = context["ti"]
        was_retrained = ti.xcom_pull(task_ids="retrain_model", key="new_f1") is not None
        f1            = ti.xcom_pull(task_ids="retrain_model", key="new_f1") or "N/A"
        max_psi       = ti.xcom_pull(task_ids="run_monitoring", key="max_psi") or 0

        if was_retrained:
            msg = f"✅ *Fraud Detection — Réentraînement & déploiement réussis*\nF1: {f1} | PSI max: {max_psi:.4f}"
        else:
            msg = f"✅ *Fraud Detection — Monitoring quotidien OK*\nPas de dérive détectée | PSI max: {max_psi:.4f}"

        try:
            requests.post(ALERT_WEBHOOK_URL, json={"text": msg}, timeout=10)
        except Exception as e:
            logger.warning(f"Notification échouée: {e}")

    def notify_failure_fn(**context) -> None:
        """Notification en cas d'échec."""
        if not ALERT_WEBHOOK_URL:
            return
        import requests
        exception = context.get("exception", "Erreur inconnue")
        try:
            requests.post(ALERT_WEBHOOK_URL, json={
                "text": f"🚨 *Fraud Detection — Pipeline ÉCHOUÉ*\n`{exception}`"
            }, timeout=10)
        except Exception as e:
            logger.warning(f"Notification d'échec échouée: {e}")

    notify_success = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success_fn,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    notify_failure = PythonOperator(
        task_id="notify_failure",
        python_callable=notify_failure_fn,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # ────────────────────────────────────────────────────────────────────────
    # Dépendances (graphe du DAG)
    # ────────────────────────────────────────────────────────────────────────
    #
    #  check_new_data ──┐
    #                   ├──► run_monitoring ──► decide_retrain ──► retrain_model ──► validate_model ──► register_model ──► deploy_model ──► notify_success
    #                                                           └──► skip_retrain ──────────────────────────────────────────────────────► notify_success
    #                                                                                                                   notify_failure (si erreur)
    #
    [check_new_data, run_monitoring_task] >> decide_retrain
    check_new_data >> run_monitoring_task  # monitoring après check

    decide_retrain >> retrain_model >> validate_model >> register_model >> deploy_model >> notify_success
    decide_retrain >> skip_retrain >> notify_success

    [deploy_model, validate_model, retrain_model] >> notify_failure
