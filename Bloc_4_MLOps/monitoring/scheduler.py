"""
Scheduler de monitoring continu — Fraud Detection
==================================================
Daemon léger qui lance monitor.py à intervalles réguliers.
Alternative à Airflow pour les environnements simples (pas de K8s/Airflow).

Modes de déploiement :
  1. Docker service  → `docker compose up -d monitor-scheduler`
  2. Systemd service → voir monitoring/fraud-monitor.service
  3. Cron Linux      → voir monitoring/crontab.txt
  4. Script direct   → `python monitoring/scheduler.py`

Comportement :
  - Toutes les N heures (configurable via MONITOR_INTERVAL_HOURS)
  - Lance monitor.py avec les chemins configurés
  - Si dérive détectée → déclenche retrain.py en arrière-plan
  - Logue chaque run dans logs/monitoring.log
  - Alerte webhook si échec consécutifs > MAX_FAILURES
"""

import os
import sys
import time
import logging
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MONITOR_INTERVAL_HOURS = int(os.getenv("MONITOR_INTERVAL_HOURS", "6"))    # Toutes les 6h
REFERENCE_DATA_PATH    = os.getenv("REFERENCE_DATA_PATH", "data/X_test_app_sample.csv")
NEW_DATA_PATH          = os.getenv("NEW_DATA_PATH",        "data/new_data.csv")
REPORTS_DIR            = os.getenv("REPORTS_DIR",          "reports")
DRIFT_THRESHOLD        = float(os.getenv("DRIFT_THRESHOLD", "0.10"))
AUTO_RETRAIN           = os.getenv("AUTO_RETRAIN", "true").lower() == "true"
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI",  "sqlite:///mlflow.db")
ALERT_WEBHOOK_URL      = os.getenv("ALERT_WEBHOOK_URL",    "")
MAX_FAILURES           = int(os.getenv("MAX_FAILURES",     "3"))
PROJECT_ROOT           = os.getenv("PROJECT_ROOT",          os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/monitoring.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fraud-monitor-scheduler")

# ---------------------------------------------------------------------------
# État du scheduler
# ---------------------------------------------------------------------------
_state = {
    "runs_total":       0,
    "runs_success":     0,
    "runs_failed":      0,
    "consecutive_fail": 0,
    "last_run":         None,
    "last_status":      None,
    "retrains_triggered": 0,
}


# ---------------------------------------------------------------------------
# Exécution du monitoring
# ---------------------------------------------------------------------------
def run_monitoring_cycle() -> dict:
    """Lance un cycle complet de monitoring et retourne le résultat."""
    logger.info(f"=== Cycle de monitoring #{_state['runs_total'] + 1} ===")

    try:
        # Import direct du module de monitoring
        sys.path.insert(0, PROJECT_ROOT)
        from monitoring.monitor import run_monitoring

        results = run_monitoring(
            reference_path=REFERENCE_DATA_PATH,
            current_path=NEW_DATA_PATH if Path(NEW_DATA_PATH).exists() else REFERENCE_DATA_PATH,
            output_dir=REPORTS_DIR,
            drift_threshold=DRIFT_THRESHOLD,
            log_to_mlflow=True,
        )

        _state["runs_total"]       += 1
        _state["runs_success"]     += 1
        _state["consecutive_fail"]  = 0
        _state["last_run"]          = datetime.utcnow().isoformat()
        _state["last_status"]       = "success"

        logger.info(
            f"Monitoring OK | psi_max={results['max_psi']:.4f} | "
            f"drifted={results['n_drifted_features']} | "
            f"retrain={results['requires_retraining']}"
        )

        # Déclencher le réentraînement si nécessaire
        if results["requires_retraining"] and AUTO_RETRAIN:
            _trigger_retrain(results)

        return results

    except Exception as e:
        _state["runs_total"]       += 1
        _state["runs_failed"]      += 1
        _state["consecutive_fail"] += 1
        _state["last_run"]          = datetime.utcnow().isoformat()
        _state["last_status"]       = f"error: {e}"

        logger.error(f"Erreur monitoring: {e}", exc_info=True)

        if _state["consecutive_fail"] >= MAX_FAILURES:
            _send_critical_alert(e)

        return {"error": str(e)}


def _trigger_retrain(drift_results: dict) -> None:
    """Lance retrain.py en arrière-plan (thread séparé pour ne pas bloquer le scheduler)."""
    def _run():
        logger.info("Déclenchement du réentraînement automatique (drift détecté)...")
        _state["retrains_triggered"] += 1
        try:
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "pipeline/retrain.py"),
                "--trigger", "drift",
                "--auto-deploy",
            ]
            if Path(NEW_DATA_PATH).exists():
                cmd += ["--data-path", NEW_DATA_PATH]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                logger.info(f"Réentraînement terminé avec succès\n{result.stdout[-500:]}")
            else:
                logger.error(f"Réentraînement échoué (code {result.returncode})\n{result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            logger.error("Réentraînement timeout (>1h)")
        except Exception as e:
            logger.error(f"Erreur lors du réentraînement: {e}", exc_info=True)

    t = threading.Thread(target=_run, daemon=True, name="retrain-worker")
    t.start()
    logger.info("Thread de réentraînement démarré en arrière-plan.")


def _send_critical_alert(error: Exception) -> None:
    """Alerte critique si le monitoring échoue plusieurs fois de suite."""
    if not ALERT_WEBHOOK_URL:
        return
    try:
        import requests
        requests.post(ALERT_WEBHOOK_URL, json={
            "text": (
                f"🔴 *ALERTE CRITIQUE — Monitoring Fraud Detection*\n"
                f"{_state['consecutive_fail']} échecs consécutifs\n"
                f"Dernière erreur: `{error}`"
            )
        }, timeout=10)
    except Exception as e:
        logger.warning(f"Envoi alerte critique échoué: {e}")


def _print_status() -> None:
    """Affiche l'état courant du scheduler."""
    logger.info(
        f"Status | runs={_state['runs_total']} | "
        f"ok={_state['runs_success']} | fail={_state['runs_failed']} | "
        f"retrains={_state['retrains_triggered']} | "
        f"next_run_in={MONITOR_INTERVAL_HOURS}h"
    )


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------
def run_scheduler() -> None:
    """Boucle infinie du scheduler de monitoring."""
    interval_seconds = MONITOR_INTERVAL_HOURS * 3600
    logger.info(f"Scheduler démarré | intervalle={MONITOR_INTERVAL_HOURS}h | auto_retrain={AUTO_RETRAIN}")
    logger.info(f"Référence: {REFERENCE_DATA_PATH} | Courant: {NEW_DATA_PATH}")

    # Premier run immédiatement au démarrage
    run_monitoring_cycle()
    _print_status()

    while True:
        next_run = datetime.utcnow() + timedelta(seconds=interval_seconds)
        logger.info(f"Prochain cycle à {next_run.strftime('%Y-%m-%d %H:%M:%S')} UTC")

        # Attente avec interruption possible toutes les 60s (pour arrêt propre)
        elapsed = 0
        while elapsed < interval_seconds:
            time.sleep(min(60, interval_seconds - elapsed))
            elapsed += 60

        run_monitoring_cycle()
        _print_status()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scheduler de monitoring continu")
    parser.add_argument("--run-once", action="store_true",
                        help="Lancer un seul cycle et quitter (utile pour les cron jobs)")
    args = parser.parse_args()

    if args.run_once:
        results = run_monitoring_cycle()
        sys.exit(0 if "error" not in results else 1)
    else:
        run_scheduler()
