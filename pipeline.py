"""
pipeline.py — Orchestrateur principal du pipeline temps réel.

Boucle toutes les POLL_INTERVAL_S secondes :
  1. Appelle l'API → nouvelles transactions
  2. Pour chaque transaction :
     a. Insère dans PostgreSQL
     b. Lance la prédiction ML
     c. Stocke la prédiction
     d. Si fraude → envoie une alerte email
  3. À 08h00 chaque matin → envoie le rapport quotidien

Lancement :
    python pipeline.py

Arrêt propre : Ctrl+C
"""

import logging
import time
from datetime import datetime

import schedule

import alerts
import api_client
import config
import daily_report
import db
import model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ── Tâche principale : traiter les nouvelles transactions ────────────────────

def process_new_transactions() -> None:
    """Récupère, prédit et stocke toutes les nouvelles transactions."""
    try:
        for tx in api_client.get_new_transactions():
            trans_num = tx["trans_num"]

            # 1) Stockage de la transaction
            is_new = db.insert_transaction(tx)
            if not is_new:
                continue  # déjà traitée (ne devrait pas arriver ici)

            # 2) Prédiction
            fraud_proba, is_fraud = model.predict(tx)
            db.insert_prediction(trans_num, fraud_proba, is_fraud)

            if is_fraud:
                logger.warning(
                    "FRAUDE | trans=%s | amt=$%.2f | proba=%.1f%%",
                    trans_num, tx["amt"], fraud_proba * 100,
                )
                # 3) Alerte email
                sent = alerts.send_fraud_alert(tx, fraud_proba)
                if sent:
                    db.mark_alert_sent(trans_num)
            else:
                logger.debug("OK    | trans=%s | amt=$%.2f | proba=%.1f%%",
                             trans_num, tx["amt"], fraud_proba * 100)

    except Exception as e:
        logger.error("Erreur lors du traitement des transactions : %s", e)


# ── Planification ────────────────────────────────────────────────────────────

def _schedule_daily_report() -> None:
    report_time = f"{config.DAILY_REPORT_HOUR:02d}:{config.DAILY_REPORT_MINUTE:02d}"
    schedule.every().day.at(report_time).do(daily_report.send_daily_report)
    logger.info("Rapport quotidien planifié à %s.", report_time)


def main() -> None:
    logger.info("=== Démarrage du pipeline de détection de fraude ===")

    # Initialisation
    db.init_pool()
    _schedule_daily_report()

    # Planification du polling toutes les N secondes
    schedule.every(config.POLL_INTERVAL_S).seconds.do(process_new_transactions)
    logger.info("Polling API toutes les %ds.", config.POLL_INTERVAL_S)

    # Premier appel immédiat
    process_new_transactions()

    # Boucle principale
    logger.info("Pipeline actif. Ctrl+C pour arrêter.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Arrêt du pipeline.")


if __name__ == "__main__":
    main()
