"""
Monitoring MLOps — Détection de dérive avec Evidently
======================================================
Usage:
  python monitor.py --reference data/X_test_app_sample.csv
                    --current   data/new_data.csv
                    --output    reports/
                    --threshold 0.10

Ce script:
  1. Charge les données de référence (distribution entraînement) et les données courantes
  2. Calcule des métriques de dérive pour chaque feature (PSI, Jensen-Shannon, etc.)
  3. Génère un rapport HTML Evidently interactif
  4. Logue les métriques dans MLflow
  5. Déclenche une alerte si la dérive dépasse le seuil configuré
  6. Retourne exit code 1 si réentraînement requis (pour intégration CI/CD)
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import mlflow

logger = logging.getLogger("fraud-monitor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
DRIFT_EXPERIMENT    = "fraud-detection-monitoring"
ALERT_WEBHOOK_URL   = os.getenv("ALERT_WEBHOOK_URL", "")

# Seuils par défaut
DEFAULT_DRIFT_THRESHOLD  = 0.10   # PSI > 0.10 = dérive modérée
CRITICAL_DRIFT_THRESHOLD = 0.25   # PSI > 0.25 = dérive critique → réentraîner


# ---------------------------------------------------------------------------
# Calcul PSI (Population Stability Index)
# ---------------------------------------------------------------------------
def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    PSI < 0.10 : stable
    PSI 0.10–0.25 : dérive modérée (surveiller)
    PSI > 0.25 : dérive critique (réentraîner)
    """
    def _scale(arr):
        mn, mx = np.min(arr), np.max(arr)
        return (arr - mn) / (mx - mn + 1e-10)

    expected_scaled = _scale(np.array(expected, dtype=float))
    actual_scaled   = _scale(np.array(actual, dtype=float))

    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_percents = np.histogram(expected_scaled, breakpoints)[0] / len(expected_scaled)
    actual_percents   = np.histogram(actual_scaled,   breakpoints)[0] / len(actual_scaled)

    # Éviter log(0)
    expected_percents = np.where(expected_percents == 0, 1e-6, expected_percents)
    actual_percents   = np.where(actual_percents   == 0, 1e-6, actual_percents)

    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return float(psi)


# ---------------------------------------------------------------------------
# Rapport Evidently
# ---------------------------------------------------------------------------
def generate_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Génère un rapport Evidently HTML + extrait les métriques de dérive."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        report.run(reference_data=reference, current_data=current)

        ts    = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fpath = output_dir / f"drift_report_{ts}.html"
        report.save_html(str(fpath))
        logger.info(f"Rapport Evidently sauvegardé: {fpath}")

        result = report.as_dict()
        drift_metrics = _extract_drift_metrics(result)
        drift_metrics["report_path"] = str(fpath)
        return drift_metrics

    except ImportError:
        logger.warning("Evidently non installé — calcul PSI manuel uniquement.")
        return {}


def _extract_drift_metrics(evidently_result: Dict) -> Dict[str, Any]:
    """Extrait les métriques clés du résultat Evidently."""
    metrics = {}
    try:
        for metric in evidently_result.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                result = metric.get("result", {})
                metrics["dataset_drift"] = result.get("dataset_drift", False)
                metrics["drift_share"]   = result.get("drift_share", 0.0)
                metrics["n_drifted_features"] = result.get("number_of_drifted_columns", 0)
                metrics["n_features"]         = result.get("number_of_columns", 0)
    except Exception as e:
        logger.warning(f"Erreur extraction métriques Evidently: {e}")
    return metrics


# ---------------------------------------------------------------------------
# Monitoring principal
# ---------------------------------------------------------------------------
def run_monitoring(
    reference_path: str,
    current_path: str,
    output_dir: str = "reports",
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    log_to_mlflow: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline de monitoring complet.
    Retourne un dict avec les résultats et un flag `requires_retraining`.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Monitoring | référence: {reference_path} | courant: {current_path}")

    # Chargement des données
    reference = pd.read_csv(reference_path)
    current   = pd.read_csv(current_path)
    logger.info(f"Référence: {reference.shape} | Courant: {current.shape}")

    # Aligner les colonnes
    common_cols = [c for c in reference.columns if c in current.columns]
    reference   = reference[common_cols]
    current     = current[common_cols]

    # 1. Calcul PSI par feature
    psi_scores = {}
    numeric_cols = reference.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        try:
            psi = compute_psi(reference[col].dropna().values, current[col].dropna().values)
            psi_scores[col] = round(psi, 4)
        except Exception as e:
            logger.debug(f"PSI {col}: {e}")

    drifted_features = [f for f, psi in psi_scores.items() if psi > drift_threshold]
    max_psi          = max(psi_scores.values()) if psi_scores else 0.0
    avg_psi          = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0

    # 2. Rapport Evidently
    evidently_metrics = generate_evidently_report(
        reference, current, Path(output_dir)
    )

    # 3. Décision de réentraînement
    requires_retraining = (
        max_psi > CRITICAL_DRIFT_THRESHOLD
        or len(drifted_features) > len(numeric_cols) * 0.3  # >30% des features dérivent
    )

    results = {
        "timestamp":            datetime.utcnow().isoformat(),
        "reference_rows":       len(reference),
        "current_rows":         len(current),
        "n_features_monitored": len(psi_scores),
        "n_drifted_features":   len(drifted_features),
        "drifted_features":     drifted_features,
        "max_psi":              round(max_psi, 4),
        "avg_psi":              round(avg_psi, 4),
        "drift_threshold":      drift_threshold,
        "requires_retraining":  requires_retraining,
        "psi_scores":           dict(sorted(psi_scores.items(), key=lambda x: -x[1])[:20]),
        **evidently_metrics,
    }

    # 4. Sauvegarde JSON
    results_path = Path(output_dir) / f"monitoring_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Résultats sauvegardés: {results_path}")

    # 5. Log MLflow
    if log_to_mlflow:
        _log_to_mlflow(results)

    # 6. Rapport console
    _print_summary(results)

    # 7. Alerte si dérive critique
    if requires_retraining:
        _send_alert(results)

    return results


def _log_to_mlflow(results: Dict[str, Any]) -> None:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(DRIFT_EXPERIMENT)
        with mlflow.start_run(run_name=f"monitoring-{results['timestamp'][:10]}"):
            mlflow.log_metrics({
                "max_psi":              results["max_psi"],
                "avg_psi":              results["avg_psi"],
                "n_drifted_features":   results["n_drifted_features"],
                "drift_share":          results.get("drift_share", 0.0),
            })
            mlflow.log_params({
                "drift_threshold":      str(results["drift_threshold"]),
                "reference_rows":       str(results["reference_rows"]),
                "current_rows":         str(results["current_rows"]),
                "requires_retraining":  str(results["requires_retraining"]),
            })
            for feat, psi in results["psi_scores"].items():
                mlflow.log_metric(f"psi_{feat}", psi)
        logger.info("Métriques loguées dans MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging échoué: {e}")


def _print_summary(results: Dict[str, Any]) -> None:
    logger.info("=" * 60)
    logger.info("RÉSUMÉ MONITORING")
    logger.info(f"  Features surveillées : {results['n_features_monitored']}")
    logger.info(f"  Features dérivent    : {results['n_drifted_features']}")
    logger.info(f"  PSI max              : {results['max_psi']:.4f}")
    logger.info(f"  PSI moyen            : {results['avg_psi']:.4f}")
    if results["drifted_features"]:
        logger.info(f"  Features dérivantes  : {', '.join(results['drifted_features'][:5])}")
    logger.info(f"  Réentraînement requis: {'⚠️  OUI' if results['requires_retraining'] else '✅ NON'}")
    logger.info("=" * 60)


def _send_alert(results: Dict[str, Any]) -> None:
    """Envoie une alerte webhook (Slack/Teams/PagerDuty)."""
    if not ALERT_WEBHOOK_URL:
        logger.warning("ALERT_WEBHOOK_URL non configuré — alerte non envoyée.")
        return
    try:
        import requests
        payload = {
            "text": (
                f"🚨 *Dérive détectée — Fraud Detection Model*\n"
                f"PSI max: {results['max_psi']:.4f} (seuil: {CRITICAL_DRIFT_THRESHOLD})\n"
                f"Features dérivent: {results['n_drifted_features']}/{results['n_features_monitored']}\n"
                f"→ Réentraînement automatique déclenché."
            )
        }
        resp = requests.post(ALERT_WEBHOOK_URL, json=payload, timeout=10)
        logger.info(f"Alerte envoyée: HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"Envoi alerte échoué: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoring de dérive — Fraud Detection")
    parser.add_argument("--reference",  required=True, help="CSV de référence (distribution entraînement)")
    parser.add_argument("--current",    required=True, help="CSV courant (nouvelles données)")
    parser.add_argument("--output",     default="reports", help="Dossier de sortie (défaut: reports/)")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_DRIFT_THRESHOLD, help="Seuil PSI (défaut: 0.10)")
    parser.add_argument("--no-mlflow",  action="store_true", help="Désactiver le logging MLflow")
    args = parser.parse_args()

    results = run_monitoring(
        reference_path=args.reference,
        current_path=args.current,
        output_dir=args.output,
        drift_threshold=args.threshold,
        log_to_mlflow=not args.no_mlflow,
    )

    # Exit code 1 si réentraînement requis (pour CI/CD)
    sys.exit(1 if results["requires_retraining"] else 0)
