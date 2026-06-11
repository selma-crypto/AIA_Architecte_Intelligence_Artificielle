"""
config.py — Configuration centrale du pipeline.
Toutes les valeurs sensibles sont lues depuis les variables d'environnement
(ou depuis un fichier .env via python-dotenv).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Base de données PostgreSQL ──────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 5432))
DB_NAME     = os.getenv("DB_NAME", "fraud_detection")
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ── API temps réel ──────────────────────────────────────────────────────────
API_BASE_URL     = os.getenv(
    "API_BASE_URL",
    "https://sdacelo-real-time-fraud-detection.hf.space",
)
API_ENDPOINT     = "/current-transactions"
POLL_INTERVAL_S  = int(os.getenv("POLL_INTERVAL_S", 60))   # toutes les minutes

# ── Modèle ML ───────────────────────────────────────────────────────────────
MODEL_PATH       = os.getenv("MODEL_PATH", "model/fraud_model.pkl")
FRAUD_THRESHOLD  = float(os.getenv("FRAUD_THRESHOLD", 0.5))

# ── Alertes email (SMTP) ────────────────────────────────────────────────────
SMTP_HOST        = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT        = int(os.getenv("SMTP_PORT", 587))
SMTP_USER        = os.getenv("SMTP_USER", "")
SMTP_PASSWORD    = os.getenv("SMTP_PASSWORD", "")
ALERT_RECIPIENTS = os.getenv("ALERT_RECIPIENTS", "").split(",")

# ── Rapport quotidien ────────────────────────────────────────────────────────
DAILY_REPORT_HOUR   = int(os.getenv("DAILY_REPORT_HOUR", 8))
DAILY_REPORT_MINUTE = int(os.getenv("DAILY_REPORT_MINUTE", 0))
