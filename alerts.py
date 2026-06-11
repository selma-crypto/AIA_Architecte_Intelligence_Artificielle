"""
alerts.py — Envoi d'alertes email pour les fraudes détectées en temps réel.

Un email est envoyé dès qu'une transaction est classée frauduleuse,
sauf si l'alerte a déjà été envoyée (champ alert_sent en base).
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config

logger = logging.getLogger(__name__)


def _build_fraud_email(tx: dict, fraud_proba: float) -> MIMEMultipart:
    """Construit le message email d'alerte fraude."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 Fraude détectée — {tx.get('trans_num', 'N/A')}"
    msg["From"]    = config.SMTP_USER
    msg["To"]      = ", ".join(config.ALERT_RECIPIENTS)

    amt      = tx.get("amt", 0)
    merchant = tx.get("merchant", "inconnu")
    category = tx.get("category", "")
    name     = f"{tx.get('first', '')} {tx.get('last', '')}".strip()
    city     = tx.get("city", "")
    state    = tx.get("state", "")
    dt       = tx.get("trans_datetime", "")

    html = f"""
    <html><body>
    <h2 style="color:#c0392b;">⚠️ Transaction frauduleuse détectée</h2>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
      <tr><th>Champ</th><th>Valeur</th></tr>
      <tr><td>N° transaction</td><td><b>{tx.get('trans_num', 'N/A')}</b></td></tr>
      <tr><td>Date / heure</td><td>{dt}</td></tr>
      <tr><td>Titulaire</td><td>{name}</td></tr>
      <tr><td>Montant</td><td><b>${amt:.2f}</b></td></tr>
      <tr><td>Commerçant</td><td>{merchant}</td></tr>
      <tr><td>Catégorie</td><td>{category}</td></tr>
      <tr><td>Localisation</td><td>{city}, {state}</td></tr>
      <tr><td>Score de fraude</td><td><b>{fraud_proba:.1%}</b></td></tr>
    </table>
    <p style="color:#7f8c8d;font-size:12px;">
      Message automatique généré par le pipeline de détection de fraude.
    </p>
    </body></html>
    """

    plain = (
        f"FRAUDE DÉTECTÉE\n"
        f"Transaction : {tx.get('trans_num')}\n"
        f"Montant     : ${amt:.2f}\n"
        f"Commerçant  : {merchant}\n"
        f"Score       : {fraud_proba:.1%}\n"
    )

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))
    return msg


def send_fraud_alert(tx: dict, fraud_proba: float) -> bool:
    """
    Envoie une alerte email. Retourne True si l'envoi a réussi.
    """
    if not config.SMTP_USER or not config.ALERT_RECIPIENTS[0]:
        logger.warning("SMTP non configuré — alerte non envoyée pour %s", tx.get("trans_num"))
        return False

    try:
        msg = _build_fraud_email(tx, fraud_proba)
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.sendmail(
                config.SMTP_USER,
                config.ALERT_RECIPIENTS,
                msg.as_string(),
            )
        logger.info("Alerte fraude envoyée pour %s", tx.get("trans_num"))
        return True

    except Exception as e:
        logger.error("Échec envoi email pour %s : %s", tx.get("trans_num"), e)
        return False
