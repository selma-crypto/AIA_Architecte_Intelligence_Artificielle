"""
daily_report.py — Rapport quotidien envoyé chaque matin par email.

Contenu du rapport :
  - Résumé statistique de la veille (total transactions, montant, nb fraudes)
  - Tableau détaillé de toutes les fraudes détectées
"""

import logging
import smtplib
from datetime import date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config
import db

logger = logging.getLogger(__name__)


def _build_report_email(summary: dict, frauds: list[dict]) -> MIMEMultipart:
    yesterday = (date.today() - timedelta(days=1)).strftime("%d/%m/%Y")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📊 Rapport fraude du {yesterday}"
    msg["From"]    = config.SMTP_USER
    msg["To"]      = ", ".join(config.ALERT_RECIPIENTS)

    total  = summary.get("total_transactions", 0)
    amount = summary.get("total_amount") or 0
    nb_fraud = summary.get("fraud_count", 0)
    fraud_amt = summary.get("fraud_amount") or 0
    avg_proba = summary.get("avg_fraud_proba") or 0

    fraud_rate = (nb_fraud / total * 100) if total else 0

    # Lignes du tableau de fraudes
    rows_html = ""
    for f in frauds:
        rows_html += f"""
        <tr>
          <td>{f.get('trans_num','')}</td>
          <td>{f.get('trans_datetime','')}</td>
          <td>{f.get('first','')} {f.get('last','')}</td>
          <td>${float(f.get('amt',0)):.2f}</td>
          <td>{f.get('merchant','')}</td>
          <td>{f.get('category','')}</td>
          <td>{f.get('city','')}, {f.get('state','')}</td>
          <td><b>{float(f.get('fraud_proba',0)):.1%}</b></td>
        </tr>"""

    html = f"""
    <html><body>
    <h2>📊 Rapport quotidien — {yesterday}</h2>

    <h3>Résumé</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
      <tr><th>Indicateur</th><th>Valeur</th></tr>
      <tr><td>Total transactions</td><td>{total:,}</td></tr>
      <tr><td>Volume total</td><td>${amount:,.2f}</td></tr>
      <tr><td>Fraudes détectées</td><td><b style="color:#c0392b;">{nb_fraud}</b></td></tr>
      <tr><td>Montant frauduleux</td><td><b style="color:#c0392b;">${fraud_amt:,.2f}</b></td></tr>
      <tr><td>Taux de fraude</td><td>{fraud_rate:.2f} %</td></tr>
      <tr><td>Score moyen fraude</td><td>{avg_proba:.1%}</td></tr>
    </table>

    <h3>Détail des fraudes</h3>
    {"<p>Aucune fraude détectée.</p>" if not frauds else f'''
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
      <tr>
        <th>N° transaction</th><th>Date/heure</th><th>Titulaire</th>
        <th>Montant</th><th>Commerçant</th><th>Catégorie</th>
        <th>Localisation</th><th>Score</th>
      </tr>
      {rows_html}
    </table>'''}

    <p style="color:#7f8c8d;font-size:12px;">Rapport automatique — pipeline de détection de fraude.</p>
    </body></html>
    """

    plain = (
        f"RAPPORT FRAUDE — {yesterday}\n"
        f"Transactions : {total}\n"
        f"Fraudes      : {nb_fraud} ({fraud_rate:.2f}%)\n"
        f"Montant frauduleux : ${fraud_amt:,.2f}\n"
    )

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))
    return msg


def send_daily_report() -> None:
    """Génère et envoie le rapport quotidien."""
    logger.info("Génération du rapport quotidien...")

    summary = db.get_yesterday_summary()
    frauds  = db.get_yesterday_frauds()

    if not config.SMTP_USER or not config.ALERT_RECIPIENTS[0]:
        logger.warning("SMTP non configuré — rapport non envoyé.")
        return

    try:
        msg = _build_report_email(summary, frauds)
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.sendmail(
                config.SMTP_USER,
                config.ALERT_RECIPIENTS,
                msg.as_string(),
            )
        logger.info(
            "Rapport quotidien envoyé : %d transactions, %d fraudes.",
            summary.get("total_transactions", 0),
            summary.get("fraud_count", 0),
        )
    except Exception as e:
        logger.error("Échec envoi rapport quotidien : %s", e)
