"""
api_client.py — Client pour l'API de transactions temps réel.

L'API HuggingFace Space expose :
  GET /current-transactions
  → liste de transactions mises à jour toutes les minutes.

On maintient un set des trans_num déjà vus pour ne traiter
que les nouvelles transactions à chaque appel.
"""

import json
import logging
from typing import Iterator

import requests

import config

logger = logging.getLogger(__name__)

_seen_trans_nums: set[str] = set()


def fetch_current_transactions() -> list[dict]:
    """
    Appelle l'API et retourne la liste brute des transactions.

    L'API renvoie un JSON pandas orient="split" doublement encodé :
      - la réponse HTTP est une string JSON
      - qui contient {"columns": [...], "index": [...], "data": [[...]]}
    On reconstruit une liste de dicts à partir de columns + data.
    """
    url = config.API_BASE_URL.rstrip("/") + config.API_ENDPOINT
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    data = resp.json()

    # Cas 1 : doublement encodé → data est une string à re-parser
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Impossible de parser la réponse string : %s", data[:200])
            return []

    # Cas 2 : format pandas split {"columns": [...], "index": [...], "data": [[...]]}
    if isinstance(data, dict) and "columns" in data and "data" in data:
        columns = data["columns"]
        rows    = data["data"]
        return [dict(zip(columns, row)) for row in rows]

    # Cas 3 : liste de dicts directe
    if isinstance(data, list):
        return data

    # Cas 4 : dict avec clé transactions/data
    if isinstance(data, dict):
        return data.get("transactions", data.get("data", []))

    logger.warning("Format inattendu de l'API : %s", type(data))
    return []


def get_new_transactions() -> Iterator[dict]:
    """
    Générateur : appelle l'API, filtre les transactions déjà traitées
    et normalise chaque enregistrement avant de le yielder.
    """
    raw = fetch_current_transactions()
    new_count = 0

    for tx in raw:
        trans_num = str(tx.get("trans_num", tx.get("id", "")))
        if not trans_num or trans_num in _seen_trans_nums:
            continue

        _seen_trans_nums.add(trans_num)
        new_count += 1
        yield _normalize(tx)

    logger.info("API : %d transactions reçues, %d nouvelles.", len(raw), new_count)


def _normalize(tx: dict) -> dict:
    """
    Normalise les noms de champs et les types pour correspondre
    au schéma PostgreSQL et au préprocesseur du modèle.

    Adaptez cette fonction si l'API change de format.
    """
    from datetime import datetime

    # Parsing de la date/heure (formats courants)
    raw_dt = tx.get("trans_date_trans_time") or tx.get("trans_datetime") or ""
    try:
        trans_datetime = datetime.fromisoformat(raw_dt)
    except (ValueError, TypeError):
        trans_datetime = None

    # Parsing de la date de naissance
    raw_dob = tx.get("dob", "")
    try:
        from datetime import date
        dob = date.fromisoformat(raw_dob)
    except (ValueError, TypeError):
        dob = None

    return {
        "trans_num":      str(tx.get("trans_num", tx.get("id", ""))),
        "trans_datetime": trans_datetime,
        "cc_num":         str(tx.get("cc_num", "")),
        "merchant":       str(tx.get("merchant", "")),
        "category":       str(tx.get("category", "")),
        "amt":            float(tx.get("amt", 0.0)),
        "first":          str(tx.get("first", "")),
        "last":           str(tx.get("last", "")),
        "gender":         str(tx.get("gender", "")),
        "city":           str(tx.get("city", "")),
        "state":          str(tx.get("state", "")),
        "zip":            str(tx.get("zip", "")),
        "lat":            float(tx.get("lat", 0.0)),
        "long":           float(tx.get("long", 0.0)),
        "city_pop":       int(tx.get("city_pop", 0)),
        "job":            str(tx.get("job", "")),
        "dob":            dob,
        "merch_lat":      float(tx.get("merch_lat", 0.0)),
        "merch_long":     float(tx.get("merch_long", 0.0)),
    }
