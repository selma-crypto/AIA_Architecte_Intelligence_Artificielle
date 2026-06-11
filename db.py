"""
db.py — Couche d'accès PostgreSQL.

Tables créées automatiquement au premier lancement :
  - transactions  : toutes les transactions reçues depuis l'API
  - predictions   : résultat du modèle pour chaque transaction
"""

import logging
from datetime import date, datetime, timedelta
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

import config

logger = logging.getLogger(__name__)

# Pool de connexions (min 1, max 5)
_pool: ThreadedConnectionPool | None = None


def init_pool() -> None:
    """Initialise le pool de connexions et crée les tables si besoin."""
    global _pool
    _pool = ThreadedConnectionPool(
        minconn=1,
        maxconn=5,
        dsn=config.DATABASE_URL,
    )
    _create_tables()
    logger.info("Pool PostgreSQL initialisé.")


@contextmanager
def get_conn():
    """Context manager : emprunte une connexion depuis le pool."""
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


# ── Création des tables ──────────────────────────────────────────────────────

def _create_tables() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id               SERIAL PRIMARY KEY,
                    trans_num        TEXT UNIQUE NOT NULL,
                    trans_datetime   TIMESTAMP,
                    cc_num           TEXT,
                    merchant         TEXT,
                    category         TEXT,
                    amt              NUMERIC(12, 2),
                    first            TEXT,
                    last             TEXT,
                    gender           TEXT,
                    city             TEXT,
                    state            TEXT,
                    zip              TEXT,
                    lat              NUMERIC(10, 6),
                    long             NUMERIC(10, 6),
                    city_pop         INTEGER,
                    job              TEXT,
                    dob              DATE,
                    merch_lat        NUMERIC(10, 6),
                    merch_long       NUMERIC(10, 6),
                    received_at      TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id               SERIAL PRIMARY KEY,
                    trans_num        TEXT UNIQUE NOT NULL,
                    fraud_proba      NUMERIC(6, 4),
                    is_fraud         BOOLEAN,
                    predicted_at     TIMESTAMP DEFAULT NOW(),
                    alert_sent       BOOLEAN DEFAULT FALSE
                );
            """)
    logger.info("Tables vérifiées / créées.")


# ── Insertion ────────────────────────────────────────────────────────────────

def insert_transaction(tx: dict) -> bool:
    """
    Insère une transaction. Retourne True si nouvelle, False si déjà existante.
    """
    sql = """
        INSERT INTO transactions
            (trans_num, trans_datetime, cc_num, merchant, category, amt,
             first, last, gender, city, state, zip, lat, long, city_pop,
             job, dob, merch_lat, merch_long)
        VALUES
            (%(trans_num)s, %(trans_datetime)s, %(cc_num)s, %(merchant)s,
             %(category)s, %(amt)s, %(first)s, %(last)s, %(gender)s,
             %(city)s, %(state)s, %(zip)s, %(lat)s, %(long)s, %(city_pop)s,
             %(job)s, %(dob)s, %(merch_lat)s, %(merch_long)s)
        ON CONFLICT (trans_num) DO NOTHING
        RETURNING id;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tx)
            return cur.fetchone() is not None  # None = déjà existant


def insert_prediction(trans_num: str, fraud_proba: float, is_fraud: bool) -> None:
    sql = """
        INSERT INTO predictions (trans_num, fraud_proba, is_fraud)
        VALUES (%s, %s, %s)
        ON CONFLICT (trans_num) DO NOTHING;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (trans_num, fraud_proba, is_fraud))


def mark_alert_sent(trans_num: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE predictions SET alert_sent = TRUE WHERE trans_num = %s",
                (trans_num,)
            )


# ── Requêtes pour le rapport quotidien ──────────────────────────────────────

def get_yesterday_summary() -> dict:
    """Retourne les statistiques de la veille pour le rapport quotidien."""
    yesterday = date.today() - timedelta(days=1)
    start = datetime.combine(yesterday, datetime.min.time())
    end   = datetime.combine(date.today(), datetime.min.time())

    sql = """
        SELECT
            COUNT(t.id)                                    AS total_transactions,
            SUM(t.amt)                                     AS total_amount,
            COUNT(p.id) FILTER (WHERE p.is_fraud = TRUE)   AS fraud_count,
            SUM(t.amt)  FILTER (WHERE p.is_fraud = TRUE)   AS fraud_amount,
            AVG(p.fraud_proba)                             AS avg_fraud_proba
        FROM transactions t
        LEFT JOIN predictions p ON t.trans_num = p.trans_num
        WHERE t.trans_datetime >= %s AND t.trans_datetime < %s;
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, (start, end))
            row = cur.fetchone()
            return dict(row) if row else {}


def get_yesterday_frauds() -> list[dict]:
    """Retourne le détail des transactions frauduleuses de la veille."""
    yesterday = date.today() - timedelta(days=1)
    start = datetime.combine(yesterday, datetime.min.time())
    end   = datetime.combine(date.today(), datetime.min.time())

    sql = """
        SELECT t.trans_num, t.trans_datetime, t.merchant, t.category,
               t.amt, t.first, t.last, t.city, t.state, p.fraud_proba
        FROM transactions t
        JOIN predictions p ON t.trans_num = p.trans_num
        WHERE p.is_fraud = TRUE
          AND t.trans_datetime >= %s AND t.trans_datetime < %s
        ORDER BY p.fraud_proba DESC;
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, (start, end))
            return [dict(r) for r in cur.fetchall()]
