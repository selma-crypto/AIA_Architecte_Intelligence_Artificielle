"""
Tests unitaires — API FastAPI Fraud Detection
=============================================
Lance avec : pytest tests/ -v
"""

import json
import pickle
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixture: mock modèle ────────────────────────────────────────────────────
@pytest.fixture
def mock_model():
    """Modèle XGBoost simulé qui retourne une probabilité de fraude fixe."""
    model = MagicMock()
    model.predict_proba = lambda X: np.array([[0.7, 0.3]])
    model.feature_names_in_ = None
    return model


@pytest.fixture
def client(mock_model, tmp_path):
    """TestClient FastAPI avec modèle simulé injecté dans l'état global."""
    # Créer un pickle temporaire
    pkl_path = tmp_path / "fraud_xgb_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"model": mock_model, "final_columns": None}, f)

    with patch.dict("os.environ", {"MODEL_PATH": str(pkl_path)}):
        from api.main import app, _state, load_model
        _state["model"] = mock_model
        _state["model_features"] = None
        _state["model_version"] = "test-v1"
        _state["model_loaded_at"] = "2026-06-01T00:00:00"
        yield TestClient(app)


# ── Payload de transaction valide ───────────────────────────────────────────
VALID_TRANSACTION = {
    "transaction_id": "TX-TEST-001",
    "features": {
        "account_age_days": 365.0,
        "hour_local": 14,
        "dayofweek_local": 3,
        "day_local": 15,
        "is_night": 0,
        "time_since_last": 86400.0,
        "amount": 149.99,
        "avg_amount_user": 80.0,
        "avg_amount_user_past": 75.0,
        "amount_diff_user_avg": 70.0,
        "amount_delta_prev": 50.0,
        "total_transactions_user": 120,
        "transaction_count_cum": 120,
        "tx_last_24h": 2,
        "tx_last_7d": 5,
        "tx_last_30d": 18,
        "user_tx_count": 120,
        "user_fraud_rate": 0.0,
        "user_fraud_count": 0,
        "user_has_fraud_history": 0,
        "is_new_account": 0,
        "shipping_distance_km": 5.0,
        "distance_amount_ratio": 0.03,
        "country_bin_mismatch": 0,
        "avs_match": 1,
        "cvv_result": 1,
        "three_ds_flag": 1,
        "security_mismatch_score": 0.0,
    },
    "threshold": 0.30,
}


# ── Tests /health ────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "uptime_seconds" in data

    def test_health_no_model(self):
        from api.main import app, _state
        original = _state["model"]
        _state["model"] = None
        tc = TestClient(app)
        response = tc.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"
        _state["model"] = original


# ── Tests /predict ───────────────────────────────────────────────────────────
class TestPredict:
    def test_predict_success(self, client, mock_model):
        mock_model.predict_proba = lambda X: np.array([[0.7, 0.3]])
        response = client.post("/predict", json=VALID_TRANSACTION)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "risk_level" in data
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert data["transaction_id"] == "TX-TEST-001"

    def test_predict_high_risk(self, client, mock_model):
        mock_model.predict_proba = lambda X: np.array([[0.1, 0.9]])
        response = client.post("/predict", json=VALID_TRANSACTION)
        assert response.status_code == 200
        data = response.json()
        assert data["is_fraud"] is True
        assert data["risk_level"] == "CRITICAL"

    def test_predict_low_risk(self, client, mock_model):
        mock_model.predict_proba = lambda X: np.array([[0.98, 0.02]])
        response = client.post("/predict", json=VALID_TRANSACTION)
        assert response.status_code == 200
        data = response.json()
        assert data["is_fraud"] is False
        assert data["risk_level"] == "LOW"

    def test_predict_custom_threshold(self, client, mock_model):
        mock_model.predict_proba = lambda X: np.array([[0.7, 0.3]])
        payload = {**VALID_TRANSACTION, "threshold": 0.50}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        # 0.3 < 0.50 → pas de fraude avec ce seuil élevé
        assert data["is_fraud"] is False
        assert data["threshold_used"] == 0.50

    def test_predict_missing_field(self, client):
        bad_payload = {"features": {"amount": 100.0}}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_no_model(self):
        from api.main import app, _state
        original = _state["model"]
        _state["model"] = None
        tc = TestClient(app)
        response = tc.post("/predict", json=VALID_TRANSACTION)
        assert response.status_code == 503
        _state["model"] = original


# ── Tests /predict/batch ─────────────────────────────────────────────────────
class TestBatchPredict:
    def test_batch_success(self, client, mock_model):
        mock_model.predict_proba = lambda X: np.array([[0.6, 0.4]])
        payload = {"transactions": [VALID_TRANSACTION, VALID_TRANSACTION]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert "fraud_detected" in data
        assert "fraud_rate" in data
        assert len(data["predictions"]) == 2

    def test_batch_empty_rejected(self, client):
        response = client.post("/predict/batch", json={"transactions": []})
        assert response.status_code == 422


# ── Tests /metrics ───────────────────────────────────────────────────────────
class TestMetrics:
    def test_metrics_structure(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "fraud_rate" in data
        assert "avg_latency_ms" in data
        assert "model_info" in data

    def test_fraud_rate_increases(self, client, mock_model):
        mock_model.predict_proba = lambda X: np.array([[0.1, 0.9]])
        client.post("/predict", json=VALID_TRANSACTION)
        response = client.get("/metrics")
        data = response.json()
        assert data["total_predictions"] >= 1


# ── Tests /model/info ────────────────────────────────────────────────────────
class TestModelInfo:
    def test_model_info(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert "decision_threshold" in data
        assert data["decision_threshold"] == 0.30
