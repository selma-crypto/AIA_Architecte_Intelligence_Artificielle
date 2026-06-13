"""
Smoke tests post-déploiement — à lancer contre l'environnement cible.
Usage: python tests/smoke_test.py --base-url http://localhost:8000
"""
import sys
import argparse
import requests

TRANSACTION = {
    "transaction_id": "SMOKE-001",
    "features": {
        "account_age_days": 365.0, "hour_local": 14, "dayofweek_local": 3, "day_local": 15,
        "is_night": 0, "time_since_last": 86400.0, "amount": 149.99,
        "avg_amount_user": 80.0, "avg_amount_user_past": 75.0,
        "amount_diff_user_avg": 70.0, "amount_delta_prev": 50.0,
        "total_transactions_user": 120, "transaction_count_cum": 120,
        "tx_last_24h": 2, "tx_last_7d": 5, "tx_last_30d": 18, "user_tx_count": 120,
        "user_fraud_rate": 0.0, "user_fraud_count": 0, "user_has_fraud_history": 0,
        "is_new_account": 0, "shipping_distance_km": 5.0, "distance_amount_ratio": 0.03,
        "country_bin_mismatch": 0, "avs_match": 1, "cvv_result": 1,
        "three_ds_flag": 1, "security_mismatch_score": 0.0,
    }
}

def check(name, condition, msg=""):
    status = "✅" if condition else "❌"
    print(f"  {status} {name}" + (f" — {msg}" if msg else ""))
    return condition

def run(base_url):
    print(f"\nSmoke tests → {base_url}\n")
    results = []

    # Health
    r = requests.get(f"{base_url}/health", timeout=10)
    results.append(check("GET /health → 200", r.status_code == 200, f"HTTP {r.status_code}"))
    results.append(check("model_loaded = true", r.json().get("model_loaded") is True))

    # Predict
    r = requests.post(f"{base_url}/predict", json=TRANSACTION, timeout=10)
    results.append(check("POST /predict → 200", r.status_code == 200, f"HTTP {r.status_code}"))
    if r.status_code == 200:
        d = r.json()
        results.append(check("fraud_probability dans [0,1]", 0 <= d["fraud_probability"] <= 1))
        results.append(check("risk_level valide", d["risk_level"] in ["LOW","MEDIUM","HIGH","CRITICAL"]))

    # Metrics
    r = requests.get(f"{base_url}/metrics", timeout=10)
    results.append(check("GET /metrics → 200", r.status_code == 200))

    # Docs
    r = requests.get(f"{base_url}/docs", timeout=10)
    results.append(check("GET /docs → 200", r.status_code == 200))

    passed = sum(results)
    total  = len(results)
    print(f"\n{passed}/{total} tests passés\n")
    return passed == total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    sys.exit(0 if run(args.base_url) else 1)
