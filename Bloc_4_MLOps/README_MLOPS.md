# Pipeline MLOps — Détection de Fraude

Extension MLOps du projet de détection de fraude e-commerce (Groupe 4, CDSD).

## Architecture

```
Raw Data → Feature Engineering → XGBoost + Optuna → MLflow Registry
                                                            ↓
                                                   GitHub Actions CI/CD
                                                            ↓
                                            FastAPI (port 8001)
                                                            ↓
                                            Evidently Monitoring (PSI par feature)
                                                            ↓
                                   Airflow DAG (port 8082) → Auto-Retrain si dérive
```

## Démarrage rapide

```bash
# 1. Placer le modèle à la racine
cp fraud_xgb_model.pkl .

# 2. Démarrer API + MLflow + Monitoring scheduler
docker compose up -d

# 3. Démarrer avec Airflow (DAG orchestration)
docker compose --profile airflow up -d
```

## Services

| Service            | Port | Description |
|--------------------|------|-------------|
| FastAPI            | 8001 | API REST d'inférence — `/predict`, `/predict/batch`, `/metrics` |
| MLflow             | 5001 | Tracking des expériences, Model Registry |
| Airflow            | 8082 | Orchestration DAG (profil optionnel) |
| Monitor Scheduler  | —    | Daemon monitoring toutes les 6h (conteneur interne) |

## API — Endpoints principaux

### `POST /predict`
```json
{
  "transaction_id": "TX-001",
  "features": { "amount": 149.99, "account_age_days": 365, "..." : "..." },
  "threshold": 0.30
}
```
Réponse :
```json
{
  "fraud_probability": 0.823,
  "is_fraud": true,
  "risk_level": "CRITICAL",
  "threshold_used": 0.30,
  "inference_time_ms": 3.2
}
```

### `POST /predict/batch`
Jusqu'à 1000 transactions en une requête.

### `GET /metrics`
Statistiques opérationnelles temps réel : latences P95, taux de fraude, erreurs.

Documentation interactive complète : `http://localhost:8001/docs`

## CI/CD GitHub Actions

| Workflow | Déclencheur | Étapes |
|----------|-------------|--------|
| `ci.yml` | Push / PR | Lint → Tests pytest → Validate model → Build Docker |
| `cd.yml` | Push main  | Build & Push GHCR → Deploy Staging → Smoke test → Deploy Prod |

### Secrets à configurer (GitHub → Settings → Secrets)
```
MODEL_BUCKET_URL    URL S3 publique du modèle
STAGING_HOST        IP du serveur staging
STAGING_USER        Utilisateur SSH staging
STAGING_SSH_KEY     Clé SSH privée staging
PROD_HOST           IP production
PROD_USER           Utilisateur SSH prod
PROD_SSH_KEY        Clé SSH privée prod
SLACK_WEBHOOK_URL   Webhook pour les notifications
MLFLOW_TRACKING_URI URI MLflow distant (optionnel)
```

## Monitoring

```bash
# Lancer une analyse de dérive (manuel)
python monitoring/monitor.py \
  --reference data/X_test_app_sample.csv \
  --current   data/new_data.csv \
  --output    reports/

# Scheduler continu (via Docker)
docker compose up -d monitor-scheduler
```

**Seuils PSI :**
- `< 0.10` : distribution stable ✅
- `0.10 – 0.25` : dérive modérée ⚠️ (surveiller)
- `> 0.25` : dérive critique 🚨 → réentraînement automatique déclenché

## Airflow DAG

```
check_new_data → run_monitoring → decide_retrain (BranchOperator)
                                        ↓                    ↓
                                  retrain_model          skip_retrain
                                  validate_model
                                  register_model
                                  deploy_model
                                        ↓                    ↓
                                  notify_failure        notify_success
```

- Schedule : `0 2 * * *` (chaque nuit à 02:00 UTC)
- Démarrage : `docker compose --profile airflow up -d`
- UI : `http://localhost:8082` — login `admin` / `admin123`

## Réentraînement

```bash
# Manuel
python pipeline/retrain.py \
  --trigger manual \
  --data-path data/new_data.csv \
  --auto-deploy

# Avec optimisation Optuna (30 trials)
python pipeline/retrain.py \
  --trigger drift \
  --data-path data/new_data.csv \
  --optimize --n-trials 30 \
  --auto-deploy
```

**Quality gate** :
```
MIN_RECALL_PROD    = 0.80
MIN_PRECISION_PROD = 0.55
MIN_F1_PROD        = 0.65
```

## Tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v

# Smoke test post-déploiement
python tests/smoke_test.py --base-url http://localhost:8001
```

## Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `MODEL_PATH` | `fraud_xgb_model.pkl` | Chemin du modèle |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | URI MLflow |
| `DECISION_THRESHOLD` | `0.30` | Seuil de décision |
| `MONITOR_INTERVAL_HOURS` | `6` | Fréquence monitoring |
| `AUTO_RETRAIN` | `true` | Réentraînement auto si drift |
| `ALERT_WEBHOOK_URL` | _(vide)_ | Webhook alertes (Slack, Teams) |
| `MIN_RECALL_PROD` | `0.80` | Quality gate — recall minimum |
| `MIN_F1_PROD` | `0.65` | Quality gate — F1 minimum |

## Structure du projet

```
.
├── api/
│   ├── main.py                  # FastAPI — endpoints REST
│   ├── Dockerfile.api           # Image Docker de l'API
│   └── requirements-api.txt     # Dépendances API complètes
├── monitoring/
│   ├── monitor.py               # Détection de dérive Evidently + PSI
│   ├── scheduler.py             # Daemon de monitoring continu (toutes les 6h)
│   ├── fraud-monitor.service    # Systemd service (Linux)
│   └── crontab.txt              # Cron Linux alternatif
├── pipeline/
│   ├── retrain.py               # Pipeline de réentraînement automatisé
│   ├── validate_model.py        # Quality gate CI/CD
│   └── airflow_dag.py           # DAG Apache Airflow
├── tests/
│   ├── test_api.py              # Tests unitaires pytest
│   └── smoke_test.py            # Tests post-déploiement
├── scripts/
│   ├── 4_start_all_docker.bat   # Démarrer tous les services Docker
│   ├── 5_stop_all.bat           # Arrêter tous les services
│   └── 6_populate_mlflow.bat    # Peupler MLflow avec les métriques
├── .github/workflows/
│   ├── ci.yml                   # Intégration continue
│   └── cd.yml                   # Déploiement continu
├── .dvc/
│   └── config                   # Configuration DVC (remote S3/GCS/Azure)
├── dvc.yaml                     # Stages DVC : prepare → train → evaluate
├── params.yaml                  # Hyperparamètres versionnés DVC
├── docker-compose.yml           # Orchestration de tous les services
├── requirements.txt             # Dépendances racine
├── setup_mlflow_db.py           # Script de population MLflow initial
└── README_MLOPS.md              # Cette documentation
```
