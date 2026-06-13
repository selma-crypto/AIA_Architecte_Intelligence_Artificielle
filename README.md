# AIA — Architecte en Intelligence Artificielle

> Certification professionnelle · Projet fil rouge : **Détection de Fraude E-commerce**

Ce dépôt regroupe l'ensemble des travaux réalisés dans le cadre de la certification **Architecte en Intelligence Artificielle**, organisés en 4 blocs de compétences. Le projet fil rouge porte sur la construction d'un système de détection de fraude sur des transactions e-commerce, de la gouvernance des données jusqu'au déploiement en production avec MLOps.

---

## Structure du dépôt

```
AIA_Architecte_Intelligence_Artificielle/
├── Bloc_1_Gouvernance_Donnees/      # Gouvernance & qualité des données
├── Bloc_2_Architecture_Donnees/     # Architecture & ingestion des données
├── Bloc_3_Pipeline_IA/              # Pipeline ML & modélisation
└── Bloc_4_MLOps/                    # MLOps, déploiement & monitoring
```

---

## Blocs de compétences

### Bloc 1 — Gouvernance des Données
Mise en place d'une stratégie de gouvernance des données : identification des sources, qualité, traçabilité et conformité RGPD appliquées au contexte de la détection de fraude.

### Bloc 2 — Architecture des Données
Conception de l'architecture data : pipeline d'ingestion, stockage, et structuration des données de transactions e-commerce (~300 000 transactions).

### Bloc 3 — Pipeline IA & Modélisation
Construction du pipeline de machine learning complet :
- Exploration et feature engineering (**54 features** : 6 initiales + 48 créées)
- Entraînement de modèles (Logistic Regression, Random Forest, **XGBoost**)
- Optimisation des hyperparamètres avec **Optuna** (150 trials)
- Sélection du meilleur modèle (XGBoost · F1-score : 0.81)

### Bloc 4 — MLOps & Déploiement en Production
Industrialisation complète du modèle avec une stack MLOps de bout en bout.

---

## Bloc 4 — MLOps Pipeline (Détail)

### Stack technique

| Composant | Technologie |
|---|---|
| API de prédiction | FastAPI · Uvicorn |
| Tracking expériences | MLflow |
| Versioning données/modèles | DVC |
| Monitoring drift | Evidently AI (PSI) |
| Orchestration pipeline | Apache Airflow |
| CI/CD | GitHub Actions |
| Conteneurisation | Docker · Docker Compose |
| Tests | Pytest |

### Architecture du pipeline Airflow

```
check_new_data → run_monitoring → decide_retrain
                                        │
                    ┌───────────────────┴──────────────────┐
               [Drift détecté]                    [Pas de drift]
                    │                                       │
         retrain_model                              skip_retrain
         validate_model                                     │
         register_model                            notify_success
         deploy_model
         notify_success
```

### Structure Bloc 4

```
Bloc_4_MLOps/
├── api/                    # FastAPI — endpoint /predict
│   ├── main.py
│   ├── Dockerfile.api
│   └── requirements-api.txt
├── pipeline/               # Scripts ML (train, evaluate, retrain)
├── monitoring/             # Détection de drift avec Evidently
├── scripts/                # Utilitaires Docker & MLflow
├── tests/                  # Tests unitaires et smoke tests
├── data/                   # Données (versionnées via DVC)
├── logs/                   # Logs d'exécution
├── reports/                # Rapports de monitoring
├── .github/workflows/      # CI/CD GitHub Actions
├── docker-compose.yml      # Stack complète (API + MLflow + Airflow)
├── dvc.yaml                # Pipeline DVC
├── params.yaml             # Hyperparamètres du modèle
└── requirements.txt        # Dépendances Python
```

### Démarrage rapide

**Prérequis** : Docker Desktop, Python 3.9+

```bash
# 1. Cloner le repo
git clone https://github.com/selma-crypto/AIA_Architecte_Intelligence_Artificielle.git
cd AIA_Architecte_Intelligence_Artificielle/Bloc_4_MLOps

# 2. Configurer les variables d'environnement
cp .env.example .env   # puis éditer les valeurs

# 3. Lancer la stack complète
docker-compose up -d

# 4. Accéder aux interfaces
#    API FastAPI  → http://localhost:8001/docs
#    MLflow UI   → http://localhost:5000
#    Airflow UI  → http://localhost:8080
```

**Exemple de prédiction :**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```

---

## Résultats du modèle

| Modèle | Recall (Test) | Precision (Test) | F1-score (Test) |
|---|---|---|---|
| Logistic Regression | 0.8949 | 0.2194 | 0.3524 |
| Random Forest | 0.8578 | 0.6491 | 0.7390 |
| **XGBoost** ✅ | **0.8321** | **0.7919** | **0.8115** |

> Modèle retenu : **XGBoost** · Optimisation Optuna sur 150 trials

---

## Auteur

**Selma** — Candidate à la certification Architecte en Intelligence Artificielle  
GitHub : [@selma-crypto](https://github.com/selma-crypto)
