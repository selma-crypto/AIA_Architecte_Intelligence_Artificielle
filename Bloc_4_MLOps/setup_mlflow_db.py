"""
setup_mlflow_db.py — Alimente le serveur MLflow avec les métriques du modèle.
Pointe vers le MLflow Docker sur http://localhost:5001
"""
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("fraud-detection-xgboost")

with mlflow.start_run(run_name="xgboost-final"):
    mlflow.log_params({
        "n_estimators": 360, "max_depth": 3, "learning_rate": 0.08989,
        "gamma": 6.832, "min_child_weight": 16, "subsample": 0.999,
        "colsample_bytree": 0.542, "scale_pos_weight": 5.004,
        "decision_threshold": 0.30, "optimizer": "Optuna 150 trials",
        "n_train": 239756, "n_test": 59939, "n_features": 54,
    })
    mlflow.log_metrics({
        "train_recall": 0.877, "train_precision": 0.617, "train_f1": 0.724,
        "test_recall": 0.869, "test_precision": 0.614, "test_f1": 0.720,
        "test_accuracy": 0.985, "test_roc_auc": 0.981,
        "true_positives": 1149, "false_positives": 721,
        "false_negatives": 173, "true_negatives": 57896,
    })
    print("Run 1/4 loggé : xgboost-final")

scenarios = [
    ("conservateur", 0.50, 0.812, 0.791, 0.801, 289, 222),
    ("equilibre",    0.30, 0.869, 0.614, 0.720, 721, 173),
    ("agressif",     0.15, 0.934, 0.413, 0.573, 1980, 86),
]
for i, (name, seuil, rec, prec, f1, fp, fn) in enumerate(scenarios, 2):
    with mlflow.start_run(run_name=f"seuil-{name}-{seuil}"):
        mlflow.log_param("decision_threshold", seuil)
        mlflow.log_metrics({
            "recall": rec, "precision": prec, "f1_score": f1,
            "false_positives": fp, "false_negatives": fn,
        })
    print(f"Run {i}/4 loggé : seuil-{name}-{seuil}")

print("\nOuvrez http://localhost:5001 → expérience 'fraud-detection-xgboost'")
