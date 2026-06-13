@echo off
:: ============================================================
:: 6_populate_mlflow.bat — Alimente MLflow avec les metriques
:: Lance apres que fraud-mlflow soit demarre (http://localhost:5001)
:: ============================================================

call conda activate fraude-ml

set MLFLOW_TRACKING_URI=http://localhost:5001

echo ============================================================
echo  Alimentation de MLflow avec les metriques du modele...
echo ============================================================
echo.

cd /d "%~dp0.."
python setup_mlflow_db.py

echo.
echo Ouvrez http://localhost:5001 pour voir les experiences !
pause
