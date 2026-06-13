@echo off
:: ============================================================
:: 3_start_webserver.bat — Webserver Airflow (UI)
:: Laissez ce terminal ouvert !
:: ============================================================

call conda activate fraude-ml

set AIRFLOW_HOME=C:/Users/khouf/airflow
set AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////C:/Users/khouf/airflow/airflow.db
set AIRFLOW__CORE__LOAD_EXAMPLES=false

echo ============================================================
echo  Demarrage du Webserver Airflow sur le port 8082...
echo.
echo  Ouvrez dans le navigateur :
echo    http://localhost:8082
echo    Login : admin
echo    Password : admin
echo ============================================================
echo.

airflow webserver --port 8082
