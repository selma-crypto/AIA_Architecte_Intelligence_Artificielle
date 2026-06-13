@echo off
:: ============================================================
:: 2_start_scheduler.bat — Scheduler Airflow
:: Laissez ce terminal ouvert !
:: ============================================================

call conda activate fraude-ml

set AIRFLOW_HOME=C:/Users/khouf/airflow
set AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////C:/Users/khouf/airflow/airflow.db
set AIRFLOW__CORE__LOAD_EXAMPLES=false

echo ============================================================
echo  Demarrage du Scheduler Airflow...
echo  Laissez ce terminal ouvert.
echo  Ouvrez 3_start_webserver.bat dans un autre terminal.
echo ============================================================
echo.

airflow scheduler
