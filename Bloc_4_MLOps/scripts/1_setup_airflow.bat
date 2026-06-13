@echo off
:: ============================================================
:: 1_setup_airflow.bat — Initialisation Airflow (une seule fois)
:: Double-cliquez sur ce fichier ou lancez-le dans le terminal
:: ============================================================

echo ============================================================
echo  Setup Airflow — Fraud Detection MLOps
echo ============================================================

:: Activer conda fraude-ml
call conda activate fraude-ml

:: Variables Airflow
set AIRFLOW_HOME=C:/Users/khouf/airflow
set AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////C:/Users/khouf/airflow/airflow.db
set AIRFLOW__CORE__LOAD_EXAMPLES=false
set AIRFLOW__WEBSERVER__SECRET_KEY=fraud-mlops-2026

echo.
echo [1/5] Creation des dossiers Airflow...
if not exist "C:\Users\khouf\airflow\dags" mkdir "C:\Users\khouf\airflow\dags"
if not exist "C:\Users\khouf\airflow\logs" mkdir "C:\Users\khouf\airflow\logs"
if not exist "C:\Users\khouf\airflow\plugins" mkdir "C:\Users\khouf\airflow\plugins"
echo     OK

echo.
echo [2/5] Initialisation de la base de donnees...
airflow db migrate
echo     OK

echo.
echo [3/5] Creation de l'utilisateur admin...
airflow users create --username admin --password admin --firstname Admin --lastname MLOps --role Admin --email admin@fraud.local 2>nul
echo     admin / admin cree (ou deja existant)

echo.
echo [4/5] Copie du DAG...
:: Chercher airflow_dag.py dans le dossier courant ou pipeline/
if exist "pipeline\airflow_dag.py" (
    copy /Y "pipeline\airflow_dag.py" "C:\Users\khouf\airflow\dags\fraud_detection_mlops_pipeline.py"
    echo     DAG copie depuis pipeline\
) else if exist "airflow_dag.py" (
    copy /Y "airflow_dag.py" "C:\Users\khouf\airflow\dags\fraud_detection_mlops_pipeline.py"
    echo     DAG copie depuis dossier courant
) else (
    echo     ATTENTION : airflow_dag.py introuvable !
)

echo.
echo [5/5] Verification...
airflow dags list 2>nul | findstr "fraud" && echo     DAG fraud_detection_mlops_pipeline trouve !

echo.
echo ============================================================
echo  Setup termine ! Lancez maintenant :
echo    2_start_scheduler.bat   (Terminal 1)
echo    3_start_webserver.bat   (Terminal 2)
echo ============================================================
pause
