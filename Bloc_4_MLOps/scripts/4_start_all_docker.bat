@echo off
:: ============================================================
:: 4_start_all_docker.bat — Lance API + MLflow avec Docker
:: ============================================================

echo ============================================================
echo  Demarrage de la stack Docker MLOps (API + MLflow)
echo ============================================================
echo.

cd /d "%~dp0.."

docker compose up -d api mlflow

echo.
echo Attente du demarrage (15 secondes)...
timeout /t 15 /nobreak >nul

echo.
echo Status des containers :
docker compose ps

echo.
echo ============================================================
echo  Services disponibles :
echo    API Swagger  : http://localhost:8001/docs
echo    API Health   : http://localhost:8001/health
echo    MLflow UI    : http://localhost:5001
echo ============================================================
pause
