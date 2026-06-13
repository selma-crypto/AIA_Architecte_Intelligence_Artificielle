@echo off
:: ============================================================
:: 5_stop_all.bat — Arrete tous les containers Docker
:: ============================================================

echo Arret de tous les services Docker...
cd /d "%~dp0.."
docker compose down
echo Termine.
pause
