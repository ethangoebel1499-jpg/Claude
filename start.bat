@echo off
REM Start the paper trader and the dashboard: double-click this file.
REM Two windows open: "Paper trader" (leave it running) and the dashboard. Close both to stop.
cd /d "%~dp0"
if not exist .venv (
  echo Run setup.bat first.
  pause
  exit /b 1
)
start "Paper trader" .venv\Scripts\python.exe -m livepaper run
REM opens in your browser; the printed address also works on your phone on the same Wi-Fi
.venv\Scripts\python.exe -m livepaper dashboard --phone
pause
