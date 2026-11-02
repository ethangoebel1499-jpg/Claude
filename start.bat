@echo off
REM Opens the Paper Trader app (same as the Desktop / Start menu shortcut).
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0launcher.pyw"
