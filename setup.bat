@echo off
REM One-time setup on a Windows laptop: double-click this file.
REM Run "setup.bat --streams" to also install the stream agent's audio transcriber (large download).
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" on the first screen, then run this again.
  pause
  exit /b 1
)
py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))"
if errorlevel 1 (
  echo Your Python is older than 3.10. Install a newer one from https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist .venv py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
echo Installing packages (a few minutes the first time)...
pip install -q -r requirements.txt streamlit requests yfinance anthropic
if errorlevel 1 (
  echo !! Package install failed - send Claude the output above.
  pause
  exit /b 1
)
pip install -q webull-openapi-python-sdk || echo !! Webull SDK did not install - the built-in paper broker still works.
if "%1"=="--streams" pip install -q faster-whisper

if not exist live.toml (
  copy livepaper\live.example.toml live.toml >nul
  echo Created live.toml ^(your paper-trading runs^).
)

echo.
echo Checking the price feeds (no orders are placed)...
python -m livepaper check
echo.
echo Setup done. Start everything by double-clicking start.bat
pause
