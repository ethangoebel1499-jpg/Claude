@echo off
REM One-time setup on a Windows laptop: double-click this file.
REM Run "setup.bat --streams" to also install the stream agent's audio transcriber (large download).
cd /d "%~dp0"

REM Opened straight from the ZIP? Windows runs it from a temporary copy that gets deleted.
echo "%~dp0" | findstr /l /i /c:".zip" >nul
if not errorlevel 1 (
  echo.
  echo This is running from INSIDE the ZIP file, in a temporary folder Windows deletes later.
  echo Your settings and paper trades would be lost.
  echo.
  echo Fix: close this, right-click Claude-master.zip, choose "Extract All...", pick a folder
  echo such as Documents, then open the extracted Claude-master folder and run setup.bat there.
  pause
  exit /b 1
)

where py >nul 2>nul
if errorlevel 1 (
  echo Python is needed. Install Python 3.13 from https://www.python.org/downloads/windows/
  echo and tick "Add python.exe to PATH" on the first screen, then run this again.
  pause
  exit /b 1
)

REM Use a Python version every package has ready-made installers for (no compiler needed).
set "PYV="
for %%v in (3.13 3.12 3.11 3.10) do (
  if not defined PYV (
    py -%%v -c "import sys" >nul 2>nul && set "PYV=%%v"
  )
)
if not defined PYV (
  echo.
  echo Your Python version isn't supported by the trading packages yet:
  py -c "import sys; print('   found Python', sys.version.split()[0])"
  echo.
  echo Install Python 3.13 from https://www.python.org/downloads/windows/ - pick the latest
  echo "Python 3.13" Windows installer 64-bit. You can keep your current Python; both work side by side.
  echo Then run setup.bat again.
  pause
  exit /b 1
)
echo Using Python %PYV%

REM A .venv made by another Python version (e.g. a failed earlier attempt) is rebuilt.
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys; v=sys.version_info; sys.exit(0 if f'{v[0]}.{v[1]}'=='%PYV%' else 1)" >nul 2>nul
  if errorlevel 1 (
    echo Removing the old .venv made with a different Python...
    rmdir /s /q .venv
  )
)
if not exist .venv\Scripts\python.exe py -%PYV% -m venv .venv
set "VPY=.venv\Scripts\python.exe"

%VPY% -m pip install --upgrade pip -q
echo Installing packages (a few minutes the first time)...
%VPY% -m pip install -q --only-binary=:all: -r requirements.txt streamlit requests yfinance anthropic
if errorlevel 1 (
  echo !! Package install failed - send Claude the output above.
  pause
  exit /b 1
)
%VPY% -m pip install -q webull-openapi-python-sdk || echo !! Webull SDK did not install - the built-in paper broker still works.
if "%1"=="--streams" %VPY% -m pip install -q faster-whisper

if not exist live.toml (
  copy livepaper\live.example.toml live.toml >nul
  echo Created live.toml ^(your paper-trading runs^).
)

echo.
echo Checking the price feeds (no orders are placed)...
%VPY% -m livepaper check
echo.
echo Setup done. Start everything by double-clicking start.bat
pause
