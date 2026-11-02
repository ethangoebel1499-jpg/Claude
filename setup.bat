@echo off
REM One-time setup on a Windows laptop: double-click this file. Nothing needs to be installed first:
REM it downloads its own private Python and every library into this folder (runtime\ and .venv\),
REM so nothing else on your computer changes. Delete the folder to remove it all.
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

set "APPDIR=%CD%"
set "UV=%APPDIR%\runtime\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%APPDIR%\runtime\python"
set "UV_CACHE_DIR=%APPDIR%\runtime\cache"
set "UV_PYTHON_PREFERENCE=only-managed"
if not exist runtime mkdir runtime

REM 1. uv: a small tool (from the makers of the Ruff linter) that downloads Python and libraries.
if not exist "%UV%" (
  echo Downloading the installer tool...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; [Net.ServicePointManager]::SecurityProtocol = 'Tls12'; $zip = Join-Path $env:APPDIR 'runtime\uv.zip'; Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip; $tmp = Join-Path $env:APPDIR 'runtime\uv_tmp'; Expand-Archive -Force $zip $tmp; Move-Item -Force (Get-ChildItem -Recurse $tmp -Filter uv.exe | Select-Object -First 1).FullName $env:UV; Remove-Item -Recurse -Force $tmp, $zip"
  if not exist "%UV%" (
    echo !! Couldn't download the installer tool. Check the internet connection and run setup.bat again.
    pause
    exit /b 1
  )
)

REM 2. A private Python 3.13 (every library has ready-made installers for it) in a fresh .venv.
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys, tkinter; sys.exit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo Replacing an old .venv...
    rmdir /s /q .venv
  )
)
if not exist .venv\Scripts\python.exe (
  echo Downloading Python 3.13 ^(about 30 MB, only this once^)...
  "%UV%" venv --python 3.13 --seed .venv
  if errorlevel 1 (
    echo !! Couldn't set up Python - send Claude the output above.
    pause
    exit /b 1
  )
)
set "VPY=%APPDIR%\.venv\Scripts\python.exe"

REM 3. The libraries.
echo Installing libraries (a minute or two the first time)...
"%UV%" pip install --python "%VPY%" -r app-requirements.txt
if errorlevel 1 (
  echo !! Library install failed - send Claude the output above.
  pause
  exit /b 1
)
"%UV%" pip install --python "%VPY%" webull-openapi-python-sdk || echo !! Webull SDK did not install - the built-in paper broker still works.
if "%1"=="--streams" "%UV%" pip install --python "%VPY%" faster-whisper

if not exist live.toml (
  copy livepaper\live.example.toml live.toml >nul
  echo Created live.toml ^(your paper-trading runs^).
)

echo.
echo Self-check...
"%VPY%" healthcheck.py
if errorlevel 1 (
  echo !! The self-check failed - send Claude the output above.
  pause
  exit /b 1
)

echo.
echo Creating the "Paper Trader" shortcuts (Desktop and Start menu)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = New-Object -ComObject WScript.Shell; foreach ($d in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) { $l = $s.CreateShortcut((Join-Path $d 'Paper Trader.lnk')); $l.TargetPath = $env:APPDIR + '\.venv\Scripts\pythonw.exe'; $l.Arguments = [char]34 + $env:APPDIR + '\bootstrap.pyw' + [char]34; $l.WorkingDirectory = $env:APPDIR; $l.IconLocation = $env:APPDIR + '\assets\icon.ico'; $l.Description = 'Paper Trader'; $l.Save() }"
if errorlevel 1 (
  echo !! Couldn't create the shortcuts - open the app with start.bat instead.
)
echo.
echo Setup done. Open "Paper Trader" from your Desktop or Start menu.
echo It updates itself from GitHub, and goes back to the last working version if an update fails.
start "" "%APPDIR%\.venv\Scripts\pythonw.exe" "%APPDIR%\bootstrap.pyw"
pause
