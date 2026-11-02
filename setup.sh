#!/usr/bin/env bash
# One-time setup on a Mac or Linux laptop:   bash setup.sh
# Add --streams to also install the stream agent's audio transcriber (large download).
set -e
cd "$(dirname "$0")"

PY=""
for p in python3.13 python3.12 python3.11 python3.10 python3; do
  # 3.10-3.13: every package has ready-made installers (newer Pythons would need a compiler)
  if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 13))'; then
    PY="$p"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10-3.13 is needed. Install Python 3.13 from https://www.python.org/downloads/ (it can sit next to"
  echo "a newer Python) and run this again."
  exit 1
fi
echo "Using $($PY --version)"

[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip -q
echo "Installing packages (a few minutes the first time)..."
pip install -q --only-binary=:all: -r app-requirements.txt
pip install -q webull-openapi-python-sdk || echo "!! Webull SDK did not install - the built-in paper broker still works."
if [ "$1" = "--streams" ]; then
  pip install -q faster-whisper || echo "!! faster-whisper did not install - you can still add streams from YouTube transcripts."
fi

if [ ! -f live.toml ]; then
  cp livepaper/live.example.toml live.toml
  echo "Created live.toml (your paper-trading runs)."
fi

echo
echo "Checking the price feeds (no orders are placed)..."
python -m livepaper check || echo "!! The check failed - send Claude the output above."
# an app you can launch with the custom icon
APPDIR="$(pwd)"
if [ "$(uname)" = "Darwin" ]; then
  APP="$HOME/Applications/Paper Trader.app"
  mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
  cp assets/icon.icns "$APP/Contents/Resources/icon.icns"
  printf '#!/bin/bash\ncd "%s"\nexec ./.venv/bin/python launcher.pyw\n' "$APPDIR" > "$APP/Contents/MacOS/Paper Trader"
  chmod +x "$APP/Contents/MacOS/Paper Trader"
  cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Paper Trader</string>
  <key>CFBundleExecutable</key><string>Paper Trader</string>
  <key>CFBundleIconFile</key><string>icon.icns</string>
  <key>CFBundleIdentifier</key><string>local.papertrader</string>
  <key>CFBundlePackageType</key><string>APPL</string>
</dict></plist>
PLIST
  echo "Created ~/Applications/Paper Trader.app (drag it to the Dock if you like)."
else
  mkdir -p "$HOME/.local/share/applications"
  cat > "$HOME/.local/share/applications/paper-trader.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Paper Trader
Exec=$APPDIR/.venv/bin/python $APPDIR/launcher.pyw
Icon=$APPDIR/assets/icon.png
Path=$APPDIR
DESK
  echo "Added Paper Trader to your applications menu."
fi
echo
echo "Setup done. Open Paper Trader from Applications (or: bash start.sh)."
echo "It updates itself from GitHub every time it starts."
