#!/usr/bin/env bash
# One-time setup on a Mac or Linux laptop:   bash setup.sh
# Nothing needs to be installed first: it downloads its own Python and libraries into this folder.
# Add --streams to also install the stream agent's audio transcriber (large download).
set -e
cd "$(dirname "$0")"

export APPDIR="$(pwd)"
export UV_PYTHON_INSTALL_DIR="$APPDIR/runtime/python" UV_CACHE_DIR="$APPDIR/runtime/cache" UV_PYTHON_PREFERENCE=only-managed
UV="$APPDIR/runtime/uv"
mkdir -p runtime

# 1. uv: a small tool that downloads Python and the libraries into this folder (nothing system-wide)
if [ ! -x "$UV" ]; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) T=aarch64-apple-darwin ;;
    Darwin-*)     T=x86_64-apple-darwin ;;
    Linux-aarch64|Linux-arm64) T=aarch64-unknown-linux-gnu ;;
    *)            T=x86_64-unknown-linux-gnu ;;
  esac
  echo "Downloading the installer tool..."
  curl -fsSL "https://github.com/astral-sh/uv/releases/latest/download/uv-$T.tar.gz" | tar -xz -C runtime
  mv "runtime/uv-$T/uv" "$UV" && rm -rf "runtime/uv-$T"
fi

# 2. a private Python 3.13 in a fresh .venv
if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import sys, tkinter; sys.exit(sys.version_info[:2] != (3, 13))' 2>/dev/null; then
  echo "Replacing an old .venv..."; rm -rf .venv
fi
[ -x .venv/bin/python ] || { echo "Downloading Python 3.13 (only this once)..."; "$UV" venv --python 3.13 --seed .venv; }

# 3. the libraries
echo "Installing libraries (a minute or two the first time)..."
"$UV" pip install --python .venv/bin/python -r app-requirements.txt
"$UV" pip install --python .venv/bin/python webull-openapi-python-sdk || echo "!! Webull SDK did not install - the built-in paper broker still works."
if [ "$1" = "--streams" ]; then
  "$UV" pip install --python .venv/bin/python faster-whisper || echo "!! faster-whisper did not install."
fi
. .venv/bin/activate

if [ ! -f live.toml ]; then
  cp livepaper/live.example.toml live.toml
  echo "Created live.toml (your paper-trading runs)."
fi

echo
echo "Self-check..."
python healthcheck.py || { echo "!! The self-check failed - send Claude the output above."; exit 1; }
# an app you can launch with the custom icon
APPDIR="$(pwd)"
if [ "$(uname)" = "Darwin" ]; then
  APP="$HOME/Applications/Paper Trader.app"
  mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
  cp assets/icon.icns "$APP/Contents/Resources/icon.icns"
  printf '#!/bin/bash\ncd "%s"\nexec ./.venv/bin/python bootstrap.pyw\n' "$APPDIR" > "$APP/Contents/MacOS/Paper Trader"
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
Exec=$APPDIR/.venv/bin/python $APPDIR/bootstrap.pyw
Icon=$APPDIR/assets/icon.png
Path=$APPDIR
DESK
  echo "Added Paper Trader to your applications menu."
fi
echo
echo "Setup done. Open Paper Trader from Applications (or: bash start.sh)."
echo "It updates itself from GitHub every time it starts."
