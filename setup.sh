#!/usr/bin/env bash
# One-time setup on a Mac or Linux laptop:   bash setup.sh
# Add --streams to also install the stream agent's audio transcriber (large download).
set -e
cd "$(dirname "$0")"

PY=""
for p in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    PY="$p"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer is needed. Mac: install it from https://www.python.org/downloads/ and run this again."
  exit 1
fi
echo "Using $($PY --version)"

[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip -q
echo "Installing packages (a few minutes the first time)..."
pip install -q -r requirements.txt streamlit requests yfinance anthropic
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
echo
echo "Setup done. Start everything with:   bash start.sh"
