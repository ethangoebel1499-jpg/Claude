#!/usr/bin/env bash
# Opens the Paper Trader app (same as the Applications shortcut).
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Run   bash setup.sh   first."; exit 1
fi
exec ./.venv/bin/python launcher.pyw
