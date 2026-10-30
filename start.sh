#!/usr/bin/env bash
# Start the paper trader and the dashboard:   bash start.sh
# Close with Ctrl+C (stops both). Your open paper positions are saved and pick up next time.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Run   bash setup.sh   first."; exit 1
fi
. .venv/bin/activate

python -m livepaper run >> trader.log 2>&1 &
TRADER=$!
trap 'kill $TRADER 2>/dev/null; echo "Stopped."' EXIT
echo "Paper trader running in the background (log: trader.log)."

# opens in your browser; the printed address also works on your phone on the same Wi-Fi
python -m livepaper dashboard --phone
