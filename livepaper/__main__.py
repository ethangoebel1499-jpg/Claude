"""livepaper - real-time paper trading of the tested strategies, with a dashboard.

  python -m livepaper run       [--config live.toml]   start trading (Ctrl+C to stop)
  python -m livepaper dashboard [--config live.toml]   open the dashboard in your browser
  python -m livepaper check     [--config live.toml]   test the data feeds / Webull keys, place no orders
  python -m livepaper sync      [--config live.toml]   push the trade record to the paper-trades branch now
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from livepaper.runner import build, load_config, run_forever  # noqa: E402
from livepaper.store import Store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m livepaper", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "dashboard", "check", "sync"])
    ap.add_argument("--config", default="live.toml")
    a = ap.parse_args()
    if not Path(a.config).exists():
        raise SystemExit(f"{a.config} not found: copy livepaper/live.example.toml to live.toml and edit it")
    cfg = load_config(a.config)
    db = cfg.get("app", {}).get("db", "livepaper.db")

    if a.command == "dashboard":
        env = {**os.environ, "LIVEPAPER_DB": db, "LIVEPAPER_CONFIG": a.config}
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(HERE / "dashboard.py")], env=env))

    store = Store(db)
    if a.command == "run":
        run_forever(cfg, store)
    elif a.command == "sync":
        from livepaper.sync import sync
        print(sync(store))
    else:
        runners = build(cfg, store)
        for rn in runners:
            bars = rn._bars()
            print(f"{rn.r['name']}: {len(bars)} closed {rn.r['timeframe']} candles, last {bars.index[-1]}, "
                  f"price {rn._price():,.4f}")
            if rn.r["feed"] == "webull":
                base = {"crypto": "crypto"}.get(rn.r["asset"], "stock")
                raw = rn.feed.raw_bars(rn.r["symbol"], base, "M60", count=2)
                print(f"   raw Webull bar sample: {str(raw)[:400]}")
            print(rn.checklist_fn(bars, -1, label=rn.r["name"]))
            print()
        if cfg.get("app", {}).get("broker") == "webull" and runners:
            print(f"Webull paper account found: ...{str(runners[0].pm.broker.account)[-4:]} (no orders placed)")


if __name__ == "__main__":
    main()
