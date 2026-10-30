"""livepaper - real-time paper trading of the tested strategies, with a dashboard.

  python -m livepaper run       [--config live.toml]   start trading (Ctrl+C to stop)
  python -m livepaper dashboard [--config live.toml]   open the dashboard in your browser
                                [--phone]              ...and on your phone (same Wi-Fi)
  python -m livepaper check     [--config live.toml]   test the data feeds / Webull keys, place no orders
                                [--test-stop]          ...and confirm Webull accepts stop orders
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
from mcbot.fmt import price  # noqa: E402


def lan_addresses() -> list[str]:
    """This computer's address(es) on the local network."""
    import socket
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.168.1.1", 9))          # picks the outgoing interface; nothing is sent
            ips.add(sock.getsockname()[0])
    except OSError:
        pass
    try:
        ips.update(ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127."))
    except OSError:
        pass
    return sorted(ips) or ["<this computer's IP>"]


def test_stops(runners) -> None:
    """One tiny buy-stop far above the price per asset type (needs no holdings), then cancel it."""
    seen = set()
    for rn in runners:
        if rn.r["asset"] in seen:
            continue
        seen.add(rn.r["asset"])
        px = rn._price()
        qty = 1 if rn.r["asset"] == "stock" else round(10 / px, 6) or 1e-6
        coid = rn.pm.broker.place_stop(rn.r["name"], "buy", qty, px * 1.5)
        if coid is None:
            print(f"🔴 {rn.r['asset']}: Webull REFUSED the stop order (reason above). Trades in "
                  f"{rn.r['asset']} are only protected while the trader runs.")
            continue
        rn.pm.broker.cancel_stop(rn.r["name"], coid, qty, px * 1.5)
        print(f"🟢 {rn.r['asset']}: Webull accepted a stop order ({rn.r['symbol']} @ {price(px * 1.5)}), now "
              f"cancelled. Trades in {rn.r['asset']} keep their stop with this computer off.")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m livepaper", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "dashboard", "check", "sync"])
    ap.add_argument("--config", default="live.toml")
    ap.add_argument("--phone", action="store_true",
                    help="dashboard: also serve it to your phone / other devices on your Wi-Fi")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--test-stop", action="store_true",
                    help="check: place and cancel a far-away stop order at Webull for each kind of asset")
    a = ap.parse_args()
    if not Path(a.config).exists():
        raise SystemExit(f"{a.config} not found: copy livepaper/live.example.toml to live.toml and edit it")
    cfg = load_config(a.config)
    db = cfg.get("app", {}).get("db", "livepaper.db")

    if a.command == "dashboard":
        env = {**os.environ, "LIVEPAPER_DB": db, "LIVEPAPER_CONFIG": a.config}
        cmd = [sys.executable, "-m", "streamlit", "run", str(HERE / "dashboard.py"), "--server.port", str(a.port),
               "--browser.gatherUsageStats", "false"]
        if a.phone:
            cmd += ["--server.address", "0.0.0.0"]
            print("\nOpen one of these on your phone (same Wi-Fi), then Share -> 'Add to Home Screen':")
            for ip in lan_addresses():
                print(f"   http://{ip}:{a.port}")
            print("Away from home: install Tailscale on this computer and your phone, then use the computer's "
                  "Tailscale address instead.\n")
        raise SystemExit(subprocess.call(cmd, env=env))

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
                  f"price {price(rn._price())}")
            if rn.r["feed"] == "webull":
                base = {"crypto": "crypto"}.get(rn.r["asset"], "stock")
                raw = rn.feed.raw_bars(rn.r["symbol"], base, "M60", count=2)
                print(f"   raw Webull bar sample: {str(raw)[:400]}")
            print(rn.checklist_fn(bars, -1, label=rn.r["name"]))
            print()
        broker = runners[0].pm.broker if runners else None
        if hasattr(broker, "place_stop"):
            print(f"Webull paper account found: ...{str(broker.account)[-4:]}")
            if a.test_stop:
                test_stops(runners)
            else:
                print("No orders placed. Add --test-stop to confirm Webull accepts the stop orders that protect "
                      "trades while this computer is off.")


if __name__ == "__main__":
    main()
