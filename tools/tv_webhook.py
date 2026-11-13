"""Receive TradingView strategy alerts (webhooks) and paper-trade them exactly, with no screen reading.

  python tools/tv_webhook.py --token YOUR-SECRET            # listens on port 8787
  python tools/tv_webhook.py --token YOUR-SECRET --dry-run  # logs what it would do, sends nothing

Setup: tools/TV_WEBHOOK.md. In short: one TradingView alert on the GRIND strategy, condition "Order fills only",
webhook URL = your public address, message = the JSON in that file.

What it does with each alert (TradingView tells us the position before and after the fill):
  - the position grows in a direction (a new entry or a pyramid add) -> opens a paper run named TV-<date>-<time>
    at TradingView's own fill price; the run then follows the TV ladder (stop 9%, TP 1.7 / 2.3 / 3.4% = 33/33/34%)
  - the position goes flat -> closes every run it opened
  - the position shrinks but isn't flat (a partial take profit) -> ignored (the runs' own ladder handles those)
Every alert and every action is appended to runtime/tv_webhook_log.csv.
"""

from __future__ import annotations

import argparse
import csv
import hmac
import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "runtime" / "tv_webhook_state.json"
LOG = ROOT / "runtime" / "tv_webhook_log.csv"
LOG_COLS = ["logged_at", "event", "run", "side", "price", "size_pct", "mode", "note", "payload"]
LOCK = Lock()


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"size": 0.0, "open": [], "seen": []}      # size: signed position after the last alert; open: [{run, side}]


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def log(event: str, a: dict, mode: str, note: str, payload: dict) -> None:
    new = not LOG.exists()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLS)
        w.writerow([datetime.now().isoformat(timespec="seconds"), event, a.get("run", ""), a.get("side", ""),
                    a.get("price", ""), a.get("size_pct", ""), mode, note, json.dumps(payload)])


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def signed_size(p: dict) -> float | None:
    """Position after the fill, signed (long +, short -). TradingView's {{strategy.position_size}} is signed already."""
    size = _num(p.get("size"))
    if size is None:
        return None
    pos = str(p.get("position", "")).lower()
    if pos == "flat":
        return 0.0
    return abs(size) if pos == "long" else -abs(size) if pos == "short" else size


def plan(p: dict, st: dict, account: float, max_open: int = 8) -> list[dict]:
    """Alert -> actions. Pure: changes nothing; the caller applies and then saves st['size'] and st['open']."""
    new = signed_size(p)
    price = _num(p.get("price"))
    if new is None or price is None:
        return []
    old = st.get("size", 0.0)
    acts: list[dict] = []
    flips = old and new and (old > 0) != (new > 0)
    if new == 0 or flips:                                       # flat, or reversed: everything open is closed first
        for o in st["open"]:
            acts.append({"action": "close", "run": o["run"], "side": o["side"], "price": price})
    still_open = 0 if (new == 0 or flips) else len(st["open"])   # pyramiding: each add is its own run, up to max_open
    if new != 0 and (flips or abs(new) > abs(old) + 1e-12) and still_open < max_open:   # grew, or reversed: a new entry
        side = "long" if new > 0 else "short"
        added = abs(new) if flips or old == 0 else abs(new) - abs(old)
        usd = (_num(p.get("contracts")) or added) * price
        pct = max(1, min(100, round(usd / account * 100)))
        t = _when(p.get("time"))
        base = f"TV-{t:%m%d-%H%M}"
        taken = {o["run"] for o in st["open"]} | set(st.get("seen", []))
        run, n = base, 2
        while run in taken:
            run, n = f"{base}-{n}", n + 1
        acts.append({"action": side, "run": run, "side": side, "price": price, "size_pct": pct})
    return acts


def _when(s) -> datetime:
    for f in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(s), f)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def process(p: dict, st: dict, args) -> list[str]:
    acts = plan(p, st, args.account, args.max_open)
    out = []
    if not acts:
        log("ignored", {"price": p.get("price")}, "dry" if args.dry_run else "live",
            f"position {st.get('size')} -> {signed_size(p)}: nothing to do (partial exit, or unreadable)", p)
    store = None
    if acts and not args.dry_run:
        sys.path.insert(0, str(ROOT))
        from livepaper import claude_trades
        from livepaper.store import Store
        store = Store(ROOT / args.db)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    for a in acts:
        what = f"{a['action'].upper()} {a['run']} @ {a['price']}" + (f" size {a['size_pct']}%" if a.get("size_pct") else "")
        if args.dry_run:
            log("dry-run " + a["action"], a, "dry", "", p)
            out.append("DRY-RUN would: " + what)
            continue
        cmd = {"id": f"tvhook-{a['action']}-{a['run']}-{stamp}", "run": a["run"], "action": a["action"], "price": a["price"],
               "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "reason": f"TradingView alert: position {st.get('size')} -> {signed_size(p)} at {p.get('price')}"}
        if a.get("size_pct"):
            cmd["size_pct"] = a["size_pct"]
        claude_trades.poll(store, commands=[cmd])
        log(a["action"], a, "live", "", p)
        if a["action"] == "close":
            st["open"] = [o for o in st["open"] if o["run"] != a["run"]]
            st["seen"] = (st.get("seen", []) + [a["run"]])[-300:]
        else:
            st["open"].append({"run": a["run"], "side": a["side"]})
        out.append("did: " + what)
    if not args.dry_run:
        st["size"] = signed_size(p) if signed_size(p) is not None else st.get("size", 0.0)
        save_state(st)
    return out


def make_handler(args):
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, text: str) -> None:
            body = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                                         # open the address in a browser to check it is up
            self._send(200, "tv_webhook is running")

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                p = json.loads(self.rfile.read(min(n, 20_000)) or b"{}")
            except Exception:
                return self._send(400, "bad json")
            if not hmac.compare_digest(str(p.get("token", "")), args.token):
                return self._send(403, "bad token")
            p.pop("token", None)
            with LOCK:
                try:
                    out = process(p, load_state(), args)
                except Exception as e:                             # never lose an alert silently
                    log("error", {}, "live", repr(e), p)
                    print("ERROR:", repr(e))
                    return self._send(500, "error")
            for line in out:
                print(datetime.now().strftime("%H:%M:%S"), line)
            self._send(200, "ok")

        def log_message(self, *a):
            pass

    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=os.environ.get("TV_WEBHOOK_TOKEN"), help="secret that must be in every alert")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 and put a tunnel in front of it")
    ap.add_argument("--db", default="livepaper.db")
    ap.add_argument("--account", type=float, default=1_930_000, help="TradingView's equity, to turn its order size into a %")
    ap.add_argument("--max-open", type=int, default=8, help="the chart's pyramiding: most runs open at once (each entry is its own run)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.token or len(args.token) < 12:
        print("Give a secret of at least 12 characters: --token ... (or set TV_WEBHOOK_TOKEN)")
        return 2
    print(f"tv_webhook listening on http://{args.host}:{args.port}  ({'DRY RUN' if args.dry_run else 'live'}). Ctrl+C to stop.")
    ThreadingHTTPServer((args.host, args.port), make_handler(args)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
