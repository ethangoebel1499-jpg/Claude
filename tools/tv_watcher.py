"""Local TradingView watcher: reads the 'List of trades' of TradingView's GRIND BTC [45MIN] strategy off the screen
with a LOCAL vision model (Ollama) and mirrors it into the paper trader. No cloud usage.

  python tools/tv_watcher.py --dry-run --once            # screenshot -> parse -> print what it WOULD do
  python tools/tv_watcher.py --image shot.png --dry-run  # parse a saved screenshot instead of the screen
  python tools/tv_watcher.py                             # live: at :01 and :31 (and with --every N minutes)

Needs on the Paper Trader PC:  pip install mss pillow requests   and   ollama pull qwen2.5vl:7b  (any Ollama vision model:
--model).  TradingView must be showing the List of trades (the enlarged strategy tester) on the screen being captured.

How it decides (matches the rules the user set):
  * a trade is identified by its ENTRY time + price, not its number (TradingView renumbers rows on partial exits)
  * a new entry that is still open  -> opens a paper run named TV-<number>, sized TV size USD / 1,930,000 (1%, max 100)
  * a trade whose entry has an exit and no open row left for it -> closes that run (partial exits are ignored)
  * every read must be seen twice in a row (--confirm-seconds apart) before anything is sent, and sanity-checked
  * every action is appended to tools/grind_trade_log.csv and runtime/tv_watcher_state.json is the memory
Runs fill at the entry / exit price shown on TradingView (needs the autogrind-tv-exact app patch).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "runtime" / "tv_watcher_state.json"
LOG = ROOT / "tools" / "grind_trade_log.csv"
ACCOUNT_USD = 1_930_000.0
OLLAMA = "http://localhost:11434/api/chat"
LOG_COLS = ["logged_at", "event", "tv_trade", "side", "tv_entry_time", "tv_entry_price", "tv_size_btc", "tv_size_usd",
            "tv_exit_time", "tv_exit_price", "paper_run", "size_pct", "mode", "note"]

PROMPT = """This is TradingView's 'List of trades' table. Each trade has two lines: Exit (date/time, price) and Entry
(date/time, price). Read EVERY visible trade, newest first. Reply with ONLY a JSON array, no other text. One object per trade:
{"num": 2120, "side": "Short" or "Long", "entry_time": "Nov 12, 2026, 10:30", "entry_price": 71690.2,
 "size_btc": 29.31, "size_usd": 2100000, "exit_time": null or "Nov 12, 2026, 09:44", "exit_price": null or 71478.4}
Exit shows 'Open' and a dash when the trade is still open: then exit_time and exit_price are null.
size_usd: '2.1 M USD' = 2100000, '665.85 K USD' = 665850. Prices have no thousands separators in the JSON."""


# ---------------------------------------------------------------- reading the screen
def screenshot_png() -> bytes:
    import mss
    from PIL import Image
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def ask_model(png: bytes, model: str, url: str = OLLAMA) -> str:
    import requests
    r = requests.post(url, json={"model": model, "stream": False, "options": {"temperature": 0},
                                 "messages": [{"role": "user", "content": PROMPT,
                                               "images": [base64.b64encode(png).decode()]}]}, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse_rows(text: str) -> list[dict]:
    """Model reply -> clean rows. Rows that fail a sanity check are dropped (never guessed)."""
    m = re.search(r"\[.*\]", text, re.S)
    try:
        raw = json.loads(m.group(0)) if m else []
    except Exception:
        return []
    rows = []
    for r in raw:
        try:
            side = str(r["side"]).strip().lower()
            row = {"num": int(r["num"]), "side": side, "entry_time": " ".join(str(r["entry_time"]).split()),
                   "entry_price": float(r["entry_price"]), "size_btc": float(r.get("size_btc") or 0),
                   "size_usd": float(r.get("size_usd") or 0),
                   "exit_time": r.get("exit_time") or None, "exit_price": r.get("exit_price")}
            row["exit_price"] = float(row["exit_price"]) if row["exit_price"] is not None else None
        except Exception:
            continue
        if side not in ("long", "short") or not 1_000 < row["entry_price"] < 1_000_000:
            continue
        if (row["exit_price"] is None) != (row["exit_time"] is None):
            continue
        if row["exit_price"] is not None and not 1_000 < row["exit_price"] < 1_000_000:
            continue
        rows.append(row)
    return rows


def key(r: dict) -> str:
    return f"{r['side']}|{r['entry_time']}|{r['entry_price']:.1f}"


# ---------------------------------------------------------------- deciding what changed
def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"open": {}, "seen": []}     # open: key -> {run, num}; seen: keys already handled and closed


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def plan(rows: list[dict], st: dict) -> list[dict]:
    """Actions (open / close) needed to bring the paper trader in line with the TradingView list."""
    open_keys = {key(r) for r in rows if r["exit_price"] is None}
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(key(r), []).append(r)
    acts = []
    for k in sorted(open_keys):
        if k in st["open"] or k in st["seen"]:
            continue
        r = [x for x in by_key[k] if x["exit_price"] is None][0]
        pct = max(1, min(100, round(r["size_usd"] / ACCOUNT_USD * 100))) if r["size_usd"] else 100
        acts.append({"action": r["side"], "key": k, "run": f"TV-{r['num']}", "row": r, "size_pct": pct,
                     "price": r["entry_price"]})
    for k, info in st["open"].items():
        done = [x for x in by_key.get(k, []) if x["exit_price"] is not None]
        if k not in open_keys and done:          # entry has exits and nothing of it is still open: fully closed
            last = max(done, key=lambda x: x["exit_time"] or "")
            acts.append({"action": "close", "key": k, "run": info["run"], "row": last, "price": last["exit_price"],
                         "size_pct": None})
    return acts


def log(event: str, a: dict, mode: str, note: str = "") -> None:
    r = a["row"]
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLS)
        w.writerow([datetime.now().isoformat(timespec="seconds"), event, r["num"], r["side"], r["entry_time"],
                    r["entry_price"], r["size_btc"], r["size_usd"], r["exit_time"] or "", r["exit_price"] or "",
                    a["run"], a["size_pct"] or "", mode, note])


# ---------------------------------------------------------------- doing it
def apply(acts: list[dict], st: dict, dry: bool, db: str) -> None:
    store = None
    if not dry and acts:
        sys.path.insert(0, str(ROOT))
        from livepaper import claude_trades
        from livepaper.store import Store
        store = Store(ROOT / db)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    for a in acts:
        r = a["row"]
        what = f"{a['action'].upper()} {a['run']} (TV #{r['num']}) @ {a['price']}" + \
               (f" size {a['size_pct']}%" if a["size_pct"] else "")
        if dry:
            print("DRY-RUN would:", what)
            log("dry-run " + a["action"], a, "dry")
            continue
        cmd = {"id": f"tvwatch-{a['action']}-{a['run']}-{stamp}", "run": a["run"], "action": a["action"],
               "price": a["price"], "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "reason": f"TradingView GRIND BTC 30m #{r['num']} entry {r['entry_time']} @ {r['entry_price']}, "
                         f"TV size {r['size_usd']:,.0f} USD" + (f", exit {r['exit_time']} @ {r['exit_price']}"
                                                                if r["exit_price"] else "")}
        if a["size_pct"]:
            cmd["size_pct"] = a["size_pct"]
        claude_trades.poll(store, commands=[cmd])
        if a["action"] == "close":
            st["open"].pop(a["key"], None)
            st["seen"] = (st["seen"] + [a["key"]])[-300:]
        else:
            st["open"][a["key"]] = {"run": a["run"], "num": r["num"]}
        save_state(st)
        log(a["action"], a, "live")
        print("did:", what)


def read_once(args) -> list[dict]:
    png = Path(args.image).read_bytes() if args.image else screenshot_png()
    return parse_rows(ask_model(png, args.model, args.ollama))


def check(args) -> None:
    st = load_state()
    first = read_once(args)
    if not first:
        print("couldn't read the trade list (is TradingView showing it?) - nothing done")
        return
    acts = plan(first, st)
    if acts and not args.image:
        time.sleep(args.confirm_seconds)         # must read the same thing twice before acting
        second = read_once(args)
        if {(a["action"], a["key"]) for a in plan(second, st)} != {(a["action"], a["key"]) for a in acts}:
            print("second read disagreed - waiting for the next check")
            return
    apply(acts, st, args.dry_run, args.db)
    if not acts:
        print(datetime.now().strftime("%H:%M"), "nothing new")


def seed(args) -> None:
    """Tell the watcher which TradingView trades are already open in the paper trader (so it won't re-open them)."""
    st = load_state()
    for spec in args.seed or []:
        side, entry_time, price, run = [s.strip() for s in spec.split(";")]
        st["open"][f"{side.lower()}|{entry_time}|{float(price):.1f}"] = {"run": run, "num": 0}
    save_state(st)
    print("state now:", json.dumps(st["open"], indent=1))


def next_slot(now: datetime, every: int | None) -> datetime:
    if every:
        t = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        while t.minute % every:
            t += timedelta(minutes=1)
        return t
    for m in (1, 31):
        t = now.replace(minute=m, second=0, microsecond=0)
        if t > now:
            return t
    return (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--ollama", default=OLLAMA)
    ap.add_argument("--image", help="parse this screenshot file instead of capturing the screen")
    ap.add_argument("--dry-run", action="store_true", help="print and log what it would do, send nothing")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=None, help="check every N minutes instead of :01 and :31")
    ap.add_argument("--confirm-seconds", type=int, default=45)
    ap.add_argument("--db", default="livepaper.db")
    ap.add_argument("--seed", action="append", metavar="SIDE;ENTRY TIME;PRICE;RUN",
                    help='mark an already-open trade, e.g. --seed "short;Nov 12, 2026, 08:30;72714.5;AUTOGRIND-7"')
    a = ap.parse_args()
    if a.seed:
        seed(a)
        return 0
    while True:
        try:
            check(a)
        except Exception as e:                   # never die on one bad read
            print("check failed:", e)
        if a.once or a.image:
            return 0
        t = next_slot(datetime.now(), a.every)
        print("next check", t.strftime("%H:%M"))
        time.sleep(max(1, (t - datetime.now()).total_seconds()))


if __name__ == "__main__":
    raise SystemExit(main())
