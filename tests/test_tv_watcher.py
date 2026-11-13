import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import tv_watcher as w

REPLY = '''Here you go:
[{"num":2120,"side":"Short","entry_time":"Nov 12, 2026, 10:30","entry_price":71690.2,"size_btc":29.31,"size_usd":2100000,"exit_time":null,"exit_price":null},
 {"num":2117,"side":"Short","entry_time":"Nov 12, 2026, 08:30","entry_price":72714.5,"size_btc":18.59,"size_usd":1350000,"exit_time":null,"exit_price":null},
 {"num":2116,"side":"Short","entry_time":"Nov 12, 2026, 08:30","entry_price":72714.5,"size_btc":9.16,"size_usd":665850,"exit_time":"Nov 12, 2026, 09:44","exit_price":71478.4},
 {"num":9,"side":"Sideways","entry_time":"x","entry_price":5}]'''


def test_parse_drops_bad_rows():
    rows = w.parse_rows(REPLY)
    assert [r["num"] for r in rows] == [2120, 2117, 2116]


def test_partial_exit_is_ignored_and_new_entry_opens_with_tv_price_and_size():
    rows = w.parse_rows(REPLY)
    st = {"open": {"short|Nov 12, 2026, 08:30|72714.5": {"run": "AUTOGRIND-7", "num": 2117}}, "seen": []}
    acts = w.plan(rows, st)
    assert [(a["action"], a["run"], a["price"], a["size_pct"]) for a in acts] == [("short", "TV-2120", 71690.2, 100)]


def test_full_exit_closes_the_run_even_after_renumbering():
    rows = w.parse_rows('[{"num":2121,"side":"Short","entry_time":"Nov 12, 2026, 08:30","entry_price":72714.5,'
                        '"size_btc":18.59,"size_usd":1350000,"exit_time":"Nov 12, 2026, 17:00","exit_price":71000}]')
    st = {"open": {"short|Nov 12, 2026, 08:30|72714.5": {"run": "AUTOGRIND-7", "num": 2117}}, "seen": []}
    acts = w.plan(rows, st)
    assert [(a["action"], a["run"], a["price"]) for a in acts] == [("close", "AUTOGRIND-7", 71000.0)]


def test_nothing_new_when_everything_is_tracked():
    rows = w.parse_rows(REPLY)
    st = {"open": {w.key(r): {"run": "x", "num": 0} for r in rows if r["exit_price"] is None}, "seen": []}
    assert w.plan(rows, st) == []


def test_live_apply_queues_a_priced_command(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "STATE", tmp_path / "s.json")
    monkeypatch.setattr(w, "LOG", tmp_path / "log.csv")
    from livepaper import claude_trades
    monkeypatch.setattr(claude_trades, "APPLIED", tmp_path / "applied.json")
    monkeypatch.setattr(w, "ROOT", tmp_path)
    rows = w.parse_rows(REPLY)
    st = {"open": {}, "seen": []}
    acts = [a for a in w.plan(rows, st) if a["run"] == "TV-2120"]
    w.apply(acts, st, False, "t.db")
    from livepaper.store import Store
    s = Store(tmp_path / "t.db")
    cid, action = s.pending_commands("TV-2120")[0]
    assert action == "short" and s.command_price(cid) == 71690.2 and s.command_size(cid) == 100
    r = [r for r in s.runs() if r["name"] == "TV-2120"][0]
    assert r["stop"] == 0.09 and [tuple(t) for t in r["take_profits"]] == [(0.017, 0.33), (0.023, 0.33), (0.034, 0.34)]
    assert (tmp_path / "log.csv").exists() and "short|Nov 12, 2026, 10:30|71690.2" in st["open"]


def test_exit_before_entry_is_a_misread_and_is_dropped():
    rows = w.parse_rows('[{"num":2120,"side":"Short","entry_time":"Nov 12, 2026, 10:30","entry_price":71690.2,'
                        '"size_btc":29.3,"size_usd":2100000,"exit_time":"Nov 12, 2026, 09:44","exit_price":71478.4}]')
    assert rows == []


def test_old_rows_misread_as_open_never_open_a_trade():
    st = {"open": {"short|Nov 12, 2026, 08:30|72714.5": {"run": "AUTOGRIND-7", "num": 0}}, "seen": []}
    rows = [{"num": 2115, "side": "long", "entry_time": "Nov 11, 2026, 16:30", "entry_price": 74492.6,
             "size_btc": 1, "size_usd": 1, "exit_time": None, "exit_price": None}]
    assert w.plan(rows, st) == []
    rows[0]["entry_time"] = "Nov 12, 2026, 11:00"
    assert [a["action"] for a in w.plan(rows, st)] == ["long"]


def test_open_row_with_open_text_as_exit_time_still_counts_as_open():
    rows = w.parse_rows('''[{"num": 2140, "side": "Short", "entry_time": "Nov 13, 2026, 19:30", "entry_price": 70000.0,
        "size_btc": 1, "size_usd": 70000, "exit_time": "Open", "exit_price": null}]''')
    assert len(rows) == 1 and rows[0]["exit_time"] is None


def test_iso_dates_are_converted_to_tradingview_format():
    rows = w.parse_rows('''[{"num": 498, "side": "Short", "entry_time": "2026-11-08T17:00", "entry_price": 75155.2,
        "size_btc": 1, "size_usd": 1, "exit_time": "2026-11-09T01:30", "exit_price": 73877.6}]''')
    assert rows[0]["entry_time"] == "Nov 8, 2026, 17:00" and rows[0]["exit_time"] == "Nov 9, 2026, 01:30"
