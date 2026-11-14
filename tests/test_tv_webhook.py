import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import tv_webhook as w  # noqa: E402


def alert(pos, size, price=70000.0, contracts=None, t="2026-11-13T19:30:00Z"):
    return {"position": pos, "size": size, "price": price, "contracts": contracts if contracts is not None else abs(size),
            "time": t}


def st(size=0.0, open_=()):
    return {"size": size, "open": list(open_), "seen": []}


def test_a_new_short_opens_a_run_at_tradingview_price():
    a = w.plan(alert("short", -10, 70000, 10), st(), 1_930_000)
    assert [(x["action"], x["price"]) for x in a] == [("short", 70000.0)] and a[0]["run"] == "TV-1113-1930"
    assert a[0]["size_pct"] == round(10 * 70000 / 1_930_000 * 100) or a[0]["size_pct"] == 1


def test_a_pyramid_add_is_another_run_and_a_partial_exit_is_ignored():
    s = st(-10, [{"run": "TV-1113-1930", "side": "short"}])
    add = w.plan(alert("short", -15, 69000, 5, "2026-11-13T20:00:00Z"), s, 1_930_000)
    assert [x["action"] for x in add] == ["short"] and add[0]["run"] == "TV-1113-2000"
    assert w.plan(alert("short", -5, 68000, 10), st(-15, [{"run": "x", "side": "short"}]), 1_930_000) == []


def test_flat_closes_everything_it_opened():
    s = st(-15, [{"run": "a", "side": "short"}, {"run": "b", "side": "short"}])
    acts = w.plan(alert("flat", 0, 67000, 15), s, 1_930_000)
    assert [(x["action"], x["run"], x["price"]) for x in acts] == [("close", "a", 67000.0), ("close", "b", 67000.0)]


def test_a_reversal_closes_then_opens_the_other_way():
    s = st(-10, [{"run": "a", "side": "short"}])
    acts = w.plan(alert("long", 6, 71000, 16), s, 1_930_000)
    assert [x["action"] for x in acts] == ["close", "long"]


def test_unreadable_alerts_do_nothing_and_a_duplicate_name_gets_a_suffix():
    assert w.plan({"position": "long"}, st(), 1_930_000) == []
    s = st(0, [{"run": "TV-1113-1930", "side": "long"}])
    a = w.plan(alert("long", 3, 70000, 3), s, 1_930_000)
    assert a[0]["run"] == "TV-1113-1930-2"


def test_pyramiding_each_add_is_its_own_run_up_to_the_limit():
    from tools.tv_webhook import plan
    st = {"size": 0.0, "open": [], "seen": []}
    for i in range(1, 9):
        p = {"position": "short", "size": i * 0.5, "contracts": 0.5, "price": 68000 - i, "time": f"2026-11-13T23:{i:02d}:00Z"}
        acts = plan(p, st, 1_930_000)
        assert [a["action"] for a in acts] == ["short"]
        st["open"].append({"run": acts[0]["run"], "side": "short"}); st["size"] = -i * 0.5
    assert len({o["run"] for o in st["open"]}) == 8
    ninth = {"position": "short", "size": 4.5, "contracts": 0.5, "price": 67990, "time": "2026-11-13T23:30:00Z"}
    assert plan(ninth, st, 1_930_000) == []                                 # the chart never goes past 8
    flat = {"position": "flat", "size": 0, "contracts": 4.0, "price": 67000, "time": "2026-11-13T23:40:00Z"}
    assert [a["action"] for a in plan(flat, st, 1_930_000)] == ["close"] * 8     # flat closes every run


def test_a_second_chart_has_its_own_state_runs_symbol_and_timeframe(tmp_path, monkeypatch):
    import argparse
    from livepaper import claude_trades
    from livepaper.store import Store
    import tools.tv_webhook as w
    monkeypatch.setattr(w, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(w, "LOG", tmp_path / "log.csv")
    monkeypatch.setattr(claude_trades, "APPLIED", tmp_path / "applied.json")
    args = argparse.Namespace(dry_run=False, account=1_930_000, max_open=8, db=str(tmp_path / "lp.db"))
    eth = {"chart": "eth1d", "symbol": "ETHUSD", "timeframe": "1d", "position": "long", "size": 10, "contracts": 10,
           "price": 3100.0, "time": "2026-11-13T23:00:00Z"}
    btc = {"position": "short", "size": 1, "contracts": 1, "price": 68700.0, "time": "2026-11-13T23:05:00Z"}
    w.process(eth, w.load_state(w.chart_key(eth)), args)
    w.process(btc, w.load_state(w.chart_key(btc)), args)
    assert w.load_state("ETH1D")["open"][0]["run"].startswith("TV-ETH1D-")
    assert w.load_state()["open"][0]["run"].startswith("TV-1113-")           # the BTC chart is untouched
    runs = {r["name"]: r for r in Store(tmp_path / "lp.db").runs()}
    eth_run = next(r for n, r in runs.items() if n.startswith("TV-ETH1D-"))
    assert eth_run["symbol"] == "ETHUSD" and eth_run["timeframe"] == "1d"
    flat_btc = {"position": "flat", "size": 0, "contracts": 1, "price": 68000.0, "time": "2026-11-13T23:30:00Z"}
    w.process(flat_btc, w.load_state(), args)
    assert w.load_state()["open"] == [] and len(w.load_state("ETH1D")["open"]) == 1   # flat on BTC leaves ETH open
