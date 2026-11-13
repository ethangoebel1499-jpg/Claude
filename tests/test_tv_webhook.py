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
