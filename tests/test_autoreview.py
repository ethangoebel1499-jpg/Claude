"""The apprentice reviewing what it learns, by itself."""
import json

import pytest

from apprentice import autoreview, review
from apprentice.store import Store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "LOCAL", tmp_path / "review.json")
    monkeypatch.setattr(review, "APPLIED", tmp_path / "applied.json")
    return Store(tmp_path / "a.db")


def _stats(n, edge, pos=4, of=5):
    charts = {f"c{i}": {"trades": 10, "edge": 0.2 if i < pos else -0.1} for i in range(of)}
    return {"total": {"trades": n, "edge": edge, "avg_r": edge}, "charts": charts}


def test_concepts():
    seen = {}
    j = lambda t, i=1: autoreview.judge_concept(t, seen, i)[0]
    assert j("Place the stop loss below the last swing low.", 1) == "keep"
    assert j("Place the stop loss below the last swing low!", 2) == "drop"           # duplicate
    assert j("Join the Casper Crew VIP discord for more") == "drop"
    assert j("Use 100x leverage to grow a small account") == "drop"
    assert j("and uh you know we broke through this level guys") == "drop"
    assert j("Bitcoin is currently in a bull market and will hit 100k") == "drop"
    assert j("He takes up the bread, this is my body") == "drop"


def test_setups():
    js = lambda s, d=None: autoreview.judge_setup(s, d)[0]
    base = {"direction": "long", "stats": _stats(100, 0.2)}
    assert js({**base, "rules": [{"rule": "bb_touch_lower"}, {"rule": "rsi_bull_div"}]}) == "use"
    assert js({**base, "rules": [{"rule": "wt_green_dot"}, {"rule": "wt_red_dot"}]}) == "drop"       # impossible
    assert js({**base, "rules": [{"rule": "bb_touch_lower"}, {"rule": "rsi_bear_div"}]}) == "drop"   # backwards
    assert js({**base, "rules": [{"rule": "close_above_sma", "a": 45000}, {"rule": "hammer"}]}) == "drop"  # broken
    assert js({**base, "stats": _stats(300, 0.03), "rules": [{"rule": "hammer"}]}) == "drop"         # no edge
    assert js({**base, "stats": _stats(15, 0.4), "rules": [{"rule": "hammer"}]}) == "watch"          # too early
    assert js({**base, "rules": [{"rule": "hammer"}]}, d=3) == "drop"                                # duplicate
    assert autoreview.judge_setup({**base, "stats": {}, "rules": [{"rule": "hammer"}]}, None) is None   # untested


def test_run_applies_rates_and_protects(store):
    good = store.add_lesson("Support & resistance", "video", 10)
    junk = store.add_lesson("🔴 Shabbat Worship September 23rd", "video", 10)
    for t in ("Wait for a hammer at support before going long.", "Join my patreon for the signals"):
        store.add_concept(good, {"t": 1, "category": "entry", "text": t})
    store.add_concept(junk, {"t": 1, "category": "trend", "text": "Everything hinges on what Yeshua finished."})
    store.finish_lesson(good)
    store.finish_lesson(junk)
    s_ok = store.add_setup({"name": "ok", "direction": "long", "rules": [{"rule": "hammer"}, {"rule": "sr_bounce_up"}],
                            "lesson_id": good})
    s_bad = store.add_setup({"name": "bad", "direction": "long", "rules": [{"rule": "wt_green_dot"}, {"rule": "wt_red_dot"}],
                             "lesson_id": good})
    s_pin = store.add_setup({"name": "pinned", "direction": "long", "rules": [{"rule": "hammer"}, {"rule": "rsi_bear_div"}]})
    s_claude = store.add_setup({"name": "claude", "direction": "short", "rules": [{"rule": "shooting_star"}]})
    for sid, st in ((s_ok, _stats(80, 0.25)), (s_bad, _stats(0, 0)), (s_pin, _stats(50, 0.1)), (s_claude, _stats(50, 0.0))):
        store.update_setup(sid, stats=st, status="active")
    store.update_setup(s_pin, status="pinned")
    review.LOCAL.write_text(json.dumps({"reviewed_at": "x", "setups": [{"id": s_claude, "verdict": "use", "reason": "-"}]}))
    c = autoreview.run(store, log=lambda m: None)
    st = {s["id"]: s["status"] for s in store.setups()}
    assert st[s_bad] == "off" and st[s_ok] == "active"
    assert st[s_pin] == "pinned" and st[s_claude] == "active"                 # yours / Claude's: left alone
    assert junk not in set(store.table("lessons").id)                         # the worship service is gone
    assert "patreon" not in " ".join(store.table("concepts").text)
    assert c == {"concepts_deleted": 2, "setups_off": 1, "lessons_deleted": 1}
    v = autoreview.verdict(store, "setups", s_ok)
    assert v["verdict"] == "use" and v["stars"] == 5 and "Holds up" in v["comment"]
    assert autoreview.verdict(store, "setups", s_pin)["comment"].endswith("Claude's review covers it)")
    assert autoreview.verdict(store, "concepts", 1)["stars"] == 4


def test_broken_lengths_and_same_record_duplicates(store):
    js = lambda s, d=None: autoreview.judge_setup(s, d)[0]
    base = {"direction": "long", "stats": _stats(100, 0.2)}
    assert js({**base, "rules": [{"rule": "close_below_sma", "a": 17.566}, {"rule": "hammer"}]}) == "drop"
    assert js({**base, "rules": [{"rule": "close_below_sma", "a": 200.0}, {"rule": "hammer"}]}) == "use"
    ids = []
    for rules in ([{"rule": "rsi_bull_div"}], [{"rule": "rsi_bull_div", "a": 3, "b": 60}]):   # same setup, written twice
        sid = store.add_setup({"name": "div", "direction": "long", "rules": rules})
        store.update_setup(sid, status="active", stats=_stats(53, 0.212))
        ids.append(sid)
    autoreview.run(store, log=lambda m: None)
    st = {s["id"]: s["status"] for s in store.setups()}
    assert sorted(st[i] for i in ids) == ["active", "off"]
