import json

import numpy as np
import pandas as pd
import pytest

from apprentice import brain, teach, youtube
from apprentice.store import Store
from mcbot import builder
from mcbot.data import synthetic


@pytest.fixture(scope="module")
def df():
    return synthetic(n=1500, seed=5, timeframe="4h")


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "a.db")


LESSON = """0:05
today the falling wedge, I go long when it breaks out with big volume
1:30
I only go long on a bounce off support when there's a hammer, that's the key
2:10
never short into support, make sure you wait for the candle close
"""


def test_youtube_links():
    assert youtube.kind("https://www.youtube.com/playlist?list=PLabc123") == "playlist"
    assert youtube.kind("https://www.youtube.com/watch?v=abcdefghijk&list=PLabc123") == "playlist"
    assert youtube.kind("https://www.youtube.com/watch?v=abcdefghijk&list=RDabcdefghijk") == "video"   # auto-mix
    assert youtube.kind("https://youtu.be/abcdefghijk") == "video"
    assert youtube.kind("https://www.youtube.com/live/abcdefghijk") == "video"
    assert youtube.kind("https://example.com/watch?v=x") is None
    assert youtube.video_id("https://youtu.be/abcdefghijk?t=3") == "abcdefghijk"
    assert youtube.video_id("https://www.youtube.com/watch?v=abcdefghijk&list=PL1") == "abcdefghijk"


def test_playlist_learns_every_video_and_skips_known(store, monkeypatch):
    from apprentice import __main__ as cli
    vids = [{"id": f"v{i}", "title": f"Lesson {i}", "url": f"https://www.youtube.com/watch?v=v{i}", "playlist": "P"}
            for i in range(3)]
    monkeypatch.setattr(youtube, "videos", lambda url: vids)

    def caps(url):
        if url.endswith("v2"):
            raise RuntimeError("no English captions")
        return {"title": url[-2:]}, teach.segments(LESSON)
    monkeypatch.setattr(youtube, "captions", caps)
    url = "https://www.youtube.com/playlist?list=PLx"
    got = cli.teach_source(store, url, None, None, log=lambda *a: None)
    assert got["lessons"] == 2 and got["failed"] == 1 and got["concepts"] > 0
    again = cli.teach_source(store, url, None, None, log=lambda *a: None)
    assert again["skipped"] == 2 and again["lessons"] == 0          # already learned


def test_json3_captions():
    data = {"events": [{"tStartMs": 0, "segs": [{"utf8": "hello"}]}, {"tStartMs": 1500, "segs": [{"utf8": "there"}]},
                       {"tStartMs": 9000, "segs": [{"utf8": "next line"}]}]}
    assert youtube.parse_json3(json.dumps(data)) == [(0, "hello there"), (9, "next line")]


def test_keyword_teaching_makes_checkable_setups(store):
    lid = store.add_lesson("Wedges", "x", 3)
    got = teach.learn_lesson(store, lid, teach.segments(LESSON), "Wedges", llm=None, log=lambda *a: None)
    assert got["setups"] >= 1 and got["concepts"] >= 1
    for s in store.setups():
        assert all(r["rule"] in builder.RULES for r in s["rules"])


def test_ai_answer_is_cleaned(store):
    class Fake:
        def ask(self, system, user, schema):
            assert "pattern_break_up" in system            # it knows the rules it can check
            return {"concepts": [{"time": "0:10", "category": "entry", "text": "Wait for the close", "quote": "q"}],
                    "setups": [{"time": "0:20", "name": "Wedge", "direction": "long",
                                "rules": [{"rule": "pattern_break_up", "a": None, "b": None},
                                          {"rule": "made_up_rule", "a": 1, "b": None}],
                                "unchecked": ["only after a pullback"], "stop_atr": None, "rr": 3, "timeframe": "4h",
                                "why": "", "quote": ""},
                               {"time": "0:30", "name": "dup", "direction": "long",
                                "rules": [{"rule": "pattern_break_up", "a": None, "b": None}], "unchecked": [],
                                "stop_atr": None, "rr": None, "timeframe": "", "why": "", "quote": ""}]}
    lid = store.add_lesson("L", "x", 1)
    teach.learn_lesson(store, lid, [(0, "x")], "L", llm=Fake(), log=lambda *a: None)
    (s,) = store.setups()
    assert s["rules"] == [{"rule": "pattern_break_up"}] and s["rr"] == 3 and s["unchecked"] == ["only after a pullback"]


def test_new_rules_run_without_lookahead(df):
    for k in ("bull_engulf", "hammer", "sr_bounce_up", "pattern_break_up", "in_bullish_pattern", "squeeze",
              "higher_low"):
        full = builder.rule_series(df, {"rule": k})
        cut = builder.rule_series(df.iloc[:1200].copy(), {"rule": k})
        assert (full.iloc[:1200].to_numpy() == cut.to_numpy()).all(), k


def test_pattern_break_candle_keeps_its_pattern(df):
    pf = builder.pattern_frame(df)
    broke = pf[pf.broke != ""]
    assert len(broke) and (broke.kind != "").all()


def test_outcomes_hit_target_or_stop():
    idx = pd.date_range("2024", periods=40, freq="4h", tz="UTC")
    c = np.r_[np.full(20, 100.0), np.linspace(100, 130, 20)]
    d = pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1.0}, index=idx)
    r = brain.outcomes(d, "long", 1.5, 2.0, horizon=15)
    assert r[20] == 2.0                       # straight up: target first
    assert brain.outcomes(d, "short", 1.5, 2.0, horizon=15)[20] == -1.0


def test_learn_scores_and_rates(store, df):
    sid = store.add_setup({"name": "EMA cross", "direction": "long", "rules": [{"rule": "ema_cross_up"}]})
    res = brain.learn(store, {"syn": df}, log=lambda *a: None)
    assert res[0]["trades"] > 0
    s = store.setups()[0]
    assert s["status"] in ("active", "candidate", "retired") and "total" in s["stats"]
    w0 = s["weight"]
    store.rate("setup", sid, 1, "wrong: he wants the 4h close above the 200")
    store.rate("setup", sid, 1, "")
    brain.learn(store, {"syn": df}, log=lambda *a: None)
    assert store.setups()[0]["weight"] < w0                # bad ratings lower its trust


def test_ratings_accuracy_and_feedback_to_ai(store):
    store.rate("concept", 1, 5, "")
    store.rate("decision", "BTC 4h|t", 2, "price was at resistance")
    store.rate("setup", 7, 1, "he never said volume", {"name": "Volume break", "rules_text": ["Volume spike"]})
    acc = store.accuracy()
    assert acc["all"]["n"] == 3 and acc["all"]["avg"] == pytest.approx(8 / 3, 0.01)
    assert acc["concept"]["accurate_pct"] == 100
    notes = teach.rating_notes(store)
    assert "he never said volume" in notes and "Volume break" in notes
    assert store.votes(7) == (0, 1)


def test_decision_ratings_count_for_the_setups_that_fired(store):
    store.rate("decision", "x", 5, "", {"setups": [3]})
    assert store.votes(3) == (1, 0)


def test_missed_trades_and_mining(df):
    m = brain.missed_trades(df, [])
    assert m and all(x["move_atr"] > 0 for x in m)
    cands = brain.mine({"syn": df}, "long")
    for c in cands:
        assert c["train"]["trades"] >= brain.MIN_TRADES and "held_up" in c


def test_decide_and_strategy(df):
    from apprentice import strategy
    s = {"id": 1, "name": "Always", "direction": "long", "rules": [{"rule": "rsi_below", "a": 101}],
         "status": "active", "weight": 0.3, "stop_atr": 1.5, "rr": 2, "stats": {}}
    d = brain.decide(df, [s])
    assert d["action"] == "TRADE LONG" and d["stop"] < d["entry"] < d["target"]
    sig = strategy.signals(df, [s])
    assert sig.long_entry.any() and not sig.short_entry.any()
    assert brain.decide(df, [])["action"] == "WAIT"


def test_registry_has_apprentice():
    from livepaper.strategies import REGISTRY
    assert "apprentice" in REGISTRY


def test_awesome_oscillator_matches_tradingview(df):
    # the Pine script, written out by hand
    hl2 = (df.high + df.low) / 2
    ao = hl2.rolling(5).mean() - hl2.rolling(34).mean()
    diff = ao - ao.shift()
    green = diff > 0                                           # color = diff <= 0 ? red : green
    to_green = (diff > 0) & (diff.shift() <= 0)                # ta.crossover(diff, 0)
    to_red = (diff < 0) & (diff.shift() >= 0)                  # ta.crossunder(diff, 0)
    from mcbot.indicators import awesome_oscillator
    pd.testing.assert_series_equal(awesome_oscillator(df), ao, check_names=False)
    rule = lambda k: builder.rule_series(df, {"rule": k})
    assert (rule("ao_green") == green.fillna(False)).all()
    assert (rule("ao_red") == (diff <= 0).fillna(False)).all()
    assert (rule("ao_turns_green") == to_green.fillna(False)).all() and rule("ao_turns_green").sum() > 10
    assert (rule("ao_turns_red") == to_red.fillna(False)).all() and rule("ao_turns_red").sum() > 10
    assert (rule("ao_above_zero") == (ao > 0)).all() and (rule("ao_cross_up_zero") <= rule("ao_above_zero")).all()
    assert "ao_turns_green" in teach.system()                  # the AI can use it when reading lessons
    lesson = teach.guess("[1:00] I go long when the awesome oscillator turns green at support, bounce off support")
    assert any(r["rule"] == "ao_turns_green" for s in lesson["setups"] for r in s["rules"])
