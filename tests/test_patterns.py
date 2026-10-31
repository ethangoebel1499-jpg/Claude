import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcbot.patterns import Detector, history_note, situation
from streamlearn.pipeline import add_stream, parse_start
from streamlearn.store import Store
from tests.test_streamlearn import FIX, FakeLLM


def zig(points, per=6):
    xs = []
    for a, b in zip(points[:-1], points[1:]):
        xs += list(np.linspace(a, b, per, endpoint=False))
    c = np.array(xs + [points[-1]], float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "close": c, "high": np.maximum(o, c) + 0.3, "low": np.minimum(o, c) - 0.3,
                         "volume": 1.0}, index=pd.date_range("2026-01-01", periods=len(c), freq="4h", tz="UTC"))


LEAD = list(np.linspace(140, 121, 8))
TRIANGLE = [120, 100, 116, 100, 112, 100, 108, 100, 104.5]


def test_descending_triangle_found_and_followed_to_target():
    df = zig(LEAD + TRIANGLE + [101.5])
    s = situation(df)
    assert s["pattern"].kind == "descending_triangle" and s["pattern"].bias == "down"
    assert s["key"] in ("breakout_zone", "near_apex", "at_bottom")
    assert s["pattern"].apex > len(df) - 1                     # the lines meet ahead of price
    assert "CLOSE below" in s["text"] and "e+" not in s["text"]
    broken = zig(LEAD + TRIANGLE + [101, 96, 92, 88, 84, 79])
    s2 = situation(broken)
    assert s2["broke"] == "down" and s2["key"] == "target_hit"


def test_squeeze_at_the_apex():
    df = zig(LEAD + TRIANGLE + [101, 103.5, 100.6, 102.6, 101.2, 102, 101.5], per=6)
    det = Detector(df)
    keys = {situation(df, e, det)["key"] for e in range(len(df) - 12, len(df)) if situation(df, e, det)}
    assert keys & {"near_apex", "past_apex"}


def test_no_look_ahead():
    df = zig(LEAD + TRIANGLE + [101, 96, 92, 88, 84])
    for end in range(40, len(df)):
        full, cut = situation(df, end), situation(df.iloc[: end + 1])
        assert (full is None) == (cut is None)
        if full:
            assert (full["pattern"].kind, full["key"]) == (cut["pattern"].kind, cut["key"])


def test_double_top():
    df = zig([70, 80, 100, 90, 100.2, 93], per=10)
    s = situation(df)
    assert s and s["pattern"].kind == "double_top" and s["lower"] == pytest.approx(89.7)


def test_history_note_uses_real_stats():
    note = history_note("descending_triangle", "near_apex")
    assert note.startswith("History:") and "apex" in note


def test_stream_patterns_extracted_checked_and_shown(tmp_path, monkeypatch):
    from streamlearn import verify
    store = Store(tmp_path / "p.db")
    add_stream(store, FIX, parse_start("2026-08-20 16:00", "UTC"), "S", {}, llm=FakeLLM(), log=lambda *a: None)
    pats = store.table("patterns")
    assert len(pats) == 1 and pats.asset[0] == "ETH" and pats.top[0] == 2550 and json.loads(pats.targets[0]) == [2200]
    monkeypatch.setattr(verify.Candles, "_api", lambda self, p, since: (_ for _ in ()).throw(RuntimeError("offline")))
    verify.verify_all(store)
    out = store.table("patterns").outcome[0]
    assert out[0] in "🟢🔴⚪" and "Broke" in out or "break" in out
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("STREAMLEARN_DB", str(tmp_path / "p.db"))
    monkeypatch.setenv("STREAMLEARN_STANDALONE", "1")
    at = AppTest.from_file(str(Path(__file__).parent.parent / "streamlearn" / "dashboard.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("What he said" in m.value for m in at.markdown)
    assert any("Every situation" in e.label for e in at.expander)


def test_probability_range():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
    from pattern_stats import wilson
    p, lo, hi = wilson(50, 100)
    assert p == 50 and lo == pytest.approx(40.4, abs=0.1) and hi == pytest.approx(59.6, abs=0.1)
    assert wilson(630, 1000)[1] > 59          # 1,000 breaks pins 63% down to about 60-66%


def test_s9_buys_the_wedge_break_after_an_uptrend():
    from mcbot import s9
    # uptrend, then a falling wedge (both lines falling, top line steeper), then a break up
    pts = list(np.linspace(60, 120, 10)) + [120, 104, 114, 103, 109.5, 102.4, 106.5] + [110, 115, 120, 126, 132]
    df = zig(pts)
    fr = s9.signals(df)
    assert fr.long_entry.sum() == 1
    i = int(np.flatnonzero(fr.long_entry.to_numpy())[0])
    assert fr.long_stop.iloc[i] < df.close.iloc[i] < fr.long_tp1.iloc[i]
    r = s9.backtest(df, fr)
    assert r.stats["trades"] == 1 and r.trades[0].reason.startswith("target")
    assert "LONG:" in s9.checklist(df, i)
