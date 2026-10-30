import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from streamlearn import transcripts
from streamlearn.pipeline import add_stream, from_filename, parse_start
from streamlearn.playbook import group_rules, playbook_markdown, scorecard, scorecard_markdown
from streamlearn.store import Store
from streamlearn.verify import check, pair

FIX = Path(__file__).parent / "fixtures" / "stream_sample.txt"


class FakeLLM:
    """Stands in for Ollama / Claude: returns what a good model would extract."""
    def extract(self, chunk, title, date):
        out = {"calls": [], "rules": [], "levels": []}
        if "sixty four thousand" in chunk:
            out["calls"].append({"time": "[1:06]", "asset": "btc", "direction": "long", "kind": "entered",
                                 "timeframe": "4h", "entry": 64200, "stop": 62000, "targets": [67000],
                                 "conditions": [], "quote": "i'm long bitcoin from sixty four thousand two hundred"})
            out["levels"].append({"time": "1:13", "asset": "BTC", "price": 62000, "kind": "support"})
        if "ethereum looks weak" in chunk:
            out["calls"].append({"time": "3:30", "asset": "ETH", "direction": "short", "kind": "plan",
                                 "timeframe": "", "entry": 2400, "stop": 2550, "targets": [2200],
                                 "conditions": ["lose 2400"], "quote": "if we lose two thousand four hundred i'll short it"})
        if "level" in chunk:
            out["rules"].append({"time": "2:05", "category": "entry", "rule": "Only enter a trade when price is at a level.",
                                 "quote": "i only enter a trade if price comes to a level"})
        if "swing high" in chunk:
            out["rules"].append({"time": "9:15", "category": "entry", "rule": "Only enter trades when price is at a level",
                                 "quote": "only take trades at your levels"})
            out["rules"].append({"time": "9:15", "category": "stop", "rule": "Stop goes above the last swing high.",
                                 "quote": "put the stop above the last swing high"})
        return out


def test_parse_youtube_layouts():
    segs = transcripts.load(FIX)
    assert segs[0] == (0, "all right y'all welcome back to the live stream we got a lot to cover")
    assert (66, "i'm long bitcoin from sixty four thousand two hundred") in segs and segs[-1][0] == 555
    alone = transcripts.parse_youtube("Intro\n0:00\nhello there\n1:02:03\nlater on\n")
    assert alone == [(0, "hello there"), (3723, "later on")]
    subs = transcripts.parse_subtitles("WEBVTT\n\n00:01:05.500 --> 00:01:07.000\nlong btc\n")
    assert subs == [(65, "long btc")]


def test_chunks_overlap_and_timestamps():
    segs = transcripts.load(FIX)
    parts = transcripts.chunks(segs, minutes=8, overlap=1)
    assert len(parts) == 2 and parts[0][2].startswith("[0:00]") and "[9:15]" in parts[1][2]


def test_filename_and_timezone():
    start, title = from_filename(Path("2026-08-20 2000 Bitcoin live.txt"), "America/New_York")
    assert title == "Bitcoin live" and str(start) == "2026-08-21 00:00:00+00:00"
    assert str(parse_start("2026-08-20 20:00", "UTC")) == "2026-08-20 20:00:00+00:00"
    assert pair("btc") == "XBTUSD" and pair("ETH/USDT") == "ETHUSD" and pair("doge") == "XDGUSD"


def candles(path, start, prices):
    idx = pd.date_range(start, periods=len(prices), freq="1h", tz="UTC")
    return pd.DataFrame({"open": prices, "high": [p * 1.002 for p in prices],
                         "low": [p * 0.998 for p in prices], "close": prices}, index=idx)


def test_check_outcomes():
    c = candles(None, "2026-01-01", [100] * 3 + [101, 103, 105, 107, 110] + [110] * 200)
    win = check({"id": 1, "said_at": "2026-01-01 00:30+00:00", "direction": "long", "entry": None,
                 "stop": 95, "targets": "[108]"}, c)
    assert win["result"] == "win" and win["r_multiple"] == pytest.approx(1.6, abs=0.05)
    loss = check({"id": 2, "said_at": "2026-01-01 00:30+00:00", "direction": "short", "entry": None,
                  "stop": 104, "targets": "[90]"}, c)
    assert loss["result"] == "loss" and loss["ret_pct"] < 0
    never = check({"id": 3, "said_at": "2026-01-01 00:30+00:00", "direction": "long", "entry": 80,
                   "stop": 75, "targets": "[90]"}, c, now=pd.Timestamp("2026-02-01", tz="UTC"))
    assert never["status"] == "entry never hit"
    direc = check({"id": 4, "said_at": "2026-01-01 00:30+00:00", "direction": "long", "entry": None,
                   "stop": None, "targets": "[]"}, c, now=pd.Timestamp("2026-02-01", tz="UTC"))
    assert direc["status"] == "directional" and direc["result"] == "right"


def test_full_pipeline_with_real_candles(tmp_path, monkeypatch):
    from streamlearn import verify
    store = Store(tmp_path / "s.db")
    start = parse_start("2026-08-20 16:00", "UTC")
    sid = add_stream(store, FIX, start, "Sample stream", {"chunk_minutes": 8}, llm=FakeLLM(), log=lambda *a: None)
    calls = store.table("calls")
    assert len(calls) == 2                         # the BTC call caught by both chunks is kept once
    assert set(calls.asset) == {"BTC", "ETH"} and json.loads(calls.targets[0]) == [67000.0]
    assert calls.said_at[0] == "2026-08-20 16:01:06+00:00"
    assert add_stream(store, FIX, start, "Sample stream", {}, llm=FakeLLM(), log=lambda *a: None) == sid
    monkeypatch.setattr(verify.Candles, "_api", lambda self, p, since: (_ for _ in ()).throw(RuntimeError("offline")))
    res = verify.verify_all(store)
    assert len(res) == 2 and not res.status.str.startswith("error").any()   # checked on the repo's Kraken data
    sc = scorecard(store.calls_with_streams())
    assert sc["overall"]["calls"].iloc[0] == 2 and "Call scorecard" in scorecard_markdown(sc)
    groups = group_rules(store.table("rules"), store.table("streams"))
    entry = [g for g in groups if g["category"] == "entry"]
    assert len(entry) == 1 and entry[0]["mentions"] == 2  # near-identical rules grouped
    assert "## Stop" in playbook_markdown(groups, 1)


def test_dashboard_renders(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    store = Store(tmp_path / "d.db")
    add_stream(store, FIX, parse_start("2026-08-20 16:00", "UTC"), "Sample", {}, llm=FakeLLM(), log=lambda *a: None)
    monkeypatch.setenv("STREAMLEARN_DB", str(tmp_path / "d.db"))
    at = AppTest.from_file(str(Path(__file__).parent.parent / "streamlearn" / "dashboard.py"), default_timeout=60)
    at.run()
    assert not at.exception and at.title[0].value.startswith("🎙️")


def test_push_files_to_branch(tmp_path, monkeypatch):
    from livepaper.sync import push_files
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    run = lambda *a, cwd: subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)
    run("init", "-q", "--bare", "origin.git", cwd=tmp_path)
    run("clone", "-q", "origin.git", "work", cwd=tmp_path)
    work = tmp_path / "work"
    run("commit", "-q", "--allow-empty", "-m", "init", cwd=work)
    run("push", "-q", "origin", "HEAD:master", cwd=work)
    monkeypatch.chdir(work)
    assert push_files({"a.csv": "x\n1\n"}, "stream_study") == "pushed"
    assert push_files({"a.csv": "x\n1\n"}, "stream_study") == "nothing new"
    shown = subprocess.run(["git", "show", "paper-trades:stream_study/a.csv"], cwd=tmp_path / "origin.git",
                           capture_output=True, text=True).stdout
    assert shown == "x\n1\n"


def test_ollama_backend_request(monkeypatch):
    from streamlearn import extract
    sent = {}

    class Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": json.dumps({"calls": [], "rules": [], "levels": []})}}

    def post(url, timeout, json):
        sent.update(url=url, body=json)
        return Resp()

    monkeypatch.setattr(extract.requests, "post", post)
    out = extract.OllamaBackend("qwen2.5:14b").extract("[0:00] hi", "T", "2026-08-20")
    assert out == {"calls": [], "rules": [], "levels": []}
    assert sent["url"].endswith("/api/chat") and sent["body"]["format"] == extract.SCHEMA
    assert sent["body"]["stream"] is False and sent["body"]["options"]["temperature"] == 0


def test_claude_backend_request():
    from types import SimpleNamespace
    from streamlearn import extract
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                               content=[SimpleNamespace(type="text", text='{"calls": [], "rules": [], "levels": []}')])

    b = extract.ClaudeBackend.__new__(extract.ClaudeBackend)
    b.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    b.model, b.effort = "claude-opus-5-5", "medium"
    assert b.extract("[0:00] hi", "T", "2026-08-20")["calls"] == []
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": extract.SCHEMA}
    assert "thinking" not in kw and "temperature" not in kw
