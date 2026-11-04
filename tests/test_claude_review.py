"""Sharing the apprentice's knowledge with Claude, and applying Claude's review."""
import json

import pytest

from apprentice import review
from apprentice.store import Store as AStore
from livepaper import sync


@pytest.fixture
def apprentice(tmp_path, monkeypatch):
    monkeypatch.setenv("APPRENTICE_DB", str(tmp_path / "a.db"))
    monkeypatch.setattr(review, "LOCAL", tmp_path / "review.json")
    monkeypatch.setattr(review, "APPLIED", tmp_path / "applied.json")
    a = AStore(tmp_path / "a.db")
    lid = a.add_lesson("Bollinger + divergence", "video", 100)
    a.add_concept(lid, {"t": 30, "category": "entry", "text": "Wait for a band touch, then an oscillator divergence",
                        "quote": "wait for price to touch the band"})
    a.add_concept(lid, {"t": 900, "category": "other", "text": "like and subscribe, check out the patreon",
                        "quote": "make sure to subscribe"})
    backwards = a.add_setup({"name": "bb_touch_lower", "direction": "long", "lesson_id": lid,
                             "rules": [{"rule": "close_below_ema", "a": 50}]})
    good = a.add_setup({"name": "Short: upper band + shooting star", "direction": "short", "source": "missed",
                        "rules": [{"rule": "bb_above_upper", "a": 20, "b": 2}]})
    junk = a.add_setup({"name": "Long: volume spike", "direction": "long", "source": "missed",
                        "rules": [{"rule": "volume_spike"}]})
    return a, {"backwards": backwards, "good": good, "junk": junk}


def test_export_contains_what_claude_reviews(apprentice):
    files = sync.export_all(None)
    setups = json.loads(files["apprentice/setups.json"])
    assert {s["name"] for s in setups} == {"bb_touch_lower", "Short: upper band + shooting star", "Long: volume spike"}
    assert setups[0]["rules_text"] == ["Close below EMA (50)"]
    assert "like and subscribe" in files["apprentice/concepts.csv"]
    assert "3 setups" in files["SUMMARY.md"]


def test_applying_claudes_review(apprentice):
    a, ids = apprentice
    concepts = a.table("concepts")
    ad = int(concepts[concepts.text.str.contains("subscribe")].id.iloc[0])
    review.LOCAL.write_text(json.dumps({
        "reviewed_at": "2026-11-04T12:00:00Z", "summary": "Two keepers, one backwards.",
        "setups": [
            {"id": ids["good"], "name": "Short: upper band + shooting star", "verdict": "use", "reason": "holds up"},
            {"id": ids["junk"], "name": "Long: volume spike", "verdict": "drop", "reason": "no edge out of sample"},
            {"id": ids["backwards"], "name": "bb_touch_lower", "verdict": "fix", "reason": "needs a bullish divergence",
             "fix": {"rules": [{"rule": "bb_below_lower", "a": 20, "b": 2}], "stop_atr": 1.5}},
            {"id": 999, "name": "gone", "verdict": "drop", "reason": "-"}],
        "concepts": [{"id": ad, "verdict": "drop", "reason": "it's the video's ad"}]}))
    by = {s["id"]: s for s in a.setups()}
    assert review.describe(review.for_setup(by[ids["junk"]])).startswith("❌ Claude: **DROP**")
    assert review.for_setup({**by[ids["junk"]], "name": "something else"}) is None   # id reused: not applied
    n = review.apply_all(a)
    assert n == {"setups": 3, "concepts": 1}
    by = {s["id"]: s for s in a.setups()}
    assert by[ids["good"]]["status"] == "pinned" and by[ids["junk"]]["status"] == "off"
    assert by[ids["backwards"]]["status"] == "candidate" and by[ids["backwards"]]["rules"][0]["rule"] == "bb_below_lower"
    assert "subscribe" not in " ".join(a.table("concepts").text)
    assert review.apply_all(a) == {"setups": 0, "concepts": 0}               # applied once


def test_github_api_push_is_one_commit_of_changed_files(monkeypatch):
    calls, blobs = [], {}
    tree = {"paper_trades/trades.csv": sync._blob_sha("same\n")}

    class R:
        def __init__(self, code, js): self.status_code, self._js, self.text = code, js, json.dumps(js)
        def json(self): return self._js

    def request(method, url, headers=None, json=None, timeout=None):
        path = url.split("/repos/ethangoebel1499-jpg/Claude", 1)[1]
        calls.append((method, path))
        assert headers["Authorization"] == "Bearer tok"
        if path == "/git/ref/heads/paper-trades":
            return R(404, {}) if not blobs.get("branch") else R(200, {"object": {"sha": "head"}})
        if path == "/git/ref/heads/master":
            return R(200, {"object": {"sha": "m1"}})
        if path == "/git/refs":
            blobs["branch"] = True
            return R(201, {})
        if path.startswith("/git/commits/"):
            return R(200, {"tree": {"sha": "t1"}})
        if path.startswith("/git/trees/t1"):
            return R(200, {"tree": [{"path": p, "sha": s} for p, s in tree.items()]})
        if path == "/git/blobs":
            return R(201, {"sha": f"b{len(calls)}"})
        if path == "/git/trees":
            assert [e["path"] for e in json["tree"]] == ["apprentice/setups.json"] and json["base_tree"] == "t1"
            return R(201, {"sha": "t2"})
        if path == "/git/commits":
            return R(201, {"sha": "c2"})
        if path == "/git/refs/heads/paper-trades":
            return R(200, {})
        raise AssertionError(path)
    monkeypatch.setattr(sync.requests, "request", request)
    gh = sync.GitHub("tok")
    out = gh.push({"paper_trades/trades.csv": "same\n", "apprentice/setups.json": "[]"}, "snapshot")
    assert out == "pushed 1 file(s)"
    assert ("POST", "/git/refs") in calls                                     # branch created on first sync
    assert sum(1 for m, p in calls if (m, p) == ("POST", "/git/blobs")) == 1  # only the changed file uploaded
    assert gh.push({"paper_trades/trades.csv": "same\n"}, "snapshot") == "nothing new"
