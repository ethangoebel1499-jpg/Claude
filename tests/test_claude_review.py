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


def test_never_shares_from_an_unrelated_git_repo(tmp_path, monkeypatch):
    """The user's home folder was a git repo with no commits; the app folder sat inside it."""
    import subprocess
    home = tmp_path / "home"
    app = home / "Documents" / "Claude-master"
    app.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=home, check=True)
    monkeypatch.chdir(app)
    monkeypatch.setattr(sync, "ROOT", app)
    monkeypatch.setenv("LOCAL_SETTINGS", str(tmp_path / "s.json"))
    from livepaper import settings
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    with pytest.raises(RuntimeError, match="no GitHub token yet"):
        sync.push_files({"a.csv": "1"}, "x")
    assert not (home / ".paper-trades").exists()


def test_concepts_applied_by_themselves_with_a_clear_status(apprentice):
    a, ids = apprentice
    concepts = a.table("concepts")
    good = int(concepts[concepts.text.str.contains("band touch")].id.iloc[0])
    ad = int(concepts[concepts.text.str.contains("subscribe")].id.iloc[0])
    review.LOCAL.write_text(json.dumps({
        "reviewed_at": "2026-11-04T15:00:00Z",
        "setups": [{"id": ids["junk"], "name": "Long: volume spike", "verdict": "drop", "reason": "no edge"}],
        "concepts": [{"id": good, "text": "Wait for a band touch", "verdict": "keep", "reason": "sound"},
                     {"id": ad, "text": "like and subscribe", "verdict": "drop", "reason": "ad"}]}))
    assert review.status(a)["concepts"]["waiting"] == [ad]
    assert review.apply_concepts(a) == 1
    assert "subscribe" not in " ".join(a.table("concepts").text)
    st = review.status(a)
    assert st["concepts"] == {"total": 2, "applied": 2, "waiting": []}
    assert st["setups"]["waiting"] == [ids["junk"]]                     # setups still wait for Apply
    assert review.apply_concepts(a) == 0


def test_concept_verdict_skipped_if_the_id_now_holds_other_text(apprentice):
    a, _ = apprentice
    cid = int(a.table("concepts").id.iloc[0])
    review.LOCAL.write_text(json.dumps({"reviewed_at": "x", "concepts": [
        {"id": cid, "text": "something else entirely", "verdict": "drop", "reason": "-"}]}))
    review.apply_concepts(a)
    assert cid in set(a.table("concepts").id)                           # not deleted: it's a different concept now


def test_downloading_a_review_changes_nothing_until_apply(apprentice, tmp_path, monkeypatch):
    a, ids = apprentice
    concepts = a.table("concepts")
    ad = int(concepts[concepts.text.str.contains("subscribe")].id.iloc[0])
    body = json.dumps({"reviewed_at": "2026-11-04T16:00:00Z",
                       "setups": [{"id": ids["junk"], "name": "Long: volume spike", "verdict": "drop", "reason": "-"}],
                       "concepts": [{"id": ad, "text": "like and subscribe", "verdict": "drop", "reason": "ad"}]})

    class R:
        status_code, text = 200, body
    monkeypatch.setattr(sync.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(sync, "LOCAL_REVIEW", review.LOCAL)
    from livepaper import settings
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    settings.save({"apprentice_auto_apply_claude": False})               # you switched automatic applying off
    assert sync.fetch_reviews()["reviewed_at"] == "2026-11-04T16:00:00Z"
    assert ad in set(a.table("concepts").id)                               # still there after the sync
    assert {s["id"]: s["status"] for s in a.setups()}[ids["junk"]] != "off"
    assert review.status(a)["concepts"]["waiting"] == [ad]
    assert review.apply_all(a) == {"setups": 1, "concepts": 1}             # the button does it
    assert ad not in set(a.table("concepts").id)


def test_claudes_chart_trade_ratings_apply_like_your_own(apprentice):
    a, _ = apprentice
    t = "2026-11-04 16:00:00+00:00"
    review.LOCAL.write_text(json.dumps({"reviewed_at": "2026-11-04T18:00:00Z", "trades": [
        {"id": 1, "kind": "missed", "chart": "BTC 4h", "time": t, "direction": "long", "verdict": "good", "stars": 5,
         "reason": "at support, confirmed"},
        {"id": 2, "kind": "missed", "chart": "BTC 4h", "time": "2026-11-01 00:00:00+00:00", "direction": "short",
         "verdict": "bad", "stars": 1, "reason": "hindsight only"},
        {"id": 3, "kind": "decision", "chart": "BTC 4h", "time": t, "direction": "TRADE SHORT", "setups": [7],
         "stars": 2, "reason": "weak"},
        {"id": 4, "kind": "signal", "chart": "BTC 4h", "time": t, "direction": "long", "setups": [7], "stars": 4,
         "reason": "good"}]}))
    assert review.for_trade("missed", "BTC 4h", t, "long")["stars"] == 5
    assert review.status(a)["trades"] == {"total": 4, "applied": 0, "waiting": 4}
    assert a.table("feedback").empty                                     # nothing until Apply
    assert review.apply_all(a)["trades"] == 4
    from apprentice.__main__ import feedback_marks
    conf, rej = feedback_marks(a)
    assert conf == {"BTC 4h": {t}} and rej == {"BTC 4h": {"2026-11-01 00:00:00+00:00"}}
    assert a.votes(7) == (0, 1)                                          # the call counts, the 400 arrows don't
    assert a.accuracy()["all"]["n"] == 3                                 # arrows kept out of your accuracy average
    assert review.status(a)["trades"]["waiting"] == 0 and review.apply_trades(a) == 0



def test_review_applies_itself_on_sync_by_default(apprentice, tmp_path, monkeypatch):
    a, ids = apprentice
    lid = int(a.table("lessons").id.iloc[0])
    body = json.dumps({"reviewed_at": "2026-11-05T16:00:00Z",
                       "setups": [{"id": ids["junk"], "name": "Long: volume spike", "verdict": "drop", "reason": "-"}],
                       "lessons": [{"id": lid, "verdict": "drop", "reason": "a worship service"}]})

    class R:
        status_code, text = 200, body
    monkeypatch.setattr(sync.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(sync, "LOCAL_REVIEW", review.LOCAL)
    from livepaper import settings
    from apprentice import job
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    spawned = []
    monkeypatch.setattr(job, "spawn", lambda args, title: spawned.append(args))
    monkeypatch.setattr(job, "running", lambda kind: False)
    sync.fetch_reviews()
    assert {s["id"]: s["status"] for s in a.setups()}[ids["junk"]] == "off"
    assert lid not in set(a.table("lessons").id)                           # Claude's "drop" deleted the lesson
    assert spawned and spawned[0][0] == "learn"                            # and it re-tests
