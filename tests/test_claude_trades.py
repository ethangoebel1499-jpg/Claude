"""Trades Claude sends through GitHub get paper traded by the open app."""
from livepaper import claude_trades
from livepaper.store import Store


def test_claudes_trades_are_queued_once(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_trades, "APPLIED", tmp_path / "applied.json")
    s = Store(tmp_path / "lp.db")
    cmds = [{"id": "tv-1", "run": "AUTOGRIND", "symbol": "XBTUSD", "timeframe": "30m", "action": "short",
             "reason": "TradingView GRIND short", "at": "2026-11-08T01:20"}]
    assert claude_trades.poll(s, cmds) == 1
    run = next(r for r in s.runs() if r["name"] == "AUTOGRIND")
    assert run["strategy"] == "autogrind" and run["timeframe"] == "30m" and run["active"]
    assert run["stop"] == 0.09 and run["take_profits"][0] == [0.017, 0.33]       # the script's exits
    assert claude_trades.poll(s, cmds) == 0                                       # never twice
    assert claude_trades.poll(s, cmds + [{**cmds[0], "id": "tv-2", "action": "close"}]) == 1
    assert [a for _, a in s.pending_commands("AUTOGRIND")] == ["short", "close"]


def test_the_old_self_trading_grind_runs_are_retired(tmp_path, monkeypatch):
    from mcbot.data import synthetic
    monkeypatch.setattr(claude_trades, "APPLIED", tmp_path / "applied.json")
    s = Store(tmp_path / "lp.db")
    for name in claude_trades.RETIRED:
        s.upsert_run({**claude_trades.run_spec("AUTOGRIND"), "name": name, "strategy": "claude_grind"})
    cmds = [{"id": "old-1", "run": "Claude GRIND BTC 30m", "action": "short", "reason": "-"}]
    assert claude_trades.poll(s, cmds) == 1
    runs = {r["name"]: r for r in s.runs()}
    assert not any(runs[n]["active"] for n in claude_trades.RETIRED)
    assert [a for _, a in s.pending_commands("AUTOGRIND")] == ["short"]           # redirected to AUTOGRIND
    sig = claude_trades.registry_entry()["signals"](synthetic(n=300, seed=1, timeframe="30m"))
    assert not sig[["long_entry", "short_entry"]].to_numpy().any()                # it no longer trades by itself


def test_autogrind_takes_a_percent_of_the_account(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_trades, "APPLIED", tmp_path / "applied.json")
    s = Store(tmp_path / "lp.db")
    cmds = [{"id": "a1", "run": "AUTOGRIND", "symbol": "XBTUSD", "timeframe": "30m", "action": "long", "size_pct": 52,
             "reason": "TradingView long", "at": "2026-11-08T01:31"},
            {"id": "a2", "run": "AUTOGRIND", "action": "close", "reason": "TradingView closed", "at": "2026-11-08T03:01"},
            {"id": "a3", "run": "AUTOGRIND", "action": "long", "size_pct": 250, "reason": "bad size", "at": "x"}]
    assert claude_trades.poll(s, cmds) == 3
    run = next(r for r in s.runs() if r["name"] == "AUTOGRIND")
    assert run["strategy"] == "autogrind" and run["stop"] == 0.09
    ids = [i for i, _ in s.pending_commands("AUTOGRIND")]
    assert [s.command_size(i) for i in ids] == [52.0, None, None]          # out-of-range sizes are ignored


def test_autogrind_has_no_signals_of_its_own():
    from mcbot.data import synthetic
    e = claude_trades.autogrind_entry()
    sig = e["signals"](synthetic(n=300, seed=1, timeframe="30m"))
    assert not sig[["long_entry", "short_entry"]].to_numpy().any()
    from livepaper.strategies import REGISTRY, refresh
    refresh()
    assert "autogrind" in REGISTRY


def test_a_run_with_an_unknown_strategy_doesnt_stop_the_trader(tmp_path, monkeypatch):
    import pytest
    from livepaper import runner
    with pytest.raises(ValueError):
        runner.normalize_run({"name": "x", "strategy": "no_such_strategy", "symbol": "XBTUSD"})
    cfg = tmp_path / "live.toml"
    cfg.write_text('[[run]]\nname = "x"\nstrategy = "no_such_strategy"\nsymbol = "XBTUSD"\n')
    assert runner.load_config(cfg)["run"] == []                                   # skipped, not a crash


def test_the_claude_poll_doesnt_clobber_the_poll_seconds():
    import inspect
    from livepaper import runner
    src = inspect.getsource(runner.run_forever)
    assert "import poll\n" not in src and "time.sleep(poll)" in src
