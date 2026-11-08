"""Trades Claude sends through GitHub get paper traded by the open app."""
from livepaper import claude_trades
from livepaper.store import Store


def test_claudes_trades_are_queued_once(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_trades, "APPLIED", tmp_path / "applied.json")
    s = Store(tmp_path / "lp.db")
    cmds = [{"id": "grind-BTC-30m-short-2026-11-06 14:30", "run": "Claude GRIND BTC 30m", "symbol": "XBTUSD",
             "timeframe": "30m", "action": "short", "reason": "following GRIND's open short", "at": "2026-11-08T01:20"}]
    assert claude_trades.poll(s, cmds) == 1
    run = next(r for r in s.runs() if r["name"] == "Claude GRIND BTC 30m")
    assert run["strategy"] == "claude_grind" and run["timeframe"] == "30m" and run["active"]
    assert run["stop"] == 0.09 and run["take_profits"][0] == [0.017, 0.33]       # the script's exits
    assert [a for _, a in s.pending_commands("Claude GRIND BTC 30m")] == ["short"]
    assert claude_trades.poll(s, cmds) == 0                                       # never twice
    assert claude_trades.poll(s, cmds + [{**cmds[0], "id": "x2", "action": "long"}]) == 1
    assert [a for _, a in s.pending_commands("Claude GRIND BTC 30m")] == ["short", "long"]


def test_the_run_trades_grind_on_every_candle():
    from mcbot.data import synthetic
    from mcbot.grind import grind_signals
    e = claude_trades.registry_entry()
    df = synthetic(n=1500, seed=1, timeframe="30m")
    sig = e["signals"](df)
    assert sig.equals(grind_signals(df))                       # the script itself, on every closed candle
    assert "every closed candle" in e["checklist"](df)
    from livepaper.strategies import REGISTRY, refresh
    refresh()
    assert "claude_grind" in REGISTRY


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
