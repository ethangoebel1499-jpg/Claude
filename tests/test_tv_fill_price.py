"""A trade sent by TradingView (webhook) is filled at TradingView's price, not the live market price."""
from livepaper import claude_trades
from livepaper.core import Exits, PositionManager, SimBroker
from livepaper.runner import Runner
from livepaper.store import Store


def _runner(tmp_path, source, market=68096.5):
    store = Store(tmp_path / "lp.db")
    broker = SimBroker(fee=0, fill="market", market_price=lambda run: market)
    r = {**claude_trades.run_spec("TV-1"), "source": source, "feed": "kraken", "qty_step": 0.000001}
    ru = object.__new__(Runner)
    ru.r, ru.store, ru.price = r, store, market
    ru.pm = PositionManager("TV-1", Exits(0.09, [(0.017, 0.33), (0.023, 0.33), (0.034, 0.34)]), shorts=True,
                            broker=broker, store=store, fraction=1.0, qty_step=0.000001, start_equity=10000.0)
    return ru, store, broker


def _send(store, action, price):
    store.command("TV-1", action, None, price)


def test_tradingview_runs_fill_at_the_price_tradingview_sent(tmp_path):
    ru, store, broker = _runner(tmp_path, "claude")
    _send(store, "short", 68700.0)
    ru.manual("2026-11-13T23:40:00+00:00")
    assert ru.pm.state.position.entry_price == 68700.0
    assert broker.force_price is None                                # only for that one order
    _send(store, "close", 68000.0)
    ru.manual("2026-11-13T23:50:00+00:00")
    assert ru.pm.trades[-1]["exit_price"] == 68000.0


def test_other_runs_still_fill_at_the_market(tmp_path):
    ru, store, _ = _runner(tmp_path, "live.toml")
    _send(store, "short", 68700.0)
    ru.manual("2026-11-13T23:40:00+00:00")
    assert ru.pm.state.position.entry_price == 68096.5
