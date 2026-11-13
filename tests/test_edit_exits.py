"""Editing an open trade by hand: move the stop, lock in profit, change the targets."""
import pandas as pd

from livepaper.core import Exits, PositionManager, SimBroker
from livepaper.store import Store


def _open(side="long"):
    m = PositionManager("t", Exits(0.05, [(0.02, 0.5), (0.04, 0.5)]), shorts=True, broker=SimBroker(fee=0))
    m.open(side, 100.0, "2026-01-01T00:00:00+00:00")
    return m


def test_stop_to_entry_then_stops_out_at_entry_for_a_long():
    m = _open()
    assert "0.00% from the entry" in m.set_stop(0.0, 101.0, "t1")
    m.on_tick(101, 99.5, 101, "t2")
    assert m.trades[-1]["reason"] == "breakeven" and abs(m.trades[-1]["exit_price"] - 100.0) < 1e-9


def test_lock_profit_puts_the_stop_above_a_long_and_below_a_short():
    m = _open()
    m.set_stop(0.008, 101.5, "t1")
    m.on_tick(101.5, 100.5, 101.5, "t2")
    assert m.trades[-1]["reason"] == "trail stop" and abs(m.trades[-1]["exit_price"] - 100.8) < 1e-9
    s = _open("short")
    s.set_stop(0.008, 98.5, "t1")
    s.on_tick(98.5, 98.5, 99.5, "t2")                       # bounces up through 99.2
    assert abs(s.trades[-1]["exit_price"] - 99.2) < 1e-9


def test_refuses_a_stop_the_price_is_already_past():
    m = _open()
    assert m.set_stop(0.008, 100.2, "t1").startswith("refused")      # only +0.2% up: a +0.8% stop would fill at once
    assert m.state.position.stop_move is None


def test_new_targets_are_used_and_validated():
    m = _open()
    assert m.set_targets([0.03, 0.02], "t1").startswith("refused")      # not nearest to farthest
    assert m.set_targets([0.01], "t1").startswith("refused")            # wrong number of targets
    assert m.set_targets([0.01, 0.03], "t1").startswith("targets now")
    m.on_tick(100, 100, 101.2, "t2")
    assert m.state.position.targets_hit == 1


def test_command_value_round_trip(tmp_path):
    st = Store(str(tmp_path / "x.db"))
    st.command("r", "stop_lock", value={"lock": 0.008})
    cid = st.pending_commands("r")[0][0]
    assert st.command_value(cid) == {"lock": 0.008}
