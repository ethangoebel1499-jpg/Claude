"""The shared Kraken feed: one download per market, and no 30-second wait per run when Kraken stops answering."""
import requests

from livepaper import feeds


class FakeHTTP:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if self.fail:
            raise requests.Timeout("read timed out")

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"error": [], "result": {"XXBTZUSD": {"c": ["75000.0"]}, "last": 1}}
        return R()


def test_same_request_is_downloaded_once(monkeypatch):
    monkeypatch.setattr(feeds.KrakenFeed, "GAP_S", 0)
    f = feeds.KrakenFeed(FakeHTTP())
    assert [f.price("XBTUSD") for _ in range(5)] == [75000.0] * 5
    assert f.http.calls == 1


def test_a_timeout_pauses_kraken_instead_of_every_run_waiting(monkeypatch):
    import pytest
    monkeypatch.setattr(feeds.KrakenFeed, "GAP_S", 0)
    f = feeds.KrakenFeed(FakeHTTP(fail=True))
    for sym in ("XBTUSD", "ETHUSD", "SOLUSD"):
        with pytest.raises(RuntimeError):
            f.price(sym)
    assert f.http.calls == 1                       # the other runs skip straight away
