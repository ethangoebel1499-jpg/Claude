"""Local settings edited in the dashboard's Account tab (local_settings.json in the repo root).

Paper trading only, so values are stored in plain text - but the file is git-ignored so it
never gets pushed to GitHub. Environment variables still win if set (e.g. WEBULL_APP_KEY).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = Path(os.environ.get("LOCAL_SETTINGS", ROOT / "local_settings.json"))

DEFAULTS = {
    # paper trading
    "broker": "sim", "fee": 0.001, "poll_seconds": 30, "sync_git": True, "sync_minutes": 60,
    "max_stop": 0.10,
    "github_token": "",    # lets the app share its data with Claude (and fetch Claude's reviews) from a ZIP install      # safety cap: every trade has a stop, never wider than this
    # Webull paper (sandbox) account
    "webull_app_key": "", "webull_app_secret": "", "webull_account_id": "", "webull_region": "us",
    "webull_endpoint": "api.sandbox.webull.com",
    # crypto runs with the Webull broker: "auto" = try Webull, switch to the built-in paper broker if the account
    # can't trade crypto; "sim" = always the built-in paper broker; "webull" = always Webull
    "webull_crypto": "auto",
    # stream study agent
    "streamlearn_db": "streamlearn.db", "apprentice_db": "apprentice.db", "apprentice_reversal_exit": "learned", "apprentice_auto_review": True, "apprentice_auto_apply_claude": True,
    "proving_on": True, "claude_trades_on": True, "proving_max_trials": 10, "proving_auto_start": True, "proving_scan_hours": 24,
    "proving_markets": ["BTC:crypto", "ETH:crypto", "SOL:crypto", "SPY:stock"], "proving_timeframes": [], "stream_backend": "ollama", "ollama_model": "auto", "stream_screen": True, "ollama_vision_model": "auto",
    "ollama_url": "http://localhost:11434", "anthropic_api_key": "", "timezone": "America/New_York",
}
ENV = {"webull_app_key": "WEBULL_APP_KEY", "webull_app_secret": "WEBULL_APP_SECRET",
       "webull_account_id": "WEBULL_ACCOUNT_ID", "anthropic_api_key": "ANTHROPIC_API_KEY"}


def load() -> dict:
    data = dict(DEFAULTS)
    if PATH.exists():
        try:
            data.update(json.loads(PATH.read_text()))
        except json.JSONDecodeError:
            pass
    return data


def saved() -> dict:
    """Only what the user actually saved (no defaults)."""
    if PATH.exists():
        try:
            return json.loads(PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save(values: dict) -> None:
    data = load()
    data.update(values)
    PATH.write_text(json.dumps(data, indent=2))


def get(key: str, default=None):
    env = ENV.get(key)
    if env and os.environ.get(env):
        return os.environ[env]
    v = load().get(key)
    return default if v in (None, "") else v


def mask(secret: str) -> str:
    return "" if not secret else f"{'•' * 8}{secret[-4:]}"
