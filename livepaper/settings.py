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
    # Webull paper (sandbox) account
    "webull_app_key": "", "webull_app_secret": "", "webull_account_id": "", "webull_region": "us",
    "webull_endpoint": "api.sandbox.webull.com",
    # stream study agent
    "streamlearn_db": "streamlearn.db", "stream_backend": "ollama", "ollama_model": "qwen2.5:14b",
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
