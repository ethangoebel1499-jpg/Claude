"""Your own strategies (built from rules in the dashboard's 🛠️ Builder) and your names for any strategy.
Saved on this computer in custom_strategies.json (never pushed to GitHub).

refresh() puts them into livepaper.strategies.REGISTRY, so Strategy Lab, backtests, charts and paper
trading all treat a custom strategy like a built-in one. The trader refreshes on every price check, so
a new or edited strategy is picked up without a restart.
"""

from __future__ import annotations

import json
import os
import re
from functools import partial
from pathlib import Path

from mcbot import builder

from .core import Exits

ROOT = Path(__file__).resolve().parent.parent
PATH = Path(os.environ.get("CUSTOM_STRATEGIES", ROOT / "custom_strategies.json"))
PREFIX = "custom:"
_original_labels: dict[str, tuple[str, str]] = {}


def load() -> dict:
    try:
        data = json.loads(PATH.read_text()) if PATH.exists() else {}
    except json.JSONDecodeError:
        data = {}
    data.setdefault("strategies", {})
    data.setdefault("names", {})
    return data


def save(data: dict) -> None:
    PATH.write_text(json.dumps(data, indent=2))


def slug(name: str) -> str:
    return PREFIX + (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "strategy")


def save_strategy(spec: dict, key: str | None = None) -> str:
    """Add (key=None) or update a strategy; returns its key. Renaming keeps the key, so paper-trading
    runs on it keep working."""
    data = load()
    if key is None:
        key, n = slug(spec["name"]), 2
        while key in data["strategies"]:
            key = f"{slug(spec['name'])}-{n}"
            n += 1
    data["strategies"][key] = spec
    save(data)
    return key


def delete_strategy(key: str) -> None:
    data = load()
    data["strategies"].pop(key, None)
    save(data)


def rename(key: str, name: str) -> None:
    """Your own display name for any strategy (built-in ones too); empty = back to the original."""
    data = load()
    if key.startswith(PREFIX) and key in data["strategies"]:
        data["strategies"][key]["name"] = name.strip() or data["strategies"][key]["name"]
    elif name.strip():
        data["names"][key] = name.strip()
    else:
        data["names"].pop(key, None)
    save(data)


def entry(key: str, spec: dict) -> dict:
    tps = [(None if lv is None else float(lv), float(sz)) for lv, sz in spec.get("take_profits") or [[None, 1.0]]]
    rules = [builder.describe(r) for r in spec.get("long", [])]
    short = [builder.describe(r) for r in spec.get("short", [])]
    note = "Long when: " + " AND ".join(rules) if rules else "No long rules yet."
    if short:
        note += " · Short when: " + " AND ".join(short)
    name = spec.get("name", key)
    return {"label": f"🛠️ {name}", "short": name[:24], "custom": True, "spec": spec,
            "signals": partial(builder.signals, spec=spec),
            "checklist": lambda df, bar=-1, label=None, _s=spec: builder.checklist(df, bar, label, _s),
            "exits": Exits(spec.get("stop") or None, tps), "note": note, "long_only": not spec.get("short")}


def refresh(registry: dict) -> dict:
    """Bring REGISTRY up to date with the saved custom strategies and names."""
    data = load()
    for k in [k for k in registry if k.startswith(PREFIX) and k not in data["strategies"]]:
        del registry[k]
    for k, spec in data["strategies"].items():
        try:
            registry[k] = entry(k, spec)
        except Exception:
            registry.pop(k, None)
    for k, v in registry.items():                  # your names for built-in strategies
        if k.startswith(PREFIX):
            continue
        _original_labels.setdefault(k, (v["label"], v["short"]))
        label, short = _original_labels[k]
        if data["names"].get(k):
            v["label"], v["short"] = data["names"][k], data["names"][k][:24]
        else:
            v["label"], v["short"] = label, short
    return registry
