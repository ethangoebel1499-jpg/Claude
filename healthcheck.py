"""Self-check the app runs before keeping an update (exit code 0 = healthy).

  1. every Python file compiles
  2. the modules the trader, dashboard and stream agent use import
  3. each paper-trading strategy runs on sample candles through the live position logic
"""

from __future__ import annotations

import py_compile
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FOLDERS = ("mcbot", "livepaper", "streamlearn", "apprentice", "tools")
TOP = ("launcher.pyw", "bootstrap.pyw", "updater.py", "healthcheck.py", "main.py")
MODULES = ("mcbot.backtest", "mcbot.patterns", "mcbot.fmt", "livepaper.core", "livepaper.runner", "livepaper.store",
           "livepaper.strategies", "livepaper.feeds", "livepaper.history", "livepaper.brokers", "livepaper.settings",
           "streamlearn.store", "streamlearn.pipeline", "streamlearn.verify", "streamlearn.patterns_ui",
           "streamlearn.ollama_local", "streamlearn.add_ui", "apprentice.store", "apprentice.brain",
           "apprentice.teach", "apprentice.youtube", "apprentice.strategy", "apprentice.ui")


DATA = ("livepaper/data/stocks.csv", "livepaper/static/lightweight-charts.standalone.production.js",
        "mcbot/pattern_stats.json", "assets/icon.ico", "assets/icon_256.png", "app-requirements.txt",
        "livepaper/live.example.toml")


def main() -> int:
    failures = []
    files = [p for f in FOLDERS for p in (ROOT / f).rglob("*.py")] + [ROOT / t for t in TOP if (ROOT / t).exists()]
    for f in files:
        try:
            py_compile.compile(str(f), doraise=True, cfile=None if f.suffix == ".py" else str(f) + "c.tmp")
        except py_compile.PyCompileError as e:
            failures.append(f"{f.relative_to(ROOT)} doesn't compile: {e.msg.strip()[:200]}")
    for tmp in ROOT.glob("*.pywc.tmp"):
        tmp.unlink()
    for need in DATA:                          # files the app reads at run time (not code, so imports don't catch them)
        if not (ROOT / need).exists():
            failures.append(f"missing file: {need}")
    import importlib
    for m in MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:
            failures.append(f"import {m} failed: {type(e).__name__}: {e}")
    if not failures:
        try:
            from livepaper.core import PositionManager, SimBroker, replay
            from livepaper.runner import exits_for
            from livepaper.strategies import REGISTRY
            from mcbot.data import synthetic
            df = synthetic(n=600, seed=3, timeframe="4h")
            for name, spec in REGISTRY.items():
                sig = spec["signals"](df)
                replay(df, sig, PositionManager(name, exits_for({"strategy": name}), shorts=False, broker=SimBroker(0.001)))
                spec["checklist"](df, -1, label=name)
        except Exception:
            failures.append("strategy run failed:\n" + traceback.format_exc()[-600:])
    if failures:
        print("SELF-CHECK FAILED\n" + "\n".join(failures))
        return 1
    print(f"Self-check passed: {len(files)} files, {len(MODULES)} modules, strategies run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
