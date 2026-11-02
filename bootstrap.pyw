"""What the Paper Trader shortcut opens. It runs the app (launcher.pyw) and keeps an eye on it:

  exit code 0  -> you closed the app: done
  exit code 3  -> the app updated itself: open it again on the new version
  anything else while an update is still being checked -> the new version is broken: put the
                  previous version back (updater.rollback) and open that instead
  anything else otherwise -> show where the error log is

Kept deliberately tiny so an update can't easily break the thing that undoes updates.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LOG = ROOT / "runtime" / "launcher.log"
RESTART = 3


def launch() -> int:
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8", errors="replace") as out:
        out.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} starting\n")
        out.flush()
        env = {**os.environ, "PAPERTRADER_BOOTSTRAP": "1"}
        return subprocess.call([sys.executable, str(ROOT / "launcher.pyw")], cwd=ROOT, stdout=out, stderr=out,
                               env=env, creationflags=0x08000000 if os.name == "nt" else 0)


def note(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8", errors="replace") as out:
        out.write(f"{time.strftime('%H:%M:%S')} bootstrap: {msg}\n")


def tell(msg: str) -> None:
    try:
        import tkinter
        from tkinter import messagebox
        r = tkinter.Tk()
        r.withdraw()
        messagebox.showwarning("Paper Trader", msg)
        r.destroy()
    except Exception:
        print(msg)


def run(launch=launch, sleep=time.sleep) -> int:
    starts = []
    while True:
        starts = [t for t in starts if time.time() - t < 120] + [time.time()]
        if len(starts) > 5:                       # something keeps failing: stop instead of looping
            tell(f"Paper Trader keeps closing unexpectedly. The error log is in:\n{LOG}")
            return 1
        code = launch()
        if code == 0:
            return 0
        if code == RESTART:
            continue
        import updater
        if updater.pending():
            res = updater.rollback(note, reason=f"the new version crashed (exit code {code})")
            if res["ok"]:
                if res.get("deps_changed"):
                    updater.install_deps(sys.executable.replace("pythonw", "python"), note)
                continue
        tell(f"Paper Trader closed unexpectedly (exit code {code}). The error log is in:\n{LOG}")
        return code


if __name__ == "__main__":
    sys.exit(run())
