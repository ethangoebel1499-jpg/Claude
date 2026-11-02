"""Run one `python -m apprentice ...` command in the background with its output in a log file
(the 🧠 Apprentice tab shows the log while it runs).

  python -m apprentice.job <logfile> teach <link or file> [...]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "runtime" / "apprentice_jobs"
DONE = "[finished with exit code"


def spawn(args: list[str], title: str) -> Path:
    """Start a job; returns its log file."""
    JOBS.mkdir(parents=True, exist_ok=True)
    log = JOBS / f"{int(time.time() * 1000)}.log"
    log.with_suffix(".json").write_text(json.dumps({"title": title, "args": args,
                                                    "started": datetime.now().isoformat(timespec="seconds")}))
    subprocess.Popen([sys.executable, "-m", "apprentice.job", str(log), *args], cwd=ROOT, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=0x08000000 if os.name == "nt" else 0)
    return log


def main() -> int:
    log, args = Path(sys.argv[1]), sys.argv[2:]
    with open(log, "a", encoding="utf-8", errors="replace") as out:
        code = subprocess.call([sys.executable, "-u", "-m", "apprentice", *args], cwd=ROOT, stdout=out,
                               stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        out.write(f"\n{DONE} {code}]\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
