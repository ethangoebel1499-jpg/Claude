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
    from streamlearn.jobctl import run_queued
    log, args = Path(sys.argv[1]), sys.argv[2:]
    # reading lessons uses the AI: one at a time. Tests on prices (learn) don't, so they run straight away.
    return run_queued("apprentice", log, args, queued=bool(args) and args[0] in ("teach", "import-streams")
                      and "--no-ai" not in args, env={"APPRENTICE_JOB": log.stem})


if __name__ == "__main__":
    sys.exit(main())
