"""Run one `python -m streamlearn ...` command in the background with its output in a log file,
then write how it ended - the dashboard's ➕ Add a stream shows the log while it runs.

  python -m streamlearn.job <logfile> add <transcript.txt> --start "2026-10-28 20:00" --title "..." [--url ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    from streamlearn.jobctl import run_queued
    log, args = Path(sys.argv[1]), sys.argv[2:]
    return run_queued("streamlearn", log, args, queued=bool(args) and args[0] == "add")


if __name__ == "__main__":
    sys.exit(main())
