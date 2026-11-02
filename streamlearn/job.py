"""Run one `python -m streamlearn ...` command in the background with its output in a log file,
then write how it ended - the dashboard's ➕ Add a stream shows the log while it runs.

  python -m streamlearn.job <logfile> add <transcript.txt> --start "2026-10-28 20:00" --title "..." [--url ...]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    log, args = Path(sys.argv[1]), sys.argv[2:]
    with open(log, "a", encoding="utf-8", errors="replace") as out:
        code = subprocess.call([sys.executable, "-u", "-m", "streamlearn", *args], cwd=ROOT, stdout=out,
                               stderr=subprocess.STDOUT)
        out.write(f"\n[finished with exit code {code}]\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
