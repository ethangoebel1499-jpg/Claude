"""Background jobs that use the AI (stream agent and apprentice): stop them, and queue them so only one
runs on the local AI at a time.

A job is `python -m streamlearn.job <log> ...` or `python -m apprentice.job <log> ...`, which runs the
real work as a child process. stop(log) finds the job by its log file in the computer's process list
and ends it and everything it started - except the local AI engine (Ollama), which other jobs share.
Works on Windows, Mac and Linux, including for jobs started before this file existed.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "runtime" / "ai_job.lock"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


# ---- processes ------------------------------------------------------------------------------
def processes() -> list[tuple[int, int, str]]:
    """(pid, parent pid, command line) for every process we can see."""
    try:
        if os.name == "nt":
            ps = ("Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine | "
                  "ConvertTo-Json -Compress")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                 timeout=30, creationflags=NO_WINDOW).stdout
            rows = json.loads(out or "[]")
            rows = rows if isinstance(rows, list) else [rows]
            return [(int(r["ProcessId"]), int(r["ParentProcessId"] or 0), r.get("CommandLine") or "") for r in rows]
        if Path("/proc").is_dir():
            out = []
            for p in Path("/proc").iterdir():
                if not p.name.isdigit():
                    continue
                try:
                    cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                    ppid = int((p / "stat").read_text().rsplit(")", 1)[1].split()[1])
                    out.append((int(p.name), ppid, cmd))
                except (OSError, ValueError, IndexError):
                    continue
            return out
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True, timeout=30).stdout
        rows = []
        for line in out.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) >= 2:
                rows.append((int(parts[0]), int(parts[1]), parts[2] if len(parts) > 2 else ""))
        return rows
    except Exception:
        return []


def is_ai_engine(cmd: str) -> bool:
    """Is this the local AI engine (the program itself is named ollama)?"""
    cmd = cmd.strip()
    prog = cmd[1:].split('"', 1)[0] if cmd.startswith('"') else cmd.split(None, 1)[0] if cmd else ""
    return Path(prog.replace("\\", "/")).name.lower().startswith("ollama")


def job_pids(log: Path | str, table=None) -> list[int]:
    """The job's own process and everything under it, except the shared AI engine."""
    table = processes() if table is None else table
    import re
    # only the job wrapper itself: `python -m streamlearn.job <log> ...` / `python -m apprentice.job <log> ...`
    # (the log path may be quoted on Windows)
    me = re.compile(r"-m\s+(?:streamlearn|apprentice)\.job\s+[\"']?" + re.escape(str(log)) + r"[\"']?(?:\s|$)")
    roots = [pid for pid, _, cmd in table if me.search(cmd)]
    kids: dict[int, list[int]] = {}
    for pid, ppid, _ in table:
        kids.setdefault(ppid, []).append(pid)
    cmds = {pid: cmd for pid, _, cmd in table}
    out, todo = [], list(roots)
    while todo:
        pid = todo.pop()
        if pid in out or pid == os.getpid():
            continue
        if is_ai_engine(cmds.get(pid, "")):
            continue                             # leave the AI engine running for other jobs
        out.append(pid)
        todo += kids.get(pid, [])
    return out


def running(log: Path | str) -> bool:
    return bool(job_pids(log))


def _kill(pid: int, hard: bool = False) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, creationflags=NO_WINDOW)
        else:
            os.kill(pid, signal.SIGKILL if hard else signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass


def stop(log: Path | str) -> int:
    """End a job and its work processes (children first, so nothing is left behind). Returns how many."""
    pids = job_pids(log)
    for pid in reversed(pids):
        _kill(pid)
    if os.name != "nt" and pids:
        time.sleep(1)
        left = set(job_pids(log))
        for pid in pids:
            if pid in left:
                _kill(pid, hard=True)
    log = Path(log)
    if pids and log.exists():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\nStopped by you.\n[finished with exit code -1]\n")
    if LOCK.exists():
        try:
            if int(LOCK.read_text().split()[0]) in pids:
                LOCK.unlink()
        except (OSError, ValueError, IndexError):
            pass
    return len(pids)


def delete_log(log: Path) -> None:
    for p in (log, log.with_suffix(".json")):
        try:
            p.unlink()
        except OSError:
            pass


# ---- one AI job at a time -------------------------------------------------------------------
def alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                                          # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_turn(out, poll: float = 5) -> None:
    """Block until no other AI job is running, then hold the turn (released when this process exits)."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    told = False
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time():.0f}".encode())
            os.close(fd)
            return
        except FileExistsError:
            try:
                holder = int(LOCK.read_text().split()[0])
            except (OSError, ValueError, IndexError):
                holder = -1
            if holder == os.getpid():
                return
            if not alive(holder):
                try:
                    LOCK.unlink()
                except OSError:
                    pass
                continue
            if not told:
                out.write("Waiting for the job before it to finish (only one job uses the AI at a time)...\n")
                out.flush()
                told = True
            time.sleep(poll)


def release() -> None:
    try:
        if LOCK.exists() and int(LOCK.read_text().split()[0]) == os.getpid():
            LOCK.unlink()
    except (OSError, ValueError, IndexError):
        pass


def run_queued(module: str, log: Path, args: list[str], queued: bool, env: dict | None = None) -> int:
    """Body of a job wrapper: wait for the AI if needed, run `python -m <module> <args>` into the log."""
    with open(log, "a", encoding="utf-8", errors="replace") as out:
        if queued:
            wait_turn(out)
        try:
            code = subprocess.call([sys.executable, "-u", "-m", module, *args], cwd=ROOT, stdout=out,
                                   stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                                                  **(env or {})})
        finally:
            if queued:
                release()
        out.write(f"\n[finished with exit code {code}]\n")
    return code
