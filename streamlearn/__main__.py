"""streamlearn - study livestreams: extract calls and rules, check the calls, build a playbook.

  python -m streamlearn add <file> --start "2026-10-28 20:00" [--title "..."] [--url <YouTube link>] [--reextract]
  python -m streamlearn watch <folder>     process each new file dropped in the folder
                                           (name it "2026-10-28 2000 Title.txt", time in your timezone)
  python -m streamlearn verify             check every call against real prices
  python -m streamlearn report             write the playbook + scorecard (markdown)
  python -m streamlearn list               streams and counts
  python -m streamlearn dashboard          open the study dashboard in your browser
  python -m streamlearn sync               push calls, rules, playbook and scorecard to the paper-trades branch

Inputs: YouTube "Show transcript" copy-paste (.txt), subtitles (.srt/.vtt), or audio/video
(transcribed locally with faster-whisper). Settings: streamlearn.toml (see streamlearn/streamlearn.example.toml).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from streamlearn import transcripts  # noqa: E402
from streamlearn.pipeline import add_stream, from_filename, parse_start  # noqa: E402
from streamlearn.playbook import group_rules, playbook_markdown, scorecard, scorecard_markdown  # noqa: E402
from streamlearn.store import Store  # noqa: E402
from streamlearn.verify import verify_all  # noqa: E402


def load_cfg(path: str) -> dict:
    """streamlearn.toml, overridden by what was saved in the dashboard's Account tab."""
    from livepaper import settings
    cfg = {}
    if Path(path).exists():
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    saved = settings.saved()
    for ours, theirs in (("stream_backend", "backend"), ("ollama_model", "ollama_model"), ("ollama_url", "ollama_url"),
                         ("timezone", "timezone"), ("streamlearn_db", "db")):
        if saved.get(ours):
            cfg[theirs] = saved[ours]
    if settings.get("anthropic_api_key"):
        cfg["anthropic_api_key"] = settings.get("anthropic_api_key")
    return cfg


def write_reports(store: Store, out: Path) -> tuple[str, str]:
    streams = store.table("streams")
    pb = playbook_markdown(group_rules(store.table("rules"), streams), len(streams))
    sc = scorecard_markdown(scorecard(store.calls_with_streams()))
    out.mkdir(parents=True, exist_ok=True)
    (out / "playbook.md").write_text(pb)
    (out / "scorecard.md").write_text(sc)
    return pb, sc


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m streamlearn", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["add", "watch", "verify", "report", "list", "dashboard", "sync"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--start", help='when the stream started, e.g. "2026-10-28 20:00" (your timezone)')
    ap.add_argument("--title")
    ap.add_argument("--reextract", action="store_true")
    ap.add_argument("--url", help="the stream's YouTube link, so each setup can play its part of the stream")
    ap.add_argument("--config", default="streamlearn.toml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    tz = cfg.get("timezone", "UTC")
    db = cfg.get("db", "streamlearn.db")
    out = Path(cfg.get("reports", "streamlearn_reports"))

    if a.command == "dashboard":
        env = {**os.environ, "STREAMLEARN_DB": db, "STREAMLEARN_STANDALONE": "1"}
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(HERE / "dashboard.py")], env=env))
    store = Store(db)

    if a.command == "add":
        if not a.path:
            raise SystemExit("add needs a file")
        p = Path(a.path)
        guess = from_filename(p, tz)
        if a.start:
            start = parse_start(a.start, tz)
        elif guess:
            start = guess[0]
        else:
            raise SystemExit('give --start "YYYY-MM-DD HH:MM" (or name the file "2026-10-28 2000 Title.txt")')
        title = a.title or (guess[1] if guess else p.stem)
        add_stream(store, p, start, title, cfg, reextract=a.reextract, url=a.url)
        verify_all(store, only_unchecked=True)
        write_reports(store, out)
        print(f"done -> {out}/playbook.md, {out}/scorecard.md")

    elif a.command == "watch":
        folder = Path(a.path or "streams_inbox")
        folder.mkdir(exist_ok=True)
        done = folder / "processed"
        done.mkdir(exist_ok=True)
        print(f"watching {folder}/ - drop files named like '2026-10-28 2000 Title.txt'. Ctrl+C to stop.")
        while True:
            for p in sorted(folder.iterdir()):
                if not p.is_file() or p.name.startswith("."):
                    continue
                if p.suffix.lower() not in transcripts.MEDIA | {".txt", ".srt", ".vtt"}:
                    continue
                guess = from_filename(p, tz)
                if not guess:
                    print(f"skipping {p.name}: name it 'YYYY-MM-DD HHMM Title{p.suffix}'")
                    continue
                try:
                    add_stream(store, p, guess[0], guess[1], cfg)
                    p.rename(done / p.name)
                except Exception as e:
                    print(f"{p.name} failed: {e}")
            verify_all(store, only_unchecked=True)
            write_reports(store, out)
            time.sleep(cfg.get("watch_seconds", 60))

    elif a.command == "verify":
        res = verify_all(store)
        print(res.status.value_counts().to_string() if len(res) else "no calls yet")
        write_reports(store, out)

    elif a.command == "report":
        pb, sc = write_reports(store, out)
        print(sc)
        print(f"\nplaybook: {out}/playbook.md")

    elif a.command == "list":
        s = store.table("streams")
        for t in ("calls", "rules", "levels"):
            counts = store.table(t).groupby("stream_id").size()
            s[t] = s.id.map(counts).fillna(0).astype(int)
        print(s[["id", "start", "title", "duration_s", "calls", "rules", "levels"]].to_string(index=False)
              if len(s) else "no streams yet")

    elif a.command == "sync":
        from livepaper.sync import push_files
        write_reports(store, out)
        files = {"calls.csv": store.calls_with_streams().to_csv(index=False),
                 "rules.csv": store.table("rules").to_csv(index=False),
                 "levels.csv": store.table("levels").to_csv(index=False),
                 "streams.csv": store.table("streams").to_csv(index=False),
                 "playbook.md": (out / "playbook.md").read_text(), "scorecard.md": (out / "scorecard.md").read_text()}
        print(push_files(files, "stream_study"))


if __name__ == "__main__":
    main()
