"""➕ Add a stream: paste the YouTube link and its transcript, press Process. The work (finding the
setups with the AI model, checking them against prices) runs in the background, so you can keep
using the dashboard; progress shows here."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

from streamlearn.transcripts import parse_youtube

ROOT = Path(__file__).resolve().parent.parent
INBOX = ROOT / "streams_inbox"
JOBS = ROOT / "runtime" / "stream_jobs"
DONE = "[finished with exit code"


@st.cache_data(ttl=3600, show_spinner=False)
def youtube_info(url: str) -> dict:
    """Title and start time from the video page (best effort - fill them in by hand if this fails)."""
    import requests
    try:
        page = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en"}).text
    except Exception:
        return {}
    info = {}
    m = re.search(r'<meta name="title" content="([^"]+)"', page) or re.search(r'"title":"([^"]+)"', page)
    if m:
        info["title"] = m.group(1).replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"')
    for key in ("startDate", "uploadDate", "publishDate"):
        m = re.search(rf'"{key}":"([^"]+)"', page) or re.search(rf'itemprop="{key}" content="([^"]+)"', page)
        if m:
            info["start"] = m.group(1)
            break
    return info


def backend_problem() -> str | None:
    from livepaper import settings
    if settings.get("stream_backend") == "claude":
        return None if settings.get("anthropic_api_key") else \
            "The stream agent is set to Claude but there's no Anthropic API key - add it in ⚙️ Account."
    import requests
    url = settings.get("ollama_url") or "http://localhost:11434"
    try:
        requests.get(f"{url}/api/tags", timeout=1.5)
        return None
    except Exception:
        return (f"The local AI model (Ollama) isn't running at {url}. Start the Ollama app, or switch the "
                "stream agent to Claude in ⚙️ Account.")


def start_job(transcript: str, start_local: str, title: str, url: str) -> Path:
    INBOX.mkdir(exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\- ]+", "", title).strip()[:60] or "stream"
    f = INBOX / f"{start_local[:10]} {start_local[11:16].replace(':', '')} {safe}.txt"
    f.write_text(transcript, encoding="utf-8")
    log = JOBS / f"{int(time.time())}.log"
    meta = {"title": title, "file": str(f), "started": datetime.now().isoformat(timespec="seconds")}
    log.with_suffix(".json").write_text(json.dumps(meta))
    cmd = [sys.executable, "-m", "streamlearn.job", str(log), "add", str(f), "--start", start_local,
           "--title", title] + (["--url", url] if url else [])
    subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=0x08000000 if os.name == "nt" else 0)
    return log


def form() -> None:
    from livepaper import settings
    tz = settings.get("timezone") or "UTC"
    url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...", key="add_url").strip()
    info = youtube_info(url) if url.startswith("http") else {}
    start_default = datetime.now(ZoneInfo(tz)).replace(second=0, microsecond=0)
    if info.get("start"):
        try:
            start_default = datetime.fromisoformat(info["start"].replace("Z", "+00:00")).astimezone(ZoneInfo(tz))
        except ValueError:
            pass
    c1, c2, c3 = st.columns([3, 1.2, 1])
    title = c1.text_input("Title", info.get("title", ""), key=f"add_title_{url}")
    day = c2.date_input("Stream date", start_default.date(), key=f"add_day_{url}")
    at = c3.time_input("Start time", start_default.time(), key=f"add_time_{url}", step=60)
    found = "found on YouTube" if info.get("start") else "enter it from the video page"
    st.caption(f"Start time in your timezone ({tz}; change it in ⚙️ Account) - {found}. It lines his words up with "
               "the price chart, so it matters.")
    transcript = st.text_area(
        "Transcript", height=260, key="add_transcript",
        placeholder="On the YouTube video: ... (More) → Show transcript → click in the transcript panel → "
                    "Ctrl+A → Ctrl+C, then paste here (Ctrl+V).")
    segs = parse_youtube(transcript) if transcript.strip() else []
    if transcript.strip():
        if segs:
            st.caption(f"✅ Read {len(segs):,} transcript lines covering {segs[-1][0] // 3600}h "
                       f"{segs[-1][0] % 3600 // 60}m of stream.")
        else:
            st.warning("Couldn't find timestamps in that text - copy the transcript with its timestamps "
                       "(the default in YouTube's transcript panel).")
    problem = backend_problem()
    if problem:
        st.warning(problem)
    ready = bool(segs) and bool(title.strip())
    if st.button("Process stream", type="primary", disabled=not ready, key="add_go"):
        start_local = f"{day:%Y-%m-%d} {at:%H:%M}"
        start_job(transcript, start_local, title.strip(), url)
        st.success("Processing in the background - progress shows below. You can keep using the dashboard.")
    jobs()


@st.fragment(run_every=5)
def jobs() -> None:
    logs = sorted(JOBS.glob("*.log"), reverse=True)[:5] if JOBS.exists() else []
    if not logs:
        return
    st.markdown("**Recent streams**")
    for log in logs:
        meta = json.loads(log.with_suffix(".json").read_text()) if log.with_suffix(".json").exists() else {}
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if DONE in text:
            ok = f"{DONE} 0]" in text
            state = "✅ Done - its setups are in 📝 Review" if ok else "❌ Failed - see the log"
        else:
            state = "⏳ Processing..."
        with st.expander(f"{meta.get('title', log.stem)} · {state}", expanded=DONE not in text):
            lines = [x for x in text.splitlines() if x.strip()]
            st.code("\n".join(lines[-12:]) or "starting...", language=None)
