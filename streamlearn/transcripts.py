"""Turn a stream into timestamped text segments [(seconds, text), ...].

Accepted inputs:
  - YouTube "Show transcript" copy-pastes, either layout:
        0:07                      or     0:000 secondsall right y'all...
        all right y'all...               1:061 minute, 6 secondsi possibly can...
    ("Chapter 3: ..." headings and "Sync to video time" lines are skipped)
  - subtitle files: .srt / .vtt
  - audio or video files: transcribed locally with faster-whisper (pip install faster-whisper)
"""

from __future__ import annotations

import re
from pathlib import Path

TS = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})"
# "1 hour, 2 minutes, 3 seconds" as YouTube glues it after the timestamp
SPOKEN = r"(?:\d+\s+hours?,?\s*)?(?:\d+\s+minutes?,?\s*)?(?:\d+\s+seconds?)?"
GLUED = re.compile(rf"^{TS}({SPOKEN})(.*)$")
ALONE = re.compile(rf"^{TS}$")
SKIP = re.compile(r"^(chapter \d+:|sync to video time$|\[music\]$)", re.I)
MEDIA = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac", ".mp4", ".mkv", ".webm", ".mov"}


def _secs(h, m, s) -> int:
    return int(h or 0) * 3600 + int(m) * 60 + int(s)


def _spoken_secs(text: str) -> int | None:
    if not text.strip():
        return None
    h = re.search(r"(\d+)\s+hour", text)
    m = re.search(r"(\d+)\s+minute", text)
    s = re.search(r"(\d+)\s+second", text)
    return (int(h.group(1)) if h else 0) * 3600 + (int(m.group(1)) if m else 0) * 60 + (int(s.group(1)) if s else 0)


def _split_glued(line: str) -> tuple[int, str] | None:
    """'1:061 minute, 6 secondsi possibly' -> (66, 'i possibly'). The spoken duration
    repeats the timestamp, which is how we know where the digits end."""
    m = re.match(rf"^{TS}", line)
    if not m:
        return None
    # The minutes/seconds digits may run into the spoken part ("0:000 seconds"): try each split.
    head_end = m.end()
    for cut in range(head_end, max(head_end - 3, 0), -1):
        mm = re.match(rf"^{TS}$", line[:cut])
        if not mm:
            continue
        rest = line[cut:]
        sm = re.match(rf"^({SPOKEN})(.*)$", rest)
        spoken = sm.group(1) if sm else ""
        t = _secs(*mm.groups())
        if spoken and _spoken_secs(spoken) == t:
            return t, sm.group(2).strip()
    return None


def parse_youtube(text: str) -> list[tuple[int, str]]:
    segs: list[tuple[int, str]] = []
    pending: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or SKIP.match(line):
            continue
        if ALONE.match(line):
            pending = _secs(*ALONE.match(line).groups())
            continue
        glued = _split_glued(line)
        if glued:
            segs.append(glued)
            pending = None
        elif pending is not None:
            segs.append((pending, line))
            pending = None
        elif segs:                         # continuation of the previous segment
            t, prev = segs[-1]
            segs[-1] = (t, f"{prev} {line}")
    return [(t, s) for t, s in segs if s]


def parse_subtitles(text: str) -> list[tuple[int, str]]:
    segs, t = [], None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})[.,]\d{3}\s+-->", line) or \
            re.match(r"^()(\d{1,2}):(\d{2})[.,]\d{3}\s+-->", line)
        if m:
            t = _secs(*m.groups())
            continue
        if t is None or not line or line.isdigit() or line.upper().startswith("WEBVTT"):
            continue
        text_line = re.sub(r"<[^>]+>", "", line)
        if segs and segs[-1][0] == t:
            segs[-1] = (t, f"{segs[-1][1]} {text_line}")
        elif not segs or segs[-1][1] != text_line:   # auto-captions repeat lines
            segs.append((t, text_line))
    return segs


def transcribe(path: Path, model_size: str = "small", device: str = "auto") -> list[tuple[int, str]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit("Audio/video needs faster-whisper: pip install faster-whisper")
    model = WhisperModel(model_size, device=device, compute_type="auto")
    segments, _ = model.transcribe(str(path), vad_filter=True)
    return [(int(s.start), s.text.strip()) for s in segments if s.text.strip()]


def load(path: str | Path, whisper_model: str = "small") -> list[tuple[int, str]]:
    p = Path(path)
    if p.suffix.lower() in MEDIA:
        return transcribe(p, whisper_model)
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() in (".srt", ".vtt") or "-->" in text[:2000]:
        return parse_subtitles(text)
    segs = parse_youtube(text)
    if not segs:
        raise SystemExit(f"{p}: no timestamps found. Paste YouTube's 'Show transcript' text or use .srt/.vtt/audio.")
    return segs


def chunks(segs: list[tuple[int, str]], minutes: float = 8, overlap: float = 1) -> list[tuple[int, int, str]]:
    """Windows of `minutes` with `overlap` so a call spanning a boundary isn't cut in half.
    Each line is prefixed with its [mm:ss] so the model can report when things were said."""
    if not segs:
        return []
    out, start, end_t = [], 0, segs[-1][0]
    step, size = int((minutes - overlap) * 60), int(minutes * 60)
    while start <= end_t:
        part = [(t, s) for t, s in segs if start <= t < start + size]
        if part:
            body = "\n".join(f"[{t // 60}:{t % 60:02d}] {s}" for t, s in part)
            out.append((start, min(start + size, end_t), body))
        start += step
    return out
