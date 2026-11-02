"""YouTube links -> lessons: a single video or a whole playlist, with each video's captions fetched
automatically (no copy-pasting transcripts).

  videos("https://www.youtube.com/playlist?list=...")  -> [{"id", "title", "url"}, ...]
  captions("https://www.youtube.com/watch?v=...")      -> (info, [(seconds, text), ...])

Uses yt-dlp (installed with the app). Captions come from the video's own subtitles or YouTube's
automatic captions, English first. A video with no captions at all can't be learned from this way:
paste its transcript by hand instead (or keep the audio and let faster-whisper transcribe it).
"""

from __future__ import annotations

import json
import re
from urllib.parse import parse_qs, urlparse

LANGS = ("en", "en-US", "en-GB", "en-orig")


def kind(url: str) -> str | None:
    """'playlist', 'video' or None. A watch link that is part of a playlist (…&list=…) counts as the
    playlist, since that's usually why someone pastes it."""
    u = urlparse(url.strip())
    host = u.netloc.lower()
    if "youtube.com" not in host and "youtu.be" not in host:
        return None
    q = parse_qs(u.query)
    if q.get("list") and not q["list"][0].startswith(("RD", "UL")):      # RD = auto-mix, not a real playlist
        return "playlist"
    if "youtu.be" in host or q.get("v") or u.path.startswith(("/live/", "/shorts/")):
        return "video"
    return None


def video_id(url: str) -> str | None:
    u = urlparse(url.strip())
    if "youtu.be" in u.netloc:
        return u.path.strip("/").split("/")[0] or None
    q = parse_qs(u.query)
    if q.get("v"):
        return q["v"][0]
    m = re.match(r"^/(?:live|shorts|embed)/([\w-]{6,})", u.path)
    return m.group(1) if m else None


def _ydl(extra: dict | None = None):
    import yt_dlp
    return yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, **(extra or {})})


def videos(url: str) -> list[dict]:
    """Every video in a playlist (or the one video), in playlist order."""
    if kind(url) == "video":
        vid = video_id(url)
        return [{"id": vid, "title": "", "url": f"https://www.youtube.com/watch?v={vid}"}]
    lid = parse_qs(urlparse(url).query).get("list", [""])[0]
    with _ydl({"extract_flat": "in_playlist"}) as y:
        info = y.extract_info(f"https://www.youtube.com/playlist?list={lid}", download=False)
    out = []
    for e in info.get("entries") or []:
        if not e or not e.get("id"):
            continue
        if e.get("title") in ("[Private video]", "[Deleted video]"):
            continue
        out.append({"id": e["id"], "title": e.get("title") or "", "url": f"https://www.youtube.com/watch?v={e['id']}",
                    "playlist": info.get("title") or ""})
    return out


def _pick(tracks: dict) -> tuple[str, str] | None:
    """(format, url) of the best English caption track."""
    for lang in LANGS + tuple(k for k in tracks if k.startswith("en")):
        for f in tracks.get(lang) or []:
            if f.get("ext") == "json3":
                return "json3", f["url"]
        for f in tracks.get(lang) or []:
            if f.get("ext") == "vtt":
                return "vtt", f["url"]
    return None


def parse_json3(text: str) -> list[tuple[int, str]]:
    """YouTube's json3 caption format -> [(seconds, text)], merged into ~sentence-sized lines."""
    data = json.loads(text)
    segs: list[tuple[int, str]] = []
    for ev in data.get("events", []):
        words = "".join(s.get("utf8", "") for s in ev.get("segs") or []).replace("\n", " ").strip()
        if not words or words == "[Music]":
            continue
        t = int(ev.get("tStartMs", 0) // 1000)
        if segs and t - segs[-1][0] < 4 and len(segs[-1][1]) < 160:
            segs[-1] = (segs[-1][0], f"{segs[-1][1]} {words}")
        else:
            segs.append((t, words))
    return segs


def captions(url: str) -> tuple[dict, list[tuple[int, str]]]:
    """(video info, transcript segments). Raises RuntimeError when the video has no captions."""
    with _ydl({"writesubtitles": True, "writeautomaticsub": True}) as y:
        info = y.extract_info(url, download=False)
        pick = _pick(info.get("subtitles") or {}) or _pick(info.get("automatic_captions") or {})
        if not pick:
            raise RuntimeError("no English captions on this video - paste its transcript by hand instead")
        fmt, cap_url = pick
        text = y.urlopen(cap_url).read().decode("utf-8", errors="replace")
    if fmt == "json3":
        segs = parse_json3(text)
    else:
        from streamlearn.transcripts import parse_subtitles
        segs = parse_subtitles(text)
    meta = {"id": info.get("id"), "title": info.get("title") or "", "channel": info.get("channel") or "",
            "url": info.get("webpage_url") or url, "duration": info.get("duration"),
            "date": info.get("release_date") or info.get("upload_date") or ""}
    if not segs:
        raise RuntimeError("the captions were empty")
    return meta, segs
