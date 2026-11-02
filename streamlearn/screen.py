"""Read trades off the stream's screen - entries, stops and targets he shows on the chart but
doesn't say out loud.

  1. video: the stream's YouTube link (yt-dlp finds the video, nothing is downloaded in full) or a
     local video file; still frames come from ffmpeg (bundled by imageio-ffmpeg - nothing to install)
  2. moments: wherever he talks about entering, stops, targets or being in a trade, plus one frame
     every few minutes; each frame is scored for TradingView's long/short position tool (a green
     profit box and a red stop box stacked on each other)
  3. a vision model reads the best frames: ticker, timeframe, the position box (green over red =
     long, red over green = short), the entry line between them, stop and target prices on the
     right-hand price axis and what their colours mean
  4. the readings fill in missing entry / stop / targets on the setups he talked about at that time,
     or become new setups marked "seen on screen"
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

KEYWORDS = re.compile(r"\b(long|short|entered|entry|enter|in this trade|i'?m in|took (it|this|the)|position|"
                      r"stop|stopped|target|take profit|tp ?\d|filled|bought|sold|scaled|risk)\b", re.I)


# ---- video and frames -----------------------------------------------------------------------
def ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def video_source(source: str | None, url: str | None) -> str | None:
    """A local video file if there is one, otherwise a direct stream address from the YouTube link."""
    if source and Path(source).suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"} and Path(source).exists():
        return source
    if not url:
        return None
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                               "format": "best[height<=720][vcodec!=none]/bestvideo[height<=720]/best"}) as y:
            info = y.extract_info(url, download=False)
        return info.get("url") or next((f["url"] for f in reversed(info.get("formats", []))
                                        if f.get("vcodec") not in (None, "none")), None)
    except Exception:
        return None


def grab(src: str, t: int, out: Path) -> bool:
    exe = ffmpeg()
    if not exe:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([exe, "-loglevel", "error", "-y", "-ss", str(max(t, 0)), "-i", src, "-frames:v", "1",
                        "-vf", "scale='min(1280,iw)':-2", "-q:v", "3", str(out)], capture_output=True, timeout=120)
    return r.returncode == 0 and out.exists() and out.stat().st_size > 0


# ---- which moments to look at ------------------------------------------------------------------
def moments(segs: list[tuple[int, str]], every: int = 180, around: tuple[int, ...] = (0, 20)) -> list[int]:
    """Times worth a look: when he talks about a trade (and a little after), plus every few minutes."""
    ts = set()
    for t, text in segs:
        if KEYWORDS.search(text):
            ts.update(t + d for d in around)
    end = segs[-1][0] if segs else 0
    ts.update(range(60, end, every))
    out, last = [], -999
    for t in sorted(ts):                     # at most one frame per 15 s
        if t - last >= 15:
            out.append(t)
            last = t
    return out


def position_score(path: Path) -> float:
    """How much the frame looks like TradingView's position tool: large flat blocks of muted green
    (profit zone) and muted red (stop zone) sitting on top of each other. 0 = none."""
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB").resize((320, 180)), dtype=np.int16)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    green = (g > r + 25) & (g >= b - 10) & (g > 60)
    red = (r > g + 35) & (r > b + 25) & (r > 70)
    wide = lambda m: (m.sum(axis=1) > 0.12 * m.shape[1])          # rows with a wide band (boxes, not candles)
    gr, rr = wide(green), wide(red)
    if gr.sum() < 4 or rr.sum() < 4:
        return 0.0
    g_rows, r_rows = np.flatnonzero(gr), np.flatnonzero(rr)
    gap = min(abs(g_rows.max() - r_rows.min()), abs(r_rows.max() - g_rows.min()))
    return float((gr.mean() + rr.mean()) * (1.0 if gap <= 12 else 0.4))


# ---- reading a frame -------------------------------------------------------------------------
VISION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["trade_visible", "asset", "timeframe", "direction", "entry", "stop", "targets", "what_i_see",
                 "confidence"],
    "properties": {
        "trade_visible": {"type": "boolean",
                          "description": "a trade setup / position / order is drawn on the chart"},
        "asset": {"type": "string", "description": "ticker shown on the chart (e.g. BTC, ETH, SOL, SPY), or empty"},
        "timeframe": {"type": "string", "description": "chart timeframe shown (e.g. 4h, 15m), or empty"},
        "direction": {"type": "string", "enum": ["long", "short", "unclear"]},
        "entry": {"type": ["number", "null"]},
        "stop": {"type": ["number", "null"]},
        "targets": {"type": "array", "items": {"type": "number"}},
        "what_i_see": {"type": "string", "description": "one or two sentences: the drawing and the numbers"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    }}

VISION_PROMPT = """This is a frame from a crypto/stock trading livestream showing a TradingView chart.
Look for a trade the streamer has set up or is in, even if he never says the numbers:
- TradingView's Long/Short Position tool: a GREEN box (profit zone) and a RED box (stop zone) stacked on each
  other. Green ABOVE red = LONG; red ABOVE green = SHORT. The line where they meet is the ENTRY. The far edge of
  the red box is the STOP. The far edge of the green box is the TARGET. Their prices are usually printed in
  coloured labels on the right-hand price axis (green label = target, red label = stop) or inside the boxes
  ("Target: ...", "Stop: ...", risk/reward ratio).
- Horizontal order / position lines with labels (e.g. "LONG 0.5 @ 64,200", "SL", "TP1"), or an open position
  panel with entry price and P&L.
- The ticker and timeframe are in the top-left corner of the chart.
Read prices exactly as printed (they can be large, e.g. 64,215.5). Only report numbers you can actually read;
use null / [] otherwise. If no trade is drawn, set trade_visible to false.
Context - what he was saying at this moment: {context}
Return JSON matching the schema."""


def read_frame(llm, path: Path, context: str) -> dict:
    data = path.read_bytes()
    return llm.read_frame(data, VISION_PROMPT.format(context=context[:600]), VISION_SCHEMA)


def _near(segs, t, before=40, after=40) -> str:
    return " ".join(x for s, x in segs if t - before <= s <= t + after)


def read_stream(store, sid: int, segs, source: str | None, url: str | None, start, cfg: dict, llm, log=print) -> int:
    """Look at the stream's screen and merge what's on it into this stream's setups. Returns readings used."""
    src = video_source(source, url)
    if not src:
        log("Screen reading skipped: no video (add the YouTube link, or keep the video file)")
        return 0
    if not ffmpeg():
        log("Screen reading skipped: ffmpeg isn't available")
        return 0
    frames_dir = Path(cfg.get("frames", "streamlearn_frames")) / f"stream_{sid}"
    ts = moments(segs, every=int(cfg.get("screen_every", 180)))
    log(f"Looking at the screen: {len(ts)} moments to check")
    scored = []
    for i, t in enumerate(ts, 1):
        out = frames_dir / f"t{t:06d}.jpg"
        if out.exists() or grab(src, t, out):
            talk = bool(KEYWORDS.search(_near(segs, t, 5, 25)))
            scored.append((position_score(out) + (0.05 if talk else 0.0), t, out))
        if i % 20 == 0:
            log(f"  grabbed {i}/{len(ts)} frames")
    budget = int(cfg.get("screen_max_frames", 40))
    picked = sorted([x for x in scored if x[0] > 0], reverse=True)[:budget]
    log(f"Reading {len(picked)} frames that look like a trade on screen...")
    used = 0
    for score, t, out in sorted(picked, key=lambda x: x[1]):
        try:
            res = read_frame(llm, out, _near(segs, t))
        except Exception as e:
            log(f"  frame at {t // 60}:{t % 60:02d} failed: {e}")
            continue
        if not res.get("trade_visible") or res.get("confidence") == "low":
            continue
        if merge(store, sid, t, res, str(out), start, segs):
            used += 1
            log(f"  {t // 60}:{t % 60:02d} on screen: {res.get('what_i_see', '')[:120]}")
    store.commit()
    log(f"Screen: {used} trade readings added to setups")
    return used


def merge(store, sid: int, t: int, res: dict, frame: str, start, segs) -> bool:
    """Fill a nearby setup's missing numbers from the screen, or add the trade as a new setup."""
    import pandas as pd
    asset = (res.get("asset") or "").upper().replace("USDT", "").replace("USD", "").replace("/", "").strip()
    direction = res.get("direction") if res.get("direction") in ("long", "short") else None
    entry, stop = res.get("entry"), res.get("stop")
    targets = sorted({float(x) for x in res.get("targets") or [] if isinstance(x, (int, float)) and x > 0})
    if not (entry or stop or targets):
        return False
    rows = store.db.execute("SELECT id, t, asset, direction, entry, stop, targets FROM calls WHERE stream_id = ? "
                            "AND t BETWEEN ? AND ?", (sid, t - 300, t + 300)).fetchall()
    for cid, ct, a, d, e, s, tg in sorted(rows, key=lambda r: abs(r[1] - t)):
        if asset and a and asset != a.upper():
            continue
        if direction and d and direction != d:
            continue
        sets = {}
        if entry and not e:
            sets["entry"] = entry
        if stop and not s:
            sets["stop"] = stop
        if targets and not json.loads(tg or "[]"):
            sets["targets"] = json.dumps(targets)
        sets["frame"] = frame
        note = f" [screen: {res.get('what_i_see', '')}]"
        store.db.execute(f"UPDATE calls SET {', '.join(f'{k} = ?' for k in sets)}, summary = summary || ? "
                         "WHERE id = ?", [*sets.values(), note, cid])
        return True
    if not (asset and direction):
        return False
    store.add_call(sid, {"t": t, "said_at": str(start + pd.Timedelta(seconds=t)), "asset": asset,
                         "direction": direction, "kind": "screen", "timeframe": res.get("timeframe", ""),
                         "entry": entry, "stop": stop, "targets": targets, "conditions": [],
                         "quote": _near(segs, t, 20, 20)[:300],
                         "summary": f"Seen on his screen (not said): {res.get('what_i_see', '')}",
                         "t_start": max(t - 60, 0), "t_end": t + 60})
    cid = store.db.execute("SELECT max(id) FROM calls").fetchone()[0]
    store.set_frame(cid, frame)
    return True


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
