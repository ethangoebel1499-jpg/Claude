"""Screen reading: find the position drawing in the stream video and merge what's on it into setups."""
import json
import subprocess

import numpy as np
import pytest
from PIL import Image, ImageDraw

from streamlearn import screen
from streamlearn.pipeline import add_stream, parse_start
from streamlearn.store import Store

pytest.importorskip("imageio_ffmpeg")


def chart(position: bool, w=1280, h=720) -> Image.Image:
    """A dark TradingView-like chart; with position=True a long position drawing (green over red)."""
    img = Image.new("RGB", (w, h), (19, 23, 34))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(1)
    y = 400
    for x in range(40, w - 160, 12):                       # candles
        y2 = y + int(rng.normal(0, 8))
        col = (8, 153, 129) if y2 < y else (242, 54, 69)
        d.rectangle([x, min(y, y2), x + 6, max(y, y2) + 1], fill=col)
        d.line([x + 3, min(y, y2) - 6, x + 3, max(y, y2) + 6], fill=col)
        y = y2
    if position:
        d.rectangle([700, 220, 1080, 360], fill=(22, 82, 72))      # profit zone (muted green)
        d.rectangle([700, 360, 1080, 430], fill=(92, 36, 44))      # stop zone (muted red)
        d.text((1120, 214), "67,000.00", fill=(255, 255, 255))
        d.text((1120, 354), "64,200.00", fill=(255, 255, 255))
        d.text((1120, 424), "62,000.00", fill=(255, 255, 255))
    return img


def test_position_drawing_is_spotted(tmp_path):
    with_box, without = tmp_path / "a.jpg", tmp_path / "b.jpg"
    chart(True).save(with_box)
    chart(False).save(without)
    assert screen.position_score(with_box) > 0.2 and screen.position_score(without) == 0.0


def test_moments_cover_trade_talk_and_regular_checks():
    segs = [(10, "hello everyone"), (300, "i'm long from here"), (900, "stop goes under the low"), (1200, "bye")]
    ts = screen.moments(segs, every=180)
    assert 300 in ts and 320 in ts and 900 in ts and 60 in ts and 240 in ts      # trade talk + every 3 minutes


class FakeVision:
    """Stands in for Claude / qwen2.5vl reading the frame."""
    def __init__(self):
        self.seen = []

    def read_frame(self, jpeg, prompt, schema):
        self.seen.append(prompt)
        return {"trade_visible": True, "asset": "BTC", "timeframe": "4h", "direction": "long", "entry": 64200,
                "stop": 62000, "targets": [67000], "confidence": "high",
                "what_i_see": "Long position tool: entry 64,200, stop 62,000, target 67,000"}

    def extract(self, chunk, title, date, system=None):
        # he says he's long but gives no numbers - only the screen has them
        return {"calls": [{"time": "5:00", "setup_start": "4:50", "setup_end": "5:30", "summary": "Long BTC here.",
                           "asset": "BTC", "direction": "long", "kind": "entered", "timeframe": "4h", "entry": None,
                           "stop": None, "targets": [], "conditions": [], "quote": "i'm long bitcoin here"}],
                "rules": [], "levels": [], "patterns": []}


def test_numbers_only_on_screen_end_up_in_the_setup(tmp_path):
    import imageio_ffmpeg
    frames = tmp_path / "f"
    frames.mkdir()
    for i in range(12):                                     # one image per 30 s; the drawing from 4:30 on
        chart(position=i >= 9).save(frames / f"{i:03d}.png")
    video = tmp_path / "stream.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-framerate", "1/30", "-i",
                    str(frames / "%03d.png"), "-r", "1", "-pix_fmt", "yuv420p", str(video)], check=True)
    transcript = tmp_path / "t.txt"
    transcript.write_text("0:00\nwelcome\n5:00\ni'm long bitcoin here\n5:50\nok that's it\n")
    store = Store(tmp_path / "s.db")
    llm = FakeVision()
    cfg = {"frames": str(tmp_path / "frames"), "_vision_llm": llm, "screen_every": 120}
    sid = add_stream(store, transcript, parse_start("2026-08-20 16:00", "UTC"), "Drawn trade", cfg, llm=llm,
                     log=lambda *a: None)
    store.db.execute("UPDATE streams SET source = ? WHERE id = ?", (str(video), sid))
    # the transcript had no video next to it, so read the screen from the video directly
    n = screen.read_stream(store, sid, [(0, "welcome"), (300, "i'm long bitcoin here"), (350, "ok")], str(video), None,
                           parse_start("2026-08-20 16:00", "UTC"), cfg, llm, log=lambda *a: None)
    c = store.table("calls").iloc[0]
    assert n >= 1 and len(store.table("calls")) == 1               # merged into the spoken call, not duplicated
    assert c.entry == 64200 and c.stop == 62000 and json.loads(c.targets) == [67000.0]
    assert "screen:" in c.summary and c.frame
    assert all("TradingView" in p for p in llm.seen)
