"""TradingView-style candle chart for the dashboard (TradingView's open-source Lightweight Charts,
bundled in livepaper/static so it works offline).

Scroll to zoom, drag to pan, pinch on a phone, crosshair with the candle's prices. Trades show as
markers like TradingView's strategy tester: an arrow at each entry, then TP1 / TP2 / ... at each
take profit, SL at a stop, Exit at a signal exit; an open position also gets horizontal lines for
its entry, stop and remaining targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mcbot.fmt import price as fmt_price

JS = Path(__file__).with_name("static") / "lightweight-charts.standalone.production.js"
GREEN, RED, BLUE, GREY, AMBER = "#22c55e", "#ef4444", "#3b82f6", "#94a3b8", "#f59e0b"


def _ts(t) -> int:
    return int(pd.Timestamp(t).tz_localize("UTC").timestamp() if pd.Timestamp(t).tzinfo is None
               else pd.Timestamp(t).timestamp())


def _snap(times: pd.DatetimeIndex, t) -> int | None:
    """The candle a fill happened in (markers must sit on a candle)."""
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    i = times.searchsorted(t, side="right") - 1
    return _ts(times[i]) if 0 <= i < len(times) else None


def label(reason: str) -> tuple[str, str]:
    """Short TradingView-style label and colour for an exit reason."""
    r = reason.lower()
    if r.startswith("target"):
        return "TP" + r[6:], GREEN
    if r.startswith("stop") or "breakeven" in r or "trail" in r:
        return ("BE" if "breakeven" in r else "Trail" if "trail" in r else "SL"), RED
    if "opposite" in r or r == "signal":
        return "Exit", GREY
    return "Close", GREY


def markers_from_fills(candles: pd.DataFrame, fills: list[dict]) -> list[dict]:
    """fills: [{"time", "side" (long/short), "reason" (entry/target1/stop/...), "price", "qty"?}]"""
    out = []
    for f in fills:
        t = _snap(candles.index, f["time"])
        if t is None:
            continue
        long = f["side"] == "long"
        px = fmt_price(f["price"])
        if f["reason"] == "entry":
            out.append({"time": t, "position": "belowBar" if long else "aboveBar", "color": BLUE if long else AMBER,
                        "shape": "arrowUp" if long else "arrowDown", "text": f"{'Long' if long else 'Short'} {px}"})
        else:
            txt, col = label(f["reason"])
            out.append({"time": t, "position": "aboveBar" if long else "belowBar", "color": col,
                        "shape": "arrowDown" if long else "arrowUp", "text": f"{txt} {px}"})
    return sorted(out, key=lambda m: m["time"])


def markers_from_trades(candles: pd.DataFrame, trades) -> list[dict]:
    """Backtest trades (mcbot.backtest.Trade with .fills = [(time, price, fraction, reason)])."""
    fills = []
    for tr in trades:
        fills.append({"time": tr.entry_time, "side": tr.side, "reason": "entry", "price": tr.entry_price})
        fills += [{"time": t, "side": tr.side, "reason": why, "price": px} for t, px, _, why in tr.fills]
    return markers_from_fills(candles, fills)


def html(candles: pd.DataFrame, markers: list[dict] | None = None, lines: list[dict] | None = None,
         height: int = 460, show_bars: int = 150, segments: list[dict] | None = None) -> str:
    """lines: [{"price", "title", "color"}] drawn across the chart (e.g. the open position).
    segments: [{"points": [(time, price), (time, price)], "color", "width", "dashed"}] - trendlines
    such as a chart pattern's top and bottom lines."""
    segs = []
    for sg in segments or []:
        pts = sorted({_ts(t): float(p) for t, p in sg["points"]}.items())
        if len(pts) >= 2:
            segs.append({"data": [{"time": t, "value": p} for t, p in pts], "color": sg.get("color", "#facc15"),
                         "width": sg.get("width", 2), "dashed": bool(sg.get("dashed"))})
    data = [{"time": _ts(t), "open": float(r.open), "high": float(r.high), "low": float(r.low), "close": float(r.close)}
            for t, r in candles[["open", "high", "low", "close"]].iterrows()]
    last = float(candles.close.iloc[-1]) if len(candles) else 1.0
    decimals = 2 if last >= 1 else max(4, min(10, 3 - int(f"{last:e}".split("e")[1])))
    step = (candles.index[-1] - candles.index[-2]).total_seconds() if len(candles) > 1 else 3600
    cfg = {"data": data, "markers": markers or [], "lines": lines or [], "decimals": decimals,
           "show": show_bars, "step": step, "segments": segs}
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin:0; background:#0b1220; font-family:-apple-system,Segoe UI,Roboto,sans-serif; }}
#bar {{ display:flex; gap:4px; padding:6px 8px; align-items:center; }}
#bar button {{ background:#1e293b; color:#cbd5e1; border:0; border-radius:6px; padding:4px 10px; cursor:pointer; font-size:12px; }}
#bar button:hover {{ background:#334155; }}
#info {{ color:#94a3b8; font-size:12px; margin-left:auto; white-space:nowrap; overflow:hidden; }}
#chart {{ height:{height - 40}px; }}
</style></head><body>
<div id="bar"><button data-r="1">1D</button><button data-r="7">1W</button><button data-r="30">1M</button>
<button data-r="90">3M</button><button data-r="365">1Y</button><button data-r="0">All</button>
<span id="info">Scroll to zoom · drag to pan</span></div><div id="chart"></div>
<script>{JS.read_text()}</script>
<script>
const C = {json.dumps(cfg).replace("</", "<\\/")};
const fmt = v => Number(v).toLocaleString('en-US', {{minimumFractionDigits: C.decimals, maximumFractionDigits: C.decimals}});
const el = document.getElementById('chart');
const chart = LightweightCharts.createChart(el, {{
  autoSize: true,
  layout: {{ background: {{ color: '#0b1220' }}, textColor: '#cbd5e1' }},
  grid: {{ vertLines: {{ color: '#1e293b' }}, horzLines: {{ color: '#1e293b' }} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{ borderColor: '#334155' }},
  timeScale: {{ borderColor: '#334155', timeVisible: true, secondsVisible: false, rightOffset: 8 }},
  localization: {{ priceFormatter: fmt }},
}});
const linePrices = C.lines.map(l => l.price);
const s = chart.addCandlestickSeries({{ upColor: '#22c55e', downColor: '#ef4444', wickUpColor: '#22c55e',
  wickDownColor: '#ef4444', borderVisible: false,
  priceFormat: {{ type: 'custom', formatter: fmt, minMove: Math.pow(10, -C.decimals) }},
  // keep the open position's entry / stop / targets on screen, like TradingView's position tool
  autoscaleInfoProvider: orig => {{
    const r = orig();
    if (!r || !linePrices.length) return r;
    return {{ ...r, priceRange: {{ minValue: Math.min(r.priceRange.minValue, ...linePrices),
                                   maxValue: Math.max(r.priceRange.maxValue, ...linePrices) }} }};
  }} }});
s.setData(C.data);
s.setMarkers(C.markers);
for (const l of C.lines) s.createPriceLine({{ price: l.price, color: l.color, lineWidth: 2,
  lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: l.title }});
for (const g of C.segments) {{
  const ls = chart.addLineSeries({{ color: g.color, lineWidth: g.width, lastValueVisible: false,
    priceLineVisible: false, crosshairMarkerVisible: false,
    lineStyle: g.dashed ? LightweightCharts.LineStyle.Dashed : LightweightCharts.LineStyle.Solid }});
  ls.setData(g.data);
}}
const n = C.data.length;
function show(days) {{
  if (!n) return;
  if (!days) return chart.timeScale().fitContent();
  const bars = Math.max(10, Math.round(days * 86400 / C.step));
  chart.timeScale().setVisibleLogicalRange({{ from: n - bars, to: n + 5 }});
}}
// open on the latest trades: the last C.show candles, reaching back to the most recent entry if older
let from = Math.max(0, n - C.show);
if (C.markers.length) {{
  const lastT = C.markers[C.markers.length - 1].time;
  const firstOfLastTrade = C.markers.filter(m => m.shape && m.text && /^(Long|Short)/.test(m.text)).map(m => m.time)
                                    .filter(t => t <= lastT).pop() || lastT;
  const idx = C.data.findIndex(d => d.time >= firstOfLastTrade);
  if (idx >= 0 && idx - 20 < from) from = Math.max(0, idx - 20);
}}
chart.timeScale().setVisibleLogicalRange({{ from: from, to: n + 5 }});
document.querySelectorAll('#bar button').forEach(b => b.onclick = () => show(+b.dataset.r));
const info = document.getElementById('info');
chart.subscribeCrosshairMove(p => {{
  const d = p && p.seriesData && p.seriesData.get(s);
  info.textContent = d ? `O ${{fmt(d.open)}}  H ${{fmt(d.high)}}  L ${{fmt(d.low)}}  C ${{fmt(d.close)}}`
                       : 'Scroll to zoom · drag to pan';
}});
</script></body></html>"""


def show(candles: pd.DataFrame, markers=None, lines=None, height: int = 460, show_bars: int = 150,
         segments=None) -> None:
    import streamlit as st
    if candles is None or candles.empty:
        st.caption("No candles to chart yet.")
        return
    page = html(candles, markers, lines, height, show_bars, segments)
    if hasattr(st, "iframe"):                 # Streamlit 1.5x+ (components.html is being removed)
        st.iframe(page, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(page, height=height)
