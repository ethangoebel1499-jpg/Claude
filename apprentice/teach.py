"""Teaching: read a trading lesson (stream / video transcript, or plain notes) and turn it into
knowledge the apprentice can use on a chart.

Two kinds of knowledge come out of every lesson:
  concepts  what the teacher says in words (rules of thumb, what to look for, mistakes to avoid)
  setups    concrete trades, translated into rules the apprentice can check on every candle
            (the mcbot.builder rule list: EMAs, RSI, MACD, Market Cipher, candles, support /
            resistance, chart pattern breaks, ...). Anything it can't check yet is kept as text in
            `unchecked`, so you can see what it doesn't understand.

New setups start as candidates. `brain.learn()` then tests them on real prices and only the ones
that actually worked become active.
"""

from __future__ import annotations

import re

from mcbot import builder
from streamlearn.transcripts import chunks, parse_subtitles, parse_youtube

CATEGORIES = ["entry", "exit", "stop", "risk", "levels", "patterns", "indicators", "candles", "trend", "timeframe",
              "psychology", "mistakes", "other"]


def schema() -> dict:
    rule = {"type": "object", "additionalProperties": False, "required": ["rule", "a", "b"],
            "properties": {"rule": {"type": "string", "enum": list(builder.RULES)},
                           "a": {"type": ["number", "null"]}, "b": {"type": ["number", "null"]}}}
    return {
        "type": "object", "additionalProperties": False, "required": ["concepts", "setups"],
        "properties": {
            "concepts": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["time", "category", "text", "quote"],
                "properties": {"time": {"type": "string"}, "category": {"type": "string", "enum": CATEGORIES},
                               "text": {"type": "string", "description": "the lesson as one clear sentence"},
                               "quote": {"type": "string", "description": "the teacher's words, verbatim, short"}}}},
            "setups": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["time", "name", "direction", "rules", "unchecked", "stop_atr", "rr", "timeframe", "why",
                             "quote"],
                "properties": {
                    "time": {"type": "string"},
                    "name": {"type": "string", "description": "short name, e.g. 'Falling wedge breakout'"},
                    "direction": {"type": "string", "enum": ["long", "short"]},
                    "rules": {"type": "array", "items": rule,
                              "description": "conditions from the rule list that must ALL be true on the entry "
                                             "candle; a and b are the rule's numbers (null = default)"},
                    "unchecked": {"type": "array", "items": {"type": "string"},
                                  "description": "conditions he gives that none of the rules can express"},
                    "stop_atr": {"type": ["number", "null"],
                                 "description": "stop distance in ATRs if you can infer it (swing stop ~1.5), else null"},
                    "rr": {"type": ["number", "null"], "description": "target as a multiple of the risk, if given"},
                    "timeframe": {"type": "string"},
                    "why": {"type": "string", "description": "why the setup works, in the teacher's reasoning"},
                    "quote": {"type": "string"}}}},
        }}


def system() -> str:
    menu = "\n".join(f"  {k}: {builder.describe({'rule': k})}" for k in builder.RULES)
    return f"""You are an apprentice trader studying a trading lesson (a livestream or video transcript,
auto-generated, so numbers may be written as words). Each line starts with its [m:ss] time.

Learn two things:
1. concepts - every reusable piece of trading knowledge the teacher states: what to look for, when to
   enter / exit, where stops go, how to read a pattern, candle or indicator, risk rules, and mistakes
   to avoid. One clear sentence each. Skip chit-chat, ads and viewers' questions.
2. setups - every repeatable trade setup he teaches (not one-off price predictions). A video that
   explains a strategy ALWAYS has at least one setup: its long version and, if he trades it, its short
   version as a separate setup. Translate its conditions into the rules below, which you can check on
   any chart. Use only rules that match what he says; put any condition no rule can express into
   `unchecked` in plain words - still record the setup even if no rule fits. Don't invent conditions he
   didn't give. Typically 2-4 rules per setup. Examples: "price touches the lower Bollinger band" =
   bb_touch_lower; "AO makes a higher low while price makes a lower low" = ao_bull_div; "AO turns
   green" = ao_turns_green; "RSI divergence" = rsi_bull_div / rsi_bear_div.

Rules you can check (id: meaning; A and B are its numbers, null = the default shown):
{menu}

Quotes must be verbatim. Return JSON matching the schema."""


def rating_notes(store, n: int = 8) -> str:
    """Your ratings of what it learned before, with your comments - added to its instructions so the next
    lesson is read better (low ratings: what not to do; high ratings: what to keep doing)."""
    rows = [r for r in store.ratings() if r["kind"] in ("setup", "concept") and (r["comment"] or r["stars"] <= 2)]
    if not rows:
        return ""
    bad = [r for r in rows if r["stars"] <= 2][:n]
    good = [r for r in rows if r["stars"] >= 4][:max(2, n // 2)]
    lines = ["Your teacher's accuracy ratings of what you learned from earlier lessons (1-5 stars). Learn from them:"]
    for r in bad + good:
        snap = r["snapshot"]
        what = snap.get("name") or snap.get("text") or ""
        rules = "; ".join(snap.get("rules_text", [])) if snap.get("rules_text") else ""
        lines.append(f"- {r['stars']}/5 for the {r['kind']} \"{what}\"" + (f" (rules: {rules})" if rules else "")
                     + (f". Teacher's comment: {r['comment']}" if r["comment"] else ""))
    return "\n".join(lines)


def segments(text: str) -> list[tuple[int, str]]:
    """Any pasted lesson -> [(seconds, text)]. Text without timestamps (notes, an article) works too."""
    segs = parse_subtitles(text) if "-->" in text[:3000] else parse_youtube(text)
    if segs:
        return segs
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    return [(0, p) for p in paras]


def _mmss(t: str) -> int:
    from streamlearn.extract import mmss
    return mmss(t or "")


def clean_rules(rules: list[dict]) -> list[dict]:
    out, seen = [], set()
    for r in rules or []:
        rid = r.get("rule")
        if rid not in builder.RULES:
            continue
        a, b = r.get("a"), r.get("b")
        key = (rid, a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append({"rule": rid, **({"a": a} if a is not None else {}), **({"b": b} if b is not None else {})})
    return out


def _signature(direction: str, rules: list[dict]) -> tuple:
    return direction, tuple(sorted((r["rule"], r.get("a"), r.get("b")) for r in rules))


def learn_lesson(store, lesson_id: int, segs: list[tuple[int, str]], title: str, llm=None, log=print,
                 chunk_minutes: float = 10) -> dict:
    """Read a lesson in parts and save what was learned. Returns counts."""
    known = {_signature(s["direction"], s["rules"]) for s in store.setups()}
    parts = chunks(segs, chunk_minutes, 1) if segs and segs[-1][0] > 0 else \
        [(0, 0, "\n".join(s for _, s in segs[i:i + 120])) for i in range(0, len(segs), 120)]
    n_c = n_s = failed = 0
    seen_c: set[str] = set()
    notes = rating_notes(store)
    sysmsg = system() + (f"\n\n{notes}" if notes else "")
    for i, (_, _, body) in enumerate(parts, 1):
        try:
            res = llm.ask(sysmsg, f"Lesson: {title}\n\nTranscript:\n{body}", schema()) if llm else guess(body)
        except Exception as e:
            failed += 1
            log(f"  part {i}/{len(parts)} failed: {e}")
            continue
        for c in res.get("concepts", []):
            norm = re.sub(r"\W+", " ", c.get("text", "").lower()).strip()
            if not norm or norm in seen_c:
                continue
            seen_c.add(norm)
            store.add_concept(lesson_id, {"t": _mmss(c.get("time")), "category": c.get("category", "other"),
                                          "text": c["text"], "quote": c.get("quote", "")})
            n_c += 1
        for s in res.get("setups", []):
            rules = clean_rules(s.get("rules"))
            unchecked = [u for u in s.get("unchecked", []) if str(u).strip()]
            if s.get("direction") not in ("long", "short") or not (rules or unchecked):
                continue
            # a setup it understood but couldn't put into rules is kept (status "needs rules") so you can
            # see it and pick the rules yourself, instead of disappearing
            sig = _signature(s["direction"], rules) if rules else (s["direction"], (s.get("name") or "").lower())
            if sig in known:
                continue
            known.add(sig)
            store.add_setup({"name": (s.get("name") or "Setup")[:80], "direction": s["direction"], "rules": rules,
                             "status": "candidate" if rules else "needs rules", "unchecked": unchecked,
                             "stop_atr": s.get("stop_atr") or 1.5, "rr": s.get("rr") or 2.0,
                             "timeframe": s.get("timeframe", ""), "why": s.get("why", ""), "quote": s.get("quote", ""),
                             "source": "lesson", "lesson_id": lesson_id})
            n_s += 1
        store.commit()
        log(f"  part {i}/{len(parts)}: {n_c} concepts, {n_s} setups so far")
    if parts and failed == len(parts):
        raise RuntimeError("every part of the lesson failed - nothing was learned (see the errors above)")
    return {"concepts": n_c, "setups": n_s, "failed": failed, "parts": len(parts)}


# ---- without an AI model: keyword reading (rough, but works offline) ----------------------------
KEYWORDS = [
    (r"falling wedge", "in_bullish_pattern"), (r"rising wedge", "in_bearish_pattern"),
    (r"ascending triangle|bull flag|cup and handle", "in_bullish_pattern"),
    (r"descending triangle|bear flag", "in_bearish_pattern"),
    (r"break(s|ing)? out|breakout|break above", "pattern_break_up"),
    (r"break(s|ing)? down|breakdown|break below", "pattern_break_down"),
    (r"apex|squeez", "near_apex"),
    (r"engulf", None),
    (r"hammer|pin bar|long (lower )?wick", "hammer"), (r"shooting star", "shooting_star"),
    (r"bounce off support|reclaim|hold(s|ing)? support", "sr_bounce_up"),
    (r"reject(ed|ion)? (at|from) resistance|lose support|lost support", "sr_reject_down"),
    (r"oversold|rsi (is )?(below|under) 30", "rsi_below"), (r"overbought|rsi (is )?(above|over) 70", "rsi_above"),
    (r"green dot", "wt_green_dot"), (r"red dot", "wt_red_dot"),
    (r"money flow (is )?(green|positive|blue)", "mf_positive"), (r"money flow (is )?(red|negative)", "mf_negative"),
    (r"golden cross", "ema_cross_up"), (r"death cross", "ema_cross_down"),
    (r"above the 200", "close_above_sma"), (r"below the 200", "close_below_sma"),
    (r"macd cross(es)? (up|above)|bullish macd", "macd_cross_up"),
    (r"macd cross(es)? (down|below)|bearish macd", "macd_cross_down"),
    (r"volume spike|big volume|high volume", "volume_spike"),
    (r"higher low", "higher_low"), (r"lower high", "lower_high"),
    (r"(touch|tag|hit|pierc)\w* (the )?(lower|bottom) (bollinger )?band", "bb_touch_lower"),
    (r"(touch|tag|hit|pierc)\w* (the )?(upper|top) (bollinger )?band", "bb_touch_upper"),
    (r"(awesome oscillator|\bao\b).{0,40}divergen|divergen.{0,40}(awesome oscillator|\bao\b)", "AO_DIV"),
    (r"\brsi\b.{0,40}divergen|divergen.{0,40}\brsi\b", "RSI_DIV"),
    (r"\bmacd\b.{0,40}divergen|divergen.{0,40}\bmacd\b", "MACD_DIV"),
    (r"(awesome oscillator|\bao\b).{0,30}(turns?|flips?|changes?).{0,10}green", "ao_turns_green"),
    (r"(awesome oscillator|\bao\b).{0,30}(turns?|flips?|changes?).{0,10}red", "ao_turns_red"),
    (r"(awesome oscillator|\bao\b).{0,20}(above|over) (the )?zero", "ao_above_zero"),
    (r"(awesome oscillator|\bao\b).{0,20}(below|under) (the )?zero", "ao_below_zero"),
]
LESSON = re.compile(r"\b(always|never|look for|wait for|you want|rule|should|don't|make sure|the key)\b", re.I)


def guess(body: str) -> dict:
    """Keyword-based reading of one lesson part. Used when no AI model is set up."""
    concepts, setups = [], []
    for line in body.splitlines():
        m = re.match(r"^\[(\d+:\d+(?::\d+)?)\]\s*(.*)$", line)
        t, text = (m.group(1), m.group(2)) if m else ("0:00", line)
        low = text.lower()
        if LESSON.search(text) and len(text) > 25:
            concepts.append({"time": t, "category": "other", "text": text.strip()[:240], "quote": text.strip()[:240]})
        rules = []
        bull = bool(re.search(r"\b(long|buy|bullish|calls?|go up)\b", low))
        bear = bool(re.search(r"\b(short|sell|bearish|puts?|go down)\b", low))
        for pat, rid in KEYWORDS:
            if re.search(pat, low):
                rid = rid or ("bear_engulf" if bear and not bull else "bull_engulf")
                if rid.endswith("_DIV"):          # which divergence depends on the trade's direction
                    rid = f"{rid[:-4].lower()}_{'bear' if bear and not bull else 'bull'}_div"
                rules.append({"rule": rid, "a": None, "b": None})
        if len(rules) >= 2 and bull != bear:
            setups.append({"time": t, "name": text.strip()[:60], "direction": "long" if bull else "short",
                           "rules": rules, "unchecked": [], "stop_atr": None, "rr": None, "timeframe": "",
                           "why": "", "quote": text.strip()[:240]})
    return {"concepts": concepts, "setups": setups}
