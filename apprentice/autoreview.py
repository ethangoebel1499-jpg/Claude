"""The apprentice reviews what it learns, by itself (the same rules Claude uses for its reviews):

  concepts  delete ads / channel plugs, leverage talk (you trade without leverage), raw transcript fragments,
            dated market commentary, off-topic text and duplicates; keep the rest
  setups    turn off ones whose rules can't all be true, backwards divergences, checklist dumps, prices read as
            indicator settings, setups that never fire, duplicates, and ones no better than random
  lessons   delete a lesson that isn't trading material (title, or most of what it said was junk) - with the
            concepts and setups learned only from it

Every item gets a verdict, a 1-5 star rating and a comment, shown in 🧠 Apprentice → What it knows.
It never touches what you pinned, setups you made, anything you rated 4-5★, or items Claude's own review covers
(Claude's verdict wins). It runs after every lesson it reads and every 🎓 Learn (⚙️ switch in the Apprentice tab).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

STARS = {"use": 5, "keep": 4, "watch": 3, "drop": 1}

# ---- concepts ---------------------------------------------------------------------------------------
AD = re.compile(r"patreon|discord|subscribe|like the video|\blikes\b|giveaway|link (in|below)|in the description|"
                r"sign up|referral|promo code|\bvip\b|casper crew|antarctic|agora|mexc|bybit|blofin|toobit|merch|"
                r"sponsor|my channel|the channel", re.I)
LEVERAGE = re.compile(r"leverage|\b\d{2,3}x\b|liquidat|\bmargin\b", re.I)
NEWS = re.compile(r"\b(currently|right now|this week|today|yesterday|recently|at the moment|we are (now|in|at)|"
                  r"is having a|everyone is|the market is (trading|in)|bitcoin is (extremely|having|currently|in|at|"
                  r"going)|we broke|we('re| are) (looking|seeing)|will (hit|reach|go)|expected to)\b|\$\d|\b\d{2,3}k\b|"
                  r"\b\d{2},\d{3}\b", re.I)
FRAGMENT = re.compile(r"\b(uh|um|you know|guys|y'all|gonna|kind of like|i mean)\b", re.I)
OFF_TOPIC = re.compile(r"girlfriend|date with|friend zone|\bgod\b|\blord\b|yeshua|jesus|disciple|matzah|worship|"
                       r"sermon|scripture|prayer|church|\bbread\b|my body|no specific trading", re.I)


def judge_concept(text: str, seen: dict, cid: int) -> tuple[str, str]:
    text = str(text or "").strip()
    norm = re.sub(r"[^a-z ]", "", text.lower())[:70]
    if not text:
        return "drop", "Empty."
    if OFF_TOPIC.search(text):
        return "drop", "Not about trading."
    if AD.search(text):
        return "drop", "Promo / channel / exchange plug, not trading knowledge."
    if LEVERAGE.search(text):
        return "drop", "About leverage - you trade without leverage."
    if text[:1].islower() or FRAGMENT.search(text) or len(text) > 260:
        return "drop", "A raw transcript fragment, not a usable idea."
    if NEWS.search(text) or re.match(r"(bitcoin|btc|eth|ethereum|the market|price|crypto) (is|has|was)\b", text, re.I):
        return "drop", "Dated market commentary (what price was doing that day), not a rule."
    if norm in seen:
        return "drop", f"Duplicate of concept {seen[norm]}."
    seen[norm] = cid
    return "keep", "Sound and usable as written."


# ---- setups -----------------------------------------------------------------------------------------
BULL = {"ao_bull_div", "ao_hidden_bull_div", "ao_turns_green", "ao_cross_up_zero", "bull_engulf", "hammer",
        "rsi_bull_div", "rsi_hidden_bull_div", "macd_bull_div", "macd_hidden_bull_div", "wt_bull_div",
        "wt_hidden_bull_div", "wt_green_dot", "macd_cross_up", "rsi_cross_up", "pattern_break_up", "sr_bounce_up",
        "ema_cross_up"}
BEAR = {"ao_bear_div", "ao_hidden_bear_div", "ao_turns_red", "ao_cross_down_zero", "bear_engulf", "shooting_star",
        "rsi_bear_div", "rsi_hidden_bear_div", "macd_bear_div", "macd_hidden_bear_div", "wt_bear_div",
        "wt_hidden_bear_div", "wt_red_dot", "macd_cross_down", "rsi_cross_down", "pattern_break_down",
        "sr_reject_down"}
PAIRS = [("wt_green_dot", "wt_red_dot"), ("wt_below", "wt_above"), ("mf_positive", "mf_negative"),
         ("macd_positive", "macd_negative"), ("macd_cross_up", "macd_cross_down"), ("ao_green", "ao_red"),
         ("ao_above_zero", "ao_below_zero"), ("ha_green", "ha_red"), ("bb_above_mid", "bb_below_mid"),
         ("bb_below_lower", "bb_above_upper"), ("green_candles", "red_candles"), ("breakout_high", "breakdown_low"),
         ("rsi_cross_up", "rsi_cross_down"), ("ao_turns_green", "ao_turns_red"),
         ("pattern_break_up", "pattern_break_down"), ("in_bullish_pattern", "in_bearish_pattern"),
         ("up_pct", "down_pct"), ("ao_cross_up_zero", "ao_cross_down_zero"), ("bull_engulf", "bear_engulf"),
         ("hammer", "shooting_star"), ("higher_low", "lower_high")]
TRIGGERS = BULL | BEAR | {"bb_touch_lower", "bb_touch_upper", "bb_below_lower", "bb_above_upper", "breakout_high",
                          "breakdown_low", "volume_spike", "wt_below", "wt_above", "rsi_below", "rsi_above", "squeeze",
                          "near_apex", "higher_low", "lower_high", "in_bullish_pattern", "in_bearish_pattern",
                          "up_pct", "down_pct", "green_candles", "red_candles", "ha_green", "ha_red"}


def rule_problems(s: dict) -> list[str]:
    rules = s.get("rules") or []
    ids = [r["rule"] for r in rules]
    out = []
    for a, b in PAIRS:
        if a in ids and b in ids:
            out.append(f"impossible: {a} + {b}")
    for pa, pb in (("close_above_sma", "close_below_sma"), ("close_above_ema", "close_below_ema")):
        if {r.get("a") for r in rules if r["rule"] == pa} & {r.get("a") for r in rules if r["rule"] == pb}:
            out.append(f"impossible: {pa} + {pb} of the same length")
    lo = [r.get("a") or 30 for r in rules if r["rule"] == "rsi_below"]
    hi = [r.get("a") or 70 for r in rules if r["rule"] == "rsi_above"]
    if lo and hi and min(lo) <= max(hi):
        out.append("impossible: RSI below one level and above a higher one")
    if "wt_green_dot" in ids and "wt_above" in ids:
        out.append("impossible: green dot (below 0) + wave overbought")
    if "wt_red_dot" in ids and "wt_below" in ids:
        out.append("impossible: red dot (above 0) + wave oversold")
    if any(r["rule"].endswith("_div") and r.get("a") == 0 for r in rules):
        out.append("broken: divergence 'in the last 0 candles'")
    if any(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
           and (r[k] > 1000 or (r["rule"] == "volume_spike" and ((k == "a" and r[k] >= 10) or (k == "b" and r[k] < 2))))
           for r in rules for k in ("a", "b")):
        out.append("broken: a price or a typo read as an indicator setting")
    elif any(r["rule"] in ("close_above_sma", "close_below_sma", "close_above_ema", "close_below_ema", "ema_above_ema",
                           "ema_below_ema", "ema_cross_up", "ema_cross_down") and isinstance(r.get(k), float)
             and not float(r[k]).is_integer() for r in rules for k in ("a", "b")):
        out.append("broken: a moving-average length that isn't a whole number (a price or typo read as a setting)")
    opp = (BEAR if s.get("direction") == "long" else BULL) & set(ids)
    if opp:
        out.append(f"backwards: a {s.get('direction')} on {', '.join(sorted(opp))}")
    if len(ids) > 12:
        out.append(f"list: {len(ids)} rules - everything the video mentioned, not a setup")
    if ids and not set(ids) & TRIGGERS:
        out.append("no trigger: only trend filters")
    return out


def judge_setup(s: dict, dupe_of: int | None) -> tuple[str, str] | None:
    """None = not tested yet (judged after the next 🎓 Learn)."""
    st = (s.get("stats") or {}).get("total") or {}
    if not s.get("rules"):
        return None
    if not st:
        return None
    ch = (s.get("stats") or {}).get("charts") or {}
    n, edge, avg = st.get("trades") or 0, st.get("edge") or 0, st.get("avg_r") or 0
    pos = sum(1 for v in ch.values() if (v.get("edge") or 0) > 0)
    line = f"{n} trades, {avg:+.2f}R a trade, edge {edge:+.2f}R vs random, positive on {pos} of {len(ch)} charts"
    probs = rule_problems(s)
    hard = [p for p in probs if p.startswith(("impossible", "broken"))]
    if dupe_of:
        return "drop", f"Duplicate of #{dupe_of} (same rules)."
    if hard:
        return "drop", "Its rules can't work: " + "; ".join(p.split(": ", 1)[1] for p in hard) + "."
    if any(p.startswith("list") for p in probs):
        return "drop", f"A list of everything the video mentioned ({len(s['rules'])} rules), not a setup."
    if any(p.startswith("backwards") for p in probs):
        why = next(p for p in probs if p.startswith("backwards")).split(": ", 1)[1]
        return "drop", f"Backwards: {why} - the signal points the other way. {line}."
    if n == 0:
        return "drop", "Never fired on years of prices - its rules (almost) never line up together."
    if n < 10:
        return "watch", f"Too few trades to judge ({n})."
    if edge <= 0.07:
        return "drop", f"No real edge: {line}."
    if n < 30:
        return "watch", f"Promising but too early: {line}."
    if any(p.startswith("no trigger") for p in probs):
        return "watch", f"Only trend filters, no entry trigger: {line}."
    if edge >= 0.15 and pos >= max(3, len(ch) * 0.6):
        return "use", f"Holds up: {line}."
    return "watch", f"Small or uneven edge - keep paper trading it: {line}."


# ---- lessons ----------------------------------------------------------------------------------------
JUNK_TITLE = re.compile(r"worship|sermon|shabbat|\bsong\b|music|giveaway|game day|vip games|\b\d{2,3}x leverage|"
                        r"exchange (full )?tutorial|sign up", re.I)


def judge_lesson(title: str, verdicts: list[str]) -> tuple[str, str]:
    if JUNK_TITLE.search(title or ""):
        return "drop", "Not trading material (from its title)."
    if len(verdicts) >= 6 and verdicts.count("drop") / len(verdicts) >= 0.8:
        return "drop", f"{verdicts.count('drop')} of the {len(verdicts)} things it said were junk."
    return "keep", "Real trading material."


# ---- running it -------------------------------------------------------------------------------------
def run(store, log=print) -> dict:
    """Review everything, apply the verdicts, save every verdict + rating + comment. Returns counts."""
    from . import review as claude
    state = store.meta_get("auto_review") or {}
    out = {"concepts": {}, "setups": {}, "lessons": {}}
    counts = {"concepts_deleted": 0, "setups_off": 0, "lessons_deleted": 0}
    loved = {(r["kind"], str(r["ref"])) for r in store.ratings() if r["stars"] >= 4}
    claude_c = {int(x["id"]) for x in claude.load().get("concepts", []) if "id" in x}
    claude_s = {int(x["id"]) for x in claude.load().get("setups", []) if "id" in x}
    claude_l = {int(x["id"]) for x in claude.load().get("lessons", []) if "id" in x}

    # concepts
    seen: dict = {}
    by_lesson: dict[int, list[str]] = {}
    rows = store.db.execute("SELECT id, lesson_id, text FROM concepts ORDER BY id").fetchall()
    drop_ids = []
    for cid, lid, text in rows:
        v, why = judge_concept(text, seen, cid)
        by_lesson.setdefault(lid, []).append(v)
        if cid in claude_c or ("concept", str(cid)) in loved:
            continue
        out["concepts"][str(cid)] = {"verdict": v, "stars": STARS[v], "comment": why, "text": str(text)[:80]}
        if v == "drop":
            drop_ids.append(cid)
    for cid in drop_ids:
        store.db.execute("DELETE FROM concepts WHERE id = ?", (cid,))
    counts["concepts_deleted"] = len(drop_ids)

    # lessons (after concepts, so junk lessons are known)
    for lid, title, status in store.db.execute("SELECT id, title, status FROM lessons").fetchall():
        if status == "learning" or lid in claude_l or ("lesson", str(lid)) in loved:
            continue
        v, why = judge_lesson(title, by_lesson.get(lid, []))
        out["lessons"][str(lid)] = {"verdict": v, "stars": STARS[v], "comment": why, "title": title}
        if v == "drop":
            store.forget_lesson(lid)
            counts["lessons_deleted"] += 1
            log(f"  🤖 deleted lesson '{title}': {why}")

    # setups
    def sig(s):
        t = (s.get("stats") or {}).get("total") or {}
        if (t.get("trades") or 0) >= 10:            # the same trades and edge = the same setup written differently
            return (s["direction"], "record", t.get("trades"), round(t.get("edge") or 0, 3), round(t.get("avg_r") or 0, 3))
        return (s["direction"], tuple(sorted((r["rule"], r.get("a"), r.get("b")) for r in s["rules"])))
    setups = store.setups()
    best: dict = {}
    for s in setups:
        if s.get("rules"):
            k = sig(s)
            e = ((s.get("stats") or {}).get("total") or {}).get("edge")
            if k not in best or (e if e is not None else -9) > (best[k][1] if best[k][1] is not None else -9):
                best[k] = (s["id"], e)
    for s in setups:
        protected = s["status"] == "pinned" or s.get("source") == "you" or s["id"] in claude_s \
            or ("setup", str(s["id"])) in loved
        dupe = best[sig(s)][0] if s.get("rules") and best[sig(s)][0] != s["id"] else None
        j = judge_setup(s, dupe)
        if j is None:
            continue
        v, why = j
        if protected and v == "drop":
            why += " (left on: you pinned it, made it, rated it, or Claude's review covers it)"
        out["setups"][str(s["id"])] = {"verdict": v, "stars": STARS[v], "comment": why, "name": s["name"]}
        if v == "drop" and not protected and s["status"] != "off":
            store.update_setup(s["id"], status="off")
            counts["setups_off"] += 1
    store.db.commit()
    state = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **out, "counts": counts}
    store.meta_set("auto_review", state)
    log(f"  🤖 auto-review: {counts['concepts_deleted']} concepts deleted, {counts['setups_off']} setups turned off, "
        f"{counts['lessons_deleted']} lessons deleted")
    return counts


def enabled() -> bool:
    try:
        from livepaper import settings
        return bool(settings.get("apprentice_auto_review", True))
    except Exception:
        return True


def verdict(store, kind: str, ref) -> dict | None:
    """The auto-review's verdict / rating / comment on one item (kind: concepts, setups, lessons)."""
    return ((store.meta_get("auto_review") or {}).get(kind) or {}).get(str(ref))
