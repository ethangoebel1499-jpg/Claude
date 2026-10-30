"""The playbook (his recurring rules, grouped) and the scorecard (how his calls did)."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9% ]", " ", s.lower())).strip()


def group_rules(rules: pd.DataFrame, streams: pd.DataFrame, threshold: float = 0.72) -> list[dict]:
    """Cluster near-identical rules; count how many streams repeat each one."""
    if rules.empty:
        return []
    titles = streams.set_index("id")[["title", "start"]].to_dict("index")
    groups: list[dict] = []
    for _, r in rules.iterrows():
        n = _norm(r.rule)
        for g in groups:
            if g["category"] == r.category and SequenceMatcher(None, n, g["norm"]).ratio() >= threshold:
                g["examples"].append(r)
                g["streams"].add(r.stream_id)
                break
        else:
            groups.append({"category": r.category, "rule": r.rule, "norm": n, "examples": [r], "streams": {r.stream_id}})
    out = []
    for g in groups:
        ex = g["examples"][0]
        st = titles.get(ex.stream_id, {})
        out.append({"category": g["category"], "rule": g["rule"], "streams": len(g["streams"]),
                    "mentions": len(g["examples"]), "quote": ex.quote,
                    "source": f"{st.get('title', '?')} @ {ex.t // 60}:{ex.t % 60:02d}"})
    return sorted(out, key=lambda x: (-x["streams"], -x["mentions"]))


def playbook_markdown(groups: list[dict], n_streams: int) -> str:
    lines = [f"# Playbook from {n_streams} streams", "",
             "Rules he states, grouped; **streams** = how many streams repeat it (the stronger the habit, "
             "the higher). Each comes with his own words and where he said it.", ""]
    for cat in ("entry", "exit", "stop", "risk", "levels", "indicator", "timeframe", "psychology", "other"):
        sub = [g for g in groups if g["category"] == cat]
        if not sub:
            continue
        lines += [f"## {cat.capitalize()}", ""]
        for g in sub:
            lines.append(f"- **{g['rule']}** ({g['streams']} streams) - \"{g['quote']}\" ({g['source']})")
        lines.append("")
    return "\n".join(lines)


def scorecard(calls: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Win rate on calls with levels, 'right direction after 7 days' on every call."""
    if calls.empty:
        return {}
    c = calls.copy()
    c["win"] = c.result.eq("win")
    c["closed"] = c.status.eq("closed")
    c["right7"] = c.ret_7d > 0

    def agg(g):
        closed = g[g.closed]
        r7 = g.ret_7d.dropna()
        return pd.Series({
            "calls": len(g),
            "with levels, closed": len(closed),
            "win rate": closed.win.mean() * 100 if len(closed) else float("nan"),
            "avg R": closed.r_multiple.mean() if len(closed) else float("nan"),
            "right after 7d": (r7 > 0).mean() * 100 if len(r7) else float("nan"),
            "avg 7d move": r7.mean() if len(r7) else float("nan"),
        })

    return {"overall": agg(c).to_frame("all calls").T,
            "by kind": c.groupby("kind").apply(agg, include_groups=False),
            "by asset": c.groupby("asset").apply(agg, include_groups=False).sort_values("calls", ascending=False),
            "by direction": c.groupby("direction").apply(agg, include_groups=False)}


def md_table(df: pd.DataFrame) -> str:
    def cell(v):
        return "-" if isinstance(v, float) and pd.isna(v) else f"{v:,.2f}".rstrip("0").rstrip(".") \
            if isinstance(v, float) else str(v)
    head = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    rows = ["| " + " | ".join([str(i)] + [cell(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join([head, "|" + "---|" * (len(df.columns) + 1)] + rows)


def scorecard_markdown(sc: dict[str, pd.DataFrame]) -> str:
    if not sc:
        return "# Scorecard\n\nNo calls yet.\n"
    lines = ["# Call scorecard", "",
             "**Win rate**: calls with a stop and target, target hit before stop within 14 days. "
             "**Right after 7d**: price moved his way 7 days after the call (every call). Checked on Kraken candles.", ""]
    for name, df in sc.items():
        lines += [f"## {name.capitalize()}", "", md_table(df), ""]
    return "\n".join(lines)
