"""How the agent learns from your reviews: past verdicts become worked examples in its
instructions. Awarded (correct) extractions show it what to do; punished (wrong) ones, with
your note, show it what not to do."""

from __future__ import annotations

import json

import pandas as pd


def examples(store, n_good: int = 6, n_bad: int = 6) -> str:
    rev = pd.read_sql_query("SELECT * FROM reviews ORDER BY reviewed DESC", store.db)
    if rev.empty:
        return ""
    lines = ["Feedback from the reviewer on your earlier extractions. Learn from it:"]
    good = [r for r in rev.itertuples() if r.verdict == "correct"][:n_good]
    bad = [r for r in rev.itertuples() if r.verdict == "wrong"][:n_bad]
    for r in good:
        call = json.loads(r.corrected or "{}").get("call", {})
        if call.get("quote"):
            fields = {k: call.get(k) for k in ("asset", "direction", "kind", "entry", "stop", "targets")}
            lines.append(f'- CORRECT (confirmed): "{call["quote"]}" -> {json.dumps(fields)}'
                         + (f" Reviewer: {r.note}" if r.note else ""))
    for r in bad:
        call = json.loads(r.corrected or "{}").get("call", {})
        if call.get("quote"):
            lines.append(f'- WRONG, do not repeat: from "{call["quote"]}" you extracted '
                         f'{call.get("direction")} {call.get("asset")} entry {call.get("entry")}. '
                         f"Reviewer: {r.note or 'not a real trade setup'}")
    return "\n".join(lines) if len(lines) > 1 else ""


def accuracy(store) -> tuple[int, int]:
    """(awarded, punished) so far."""
    rev = pd.read_sql_query("SELECT verdict FROM reviews", store.db)
    return int((rev.verdict == "correct").sum()), int((rev.verdict == "wrong").sum())
