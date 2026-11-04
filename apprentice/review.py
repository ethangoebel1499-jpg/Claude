"""Claude's review of what the apprentice learned (reviews/claude_review.json on the paper-trades branch,
downloaded by every sync into runtime/claude_review.json).

  {"reviewed_at": "...", "summary": "...",
   "setups":   [{"id": 12, "name": "...", "verdict": "use" | "drop" | "fix", "reason": "...",
                 "fix": {"rules": [...builder rules...], "direction": "long", "stop_atr": 1.5, "rr": 2} }],
   "concepts": [{"id": 40, "verdict": "keep" | "drop" | "fix", "reason": "...", "fix_text": "..."}],
   "lessons":  [{"id": 3, "verdict": "keep" | "drop", "reason": "..."}]}

Nothing changes until you press Apply: use -> 📌 always use, drop -> 🚫 off (setups) / deleted (concepts),
fix -> the corrected rules (re-tested on the next 🎓 Learn) or the corrected wording.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "runtime" / "claude_review.json"
APPLIED = ROOT / "runtime" / "claude_review_applied.json"
ICON = {"use": "✅", "keep": "✅", "drop": "❌", "fix": "🔧"}


def load() -> dict:
    try:
        return json.loads(LOCAL.read_text())
    except Exception:
        return {}


def _items(kind: str) -> dict[int, dict]:
    return {int(x["id"]): x for x in load().get(kind, []) if "id" in x}


def for_setup(setup: dict) -> dict | None:
    item = _items("setups").get(int(setup["id"]))
    if item and item.get("name") and item["name"] != setup.get("name"):
        return None                             # ids were reused for a different setup: don't misapply
    return item


def for_concept(cid: int) -> dict | None:
    return _items("concepts").get(int(cid))


def for_lesson(lid: int) -> dict | None:
    return _items("lessons").get(int(lid))


def _applied() -> set[str]:
    try:
        return set(json.loads(APPLIED.read_text()))
    except Exception:
        return set()


def _mark(key: str) -> None:
    done = _applied() | {key}
    APPLIED.parent.mkdir(parents=True, exist_ok=True)
    APPLIED.write_text(json.dumps(sorted(done)))


def key(kind: str, item: dict) -> str:
    return f"{load().get('reviewed_at', '')}:{kind}:{item['id']}"


def is_applied(kind: str, item: dict) -> bool:
    return key(kind, item) in _applied()


def describe(item: dict) -> str:
    v = item.get("verdict", "")
    return f"{ICON.get(v, '🤖')} Claude: **{v.upper()}** - {item.get('reason', '')}"


def apply_setup(store, setup: dict, item: dict) -> str:
    v = item.get("verdict")
    if v == "use":
        store.update_setup(setup["id"], status="pinned")
        out = "pinned (always used)"
    elif v == "drop":
        store.update_setup(setup["id"], status="off")
        out = "turned off"
    elif v == "fix":
        fix = item.get("fix") or {}
        fields = {k: fix[k] for k in ("rules", "direction", "stop_atr", "rr") if fix.get(k) is not None}
        store.update_setup(setup["id"], status="candidate", **fields)
        out = "fixed - re-tested on the next 🎓 Learn"
    else:
        return "nothing to apply"
    _mark(key("setups", item))
    return out


def apply_concept(store, cid: int, item: dict) -> str:
    v = item.get("verdict")
    if v == "drop":
        store.db.execute("DELETE FROM concepts WHERE id = ?", (int(cid),))
        out = "deleted"
    elif v == "fix" and item.get("fix_text"):
        store.db.execute("UPDATE concepts SET text = ? WHERE id = ?", (item["fix_text"], int(cid)))
        out = "reworded"
    else:
        out = "kept"
    store.db.commit()
    _mark(key("concepts", item))
    return out


def apply_all(store) -> dict[str, int]:
    """Apply every review item not applied yet. Returns counts."""
    counts = {"setups": 0, "concepts": 0}
    setups = {s["id"]: s for s in store.setups()}
    for sid, item in _items("setups").items():
        s = setups.get(sid)
        if s and for_setup(s) and not is_applied("setups", item) and item.get("verdict") in ("use", "drop", "fix"):
            apply_setup(store, s, item)
            counts["setups"] += 1
    have = {int(r) for r in store.table("concepts").id} if len(store.table("concepts")) else set()
    for cid, item in _items("concepts").items():
        if cid in have and not is_applied("concepts", item) and item.get("verdict") in ("drop", "fix"):
            apply_concept(store, cid, item)
            counts["concepts"] += 1
    return counts
