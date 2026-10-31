"""Pull trade calls, rules and price levels out of a transcript chunk with an LLM.

Backends:
  ollama  (default) a model running locally in Ollama (https://ollama.com), e.g.
          `ollama pull qwen2.5:14b`. Nothing leaves your computer.
  claude  Anthropic's API (pip install anthropic, ANTHROPIC_API_KEY or `ant auth login`).
          More accurate on long, rambling transcripts; costs money per stream.
"""

from __future__ import annotations

import json

import requests

PATTERN_KINDS = ["ascending_triangle", "descending_triangle", "symmetrical_triangle", "rising_wedge",
                 "falling_wedge", "ascending_channel", "descending_channel", "range", "bull_flag", "bear_flag",
                 "pennant", "broadening", "double_top", "double_bottom", "head_shoulders",
                 "inverse_head_shoulders", "cup_handle", "other"]

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["calls", "rules", "levels", "patterns"],
    "properties": {
        "calls": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["time", "setup_start", "setup_end", "summary", "asset", "direction", "kind", "timeframe",
                         "entry", "stop", "targets", "conditions", "quote"],
            "properties": {
                "time": {"type": "string", "description": "[m:ss] timestamp where it was said"},
                "setup_start": {"type": "string", "description": "[m:ss] where he starts explaining this setup"},
                "setup_end": {"type": "string", "description": "[m:ss] where he finishes explaining it"},
                "summary": {"type": "string",
                            "description": "the setup restated plainly in 1-3 sentences: what he wants to trade, "
                                           "what must happen first, entry, stop, targets, and why"},
                "asset": {"type": "string", "description": "ticker, e.g. BTC, ETH, SOL, SPY"},
                "direction": {"type": "string", "enum": ["long", "short"]},
                "kind": {"type": "string", "enum": ["entered", "plan", "idea"],
                         "description": "entered = he says he is in the trade now; plan = a specific setup "
                                        "he will take if conditions happen; idea = a looser directional view"},
                "timeframe": {"type": "string", "description": "chart timeframe mentioned, or empty"},
                "entry": {"type": ["number", "null"]},
                "stop": {"type": ["number", "null"]},
                "targets": {"type": "array", "items": {"type": "number"}},
                "conditions": {"type": "array", "items": {"type": "string"},
                               "description": "what must happen first (e.g. 'price reclaims 64k', '4h green dot')"},
                "quote": {"type": "string", "description": "his words, verbatim, max ~40 words"},
            }}},
        "rules": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["time", "category", "rule", "quote"],
            "properties": {
                "time": {"type": "string"},
                "category": {"type": "string", "enum": ["entry", "exit", "stop", "risk", "levels", "indicator",
                                                         "timeframe", "psychology", "other"]},
                "rule": {"type": "string", "description": "the rule as one clear sentence"},
                "quote": {"type": "string"},
            }}},
        "patterns": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["time", "setup_start", "setup_end", "asset", "timeframe", "pattern", "pattern_name", "bias",
                         "top_level", "bottom_level", "if_breaks_up", "if_breaks_down", "at_apex", "targets",
                         "summary", "quote"],
            "properties": {
                "time": {"type": "string", "description": "[m:ss] where he points the pattern out"},
                "setup_start": {"type": "string"}, "setup_end": {"type": "string"},
                "asset": {"type": "string"},
                "timeframe": {"type": "string", "description": "chart timeframe, or empty"},
                "pattern": {"type": "string", "enum": PATTERN_KINDS},
                "pattern_name": {"type": "string", "description": "what he calls it, in his words"},
                "bias": {"type": "string", "enum": ["up", "down", "unclear"],
                         "description": "which way HE expects it to break"},
                "top_level": {"type": ["number", "null"],
                              "description": "price of the top line / resistance / neckline he marks, or null"},
                "bottom_level": {"type": ["number", "null"],
                                 "description": "price of the bottom line / support / neckline he marks, or null"},
                "if_breaks_up": {"type": "string", "description": "what he says price will do / he will do if it "
                                                                  "breaks up; empty if he doesn't say"},
                "if_breaks_down": {"type": "string", "description": "same for a break down"},
                "at_apex": {"type": "string", "description": "anything he says about price reaching the point where "
                                                             "the lines converge, or empty"},
                "targets": {"type": "array", "items": {"type": "number"}},
                "summary": {"type": "string", "description": "the pattern and his plan restated in 1-3 sentences"},
                "quote": {"type": "string"},
            }}},
        "levels": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["time", "asset", "price", "kind"],
            "properties": {
                "time": {"type": "string"},
                "asset": {"type": "string"},
                "price": {"type": "number"},
                "kind": {"type": "string", "enum": ["support", "resistance", "target", "level"]},
            }}},
    },
}

SYSTEM = """You extract trading information from a livestream transcript for a study log.
The transcript is auto-generated: numbers may be written as words ("sixty four thousand" = 64000,
"64k" = 64000, "sixty four five" for BTC usually = 64500). Each line starts with its [m:ss] time.

Rules:
- Only record what the speaker actually says. Never invent prices, stops or targets: use null / [] when
  he doesn't give them. Do not record viewers' questions as his calls unless he endorses them.
- A call needs an asset and a direction. Record each distinct call once (the clearest mention).
- Rules are general, reusable statements about how he trades ("I only short at a level", "stop goes
  above the last swing high"), not one-off comments.
- Levels are specific prices he marks as support, resistance or targets.
- Patterns: every chart pattern he points out (triangle, wedge, flag, channel, range, double top/bottom,
  head and shoulders, ...), even without a trade. Record the lines' prices he gives, which way he expects
  the break, what he says happens on a break up and on a break down, and anything about price reaching
  the point where the lines converge. Leave fields empty / null when he doesn't say.
- Quotes must be verbatim from the transcript.
- setup_start / setup_end bracket his whole explanation of that setup (from where he starts setting it
  up to where he moves on), so someone can replay exactly that part of the stream.
Return JSON matching the schema."""


def with_examples(examples: str) -> str:
    """The base instructions plus the reviewer's past verdicts (how the agent learns)."""
    return SYSTEM if not examples else f"{SYSTEM}\n\n{examples}"


def prompt(chunk: str, title: str, date: str) -> str:
    return f"Stream: {title} ({date})\n\nTranscript excerpt:\n{chunk}"


class OllamaBackend:
    def __init__(self, model: str = "qwen2.5:14b", url: str = "http://localhost:11434"):
        self.model, self.url = model, url.rstrip("/")

    def extract(self, chunk: str, title: str, date: str, system: str | None = None) -> dict:
        try:
            r = self._post(chunk, title, date, system or SYSTEM)
        except requests.ConnectionError:
            raise RuntimeError(f"Ollama isn't running at {self.url}: start the Ollama app (or `ollama serve`), "
                               f"or set backend = \"claude\" in streamlearn.toml") from None
        if r.status_code == 404:
            raise RuntimeError(f"model {self.model} isn't installed: run `ollama pull {self.model}`")
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])

    def _post(self, chunk: str, title: str, date: str, system: str):
        return requests.post(f"{self.url}/api/chat", timeout=600, json={
            "model": self.model, "stream": False, "format": SCHEMA, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt(chunk, title, date)}]})


class ClaudeBackend:
    """Claude Opus 5.5 with structured outputs. Server-side fallbacks are on, so a
    request a safety classifier declines is re-run on a fallback model in the same call."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", api_key: str | None = None):
        import anthropic
        # api_key from the Account tab if given; otherwise the SDK finds ANTHROPIC_API_KEY / `ant auth login`
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model, self.effort = model, effort

    def extract(self, chunk: str, title: str, date: str, system: str | None = None) -> dict:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=system or SYSTEM,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": prompt(chunk, title, date)}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined this chunk: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Claude's answer was cut off (max_tokens); use smaller chunks")
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)


def backend(cfg: dict):
    kind = cfg.get("backend", "ollama")
    if kind == "claude":
        return ClaudeBackend(cfg.get("claude_model", "claude-opus-5-5"), cfg.get("claude_effort", "medium"),
                             cfg.get("anthropic_api_key") or None)
    if kind == "ollama":
        return OllamaBackend(cfg.get("ollama_model", "qwen2.5:14b"), cfg.get("ollama_url", "http://localhost:11434"))
    raise SystemExit(f"unknown backend {kind} (use ollama or claude)")


def mmss(t: str) -> int:
    """'[12:34]' / '12:34' / '1:02:03' -> seconds (0 if unparseable)."""
    parts = t.strip("[] ").split(":")
    try:
        nums = [int(float(p)) for p in parts]
    except ValueError:
        return 0
    secs = 0
    for n in nums:
        secs = secs * 60 + n
    return secs
