# Notes for Claude sessions

## Reviewing the apprentice (the user asked for this to be routine)

The desktop app shares its data hourly to the `paper-trades` branch: `apprentice/` (lessons, concepts,
setups.json with rules in words + track record, journal, the user's ratings), `paper_trades/` (trades, fills,
runs), `streams/` (stream-agent setups, rules, patterns) and `SUMMARY.md`.

When the user asks how the apprentice / paper trading is doing, or to review what it learned:

1. `python tools/claude_review.py show` - fetches the branch and prints a digest (setups with ids, rules,
   stats; concepts; lessons). If nothing is shared, tell the user to turn on "Share with Claude" in
   ⚙️ Account and add a GitHub token (ZIP installs can't push with git).
2. Judge each setup: **use** (sound logic, rules agree on direction, positive edge out of sample, enough
   trades, works on several charts), **watch** (too early, < ~30 trades, or an edge that looks fragile -
   left running, nothing changes), **drop** (no/negative edge, contradictory rules that can never fire,
   duplicates), **fix** (good idea, wrong rules - give corrected `mcbot.builder` rule ids in `fix.rules`;
   classic mistake: a lower-band long paired with a *bearish* divergence, or an upper-band short with a
   *bullish* one). Judge concepts: **keep / drop / fix** (drop ads, outros, sponsor text, vague filler;
   fix garbled wording with `fix_text`). Lessons: **keep / drop**.
3. Write the review JSON (format in `apprentice/review.py`), `python tools/claude_review.py check review.json`,
   then `publish review.json`. It lands in `reviews/claude_review.json`; the app downloads it on its next
   sync and (by default) applies it automatically - setups, concepts, lessons marked drop are deleted, chart-trade
   ratings saved - then re-tests with 🎓 Learn. The user can switch that off in 🧠 Apprentice → 🤖 Automatic review.
   The app also runs its own rule-based review (`apprentice/autoreview.py`) after every lesson and Learn; your
   verdicts take precedence over it.
4. Tell the user the headline: what to keep, what's broken and why, and that they can press Apply all.

Keep reviews honest: small samples (< ~30 trades) are "too early", and edge vs random matters more than
win rate.
