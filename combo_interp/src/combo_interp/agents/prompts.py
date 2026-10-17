"""Task texts for the three agent roles. All units get IDENTICAL instructions (condition-blind)."""

EXPLAIN = """# Task: explain one unit of a language model

You are given one scalar "unit" computed inside an intermediate layer of a large language model. For every token
of a text the unit has an activation. `view.md` (read it first) shows contexts where the unit fires most strongly,
at intermediate strengths, and at random, plus token statistics. `data.json` holds the same data, and
`python unit_tools.py --help` lists helpers (longer contexts, grep, token statistics).

You are NOT told how the unit is built and you must not try to find out: stay inside this folder and treat the
unit as a black box. Explain the PRIMARY pole (named at the top of view.md). Remarks about the other pole are optional.

## What good looks like
- Say what makes the unit fire AND where it fires relative to the pattern. Units in language models often fire
  not on the content words but on punctuation (a comma or period where the model summarises the context so far),
  on the token just before a cue becomes predictive, or on the token that completes a word fragment.
- Ask yourself: "if I could not use lists of nouns, how would I describe the shared pattern?" Prefer a shared
  pattern over a list of examples. Decode/translate odd-looking examples before concluding they are arbitrary.
- Weak firings (<30% of max) matter less than strong ones. Relevant context may be far before the firing token.
- Be honest about structure: a unit may have one job, a main job plus a closely related offshoot, or several
  unrelated jobs. List each *independent* trigger in `parts`. Merge triggers that are just one pattern described
  with different nouns; keep unrelated triggers separate. Do not force one story onto unrelated examples, and do
  not split one coherent pattern into many parts.
- Your explanation will be TESTED by other agents who see only your text: one writes texts meant to fire the unit
  maximally, another predicts activations on unseen contexts. Specific, falsifiable explanations score better.

## Output
Write `result.json` (valid JSON, exactly these keys):
{
  "explanation": "<= 60 words: what the primary pole responds to, and on which token(s)",
  "fires_on": "<= 20 words: token positions it fires on, relative to the pattern",
  "parts": ["one short phrase per independent trigger (1-8 items)"],
  "other_pole": "<= 30 words, or empty string",
  "confidence": <integer 1-5>,
  "predicted_fire": [5 short stand-alone texts (1-3 sentences each) you predict will make the unit fire strongly (>=30% of its max) on some token],
  "near_miss": [5 short stand-alone texts, very similar in topic/wording, that you predict will NOT make it fire],
  "distant": [5 short stand-alone texts about a concept as unrelated as possible to everything in your explanation; predicted not to fire],
  "notes": "optional free text"
}
Write the predicted / near-miss / distant texts now; you will get no feedback on them. The very first token of a
text is ignored when scoring. Then stop.
"""

BLINDMAX = """# Task: write texts that maximally activate a described unit

Below is an explanation of one unit of a language model. You see ONLY this explanation, never the unit or any
data. Write 3 different candidate texts (20-120 words each) designed to make the unit fire as strongly as
possible, following the explanation literally and putting the firing pattern exactly where `fires_on` says it
fires. Vary the three candidates (different wording, different contexts). The very first token of a text is ignored.

explanation: {explanation}
fires_on: {fires_on}

Write `result.json` (valid JSON): {{"texts": ["...", "...", "..."]}}. Then stop.
"""

SIMULATE = """# Task: predict how strongly a described unit fires on given contexts

Below is an explanation of one unit of a language model. `items.json` contains {n} short contexts; in each, one token
is marked `[[like this]]`. For every item, predict how strongly the unit fires on the MARKED token, as an integer
0-10 (0 = not at all, 10 = as strongly as the best examples of the explanation). Judge each item independently,
use the whole range, and rely only on the explanation.

explanation: {explanation}
fires_on: {fires_on}

Write `result.json` (valid JSON): {{"scores": {{"<item id>": <int 0-10>, ...}}}} with one entry per item. Then stop.
"""

RUN_PROMPT = "Read TASK.md in this folder and follow it exactly. Write result.json as specified, then stop."
