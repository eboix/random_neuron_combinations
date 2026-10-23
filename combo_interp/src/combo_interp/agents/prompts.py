"""Task texts for the three agent roles. All units get IDENTICAL instructions (condition-blind).

explain   sees the unit's examples (and, optionally, can query the model) -> explanation
write     sees ONLY the explanation -> texts meant to fire the unit maximally, plus predicted-fire / near-miss / distant
simulate  sees ONLY the explanation -> predicted activations on held-out contexts
"""

EXPLAIN = """# Task: explain one unit of a language model

You are given one scalar "unit" computed inside an intermediate layer of a large language model. For every token
of a text the unit has an activation. `view.md` (read it first) shows contexts where the unit fires most strongly,
at intermediate strengths, and at random, plus token statistics. `long_contexts.md` shows several hundred tokens of
context before the strongest firings, and `data.json` holds the same data as view.md. Work only by reading these files.

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
- Your explanation will be TESTED by other agents who see only your text: one writes new texts that should and
  should not make the unit fire, another predicts the unit's activation on unseen contexts. Specific, falsifiable
  explanations score better.

## Output
Write `result.json` (valid JSON, exactly these keys):
{
  "explanation": "<= 60 words: what the primary pole responds to, and on which token(s)",
  "fires_on": "<= 20 words: token positions it fires on, relative to the pattern",
  "parts": ["one short phrase per independent trigger (1-8 items)"],
  "other_pole": "<= 30 words, or empty string",
  "confidence": <integer 1-5>,
  "notes": "optional free text"
}
Then stop.
"""

QUERY_ADDENDUM = """
## Testing hypotheses on the model (before writing result.json)
You can run your own texts through the model and see this unit's activation on every token. To do so, write a file
`queries/001.txt` (then `queries/002.txt`, `queries/003.txt`, ...) with one text per line (write \\n for a line break
inside a text; at most {per_call} texts per file). The answer appears a moment later as `queries/001.out.md`, in the same
format as view.md; if it is not there yet when you read it, read it again. You have {budget} texts in total for this
unit; every answer says how many are left. Use them to test and sharpen your hypothesis, e.g. minimal pairs that change
one feature, texts that should fire and close variants that should not.
"""

WRITE = """# Task: write test texts for a described unit

Below is an explanation of one unit (a scalar computed at every token) of a language model. You see ONLY this
explanation, never the unit or any data. Follow the explanation literally, putting the firing pattern exactly where
`fires_on` says the unit fires. Write:

- "max_texts": 3 different texts (20-120 words each) designed to make the unit fire as strongly as possible. Vary them
  (different wording, different contexts).
- "predicted_fire": 5 short stand-alone texts (1-3 sentences) that should make the unit fire strongly on some token.
- "near_miss": 5 short stand-alone texts, each very close in topic and wording to a predicted_fire text but missing
  what the explanation says triggers the unit, so it should NOT fire.
- "distant": 5 short stand-alone texts about things as unrelated as possible to the explanation; should not fire.

The very first token of a text is ignored.

explanation: {explanation}
fires_on: {fires_on}

Write `result.json` (valid JSON): {{"max_texts": [3 texts], "predicted_fire": [5 texts], "near_miss": [5 texts],
"distant": [5 texts]}}. Then stop.
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
