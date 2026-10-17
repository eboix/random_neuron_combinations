from __future__ import annotations


def _strs(x, n):
    return isinstance(x, list) and len(x) == n and all(isinstance(t, str) and t.strip() for t in x)


def validate_explain(o):
    errs = []
    if not isinstance(o, dict):
        return ["not a JSON object"]
    for k in ("explanation", "fires_on"):
        if not isinstance(o.get(k), str) or not o[k].strip():
            errs.append(f"{k} must be a non-empty string")
    p = o.get("parts")
    if not (isinstance(p, list) and 1 <= len(p) <= 8 and all(isinstance(t, str) and t.strip() for t in p)):
        errs.append("parts must be a list of 1-8 non-empty strings")
    if not isinstance(o.get("confidence"), int) or not 1 <= o["confidence"] <= 5:
        errs.append("confidence must be an int 1-5")
    for k in ("predicted_fire", "near_miss", "distant"):
        if not _strs(o.get(k), 5):
            errs.append(f"{k} must be a list of exactly 5 non-empty strings")
    if "other_pole" in o and not isinstance(o["other_pole"], str):
        errs.append("other_pole must be a string")
    return errs


def validate_blindmax(o):
    if isinstance(o, dict) and _strs(o.get("texts"), 3):
        return []
    return ["texts must be a list of exactly 3 non-empty strings"]


def validate_simulate(o, n_items):
    if not (isinstance(o, dict) and isinstance(o.get("scores"), dict)):
        return ["scores must be an object mapping item id -> int"]
    errs = []
    for i in range(n_items):
        v = o["scores"].get(str(i))
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 10:
            errs.append(f"item {i}: score must be a number 0-10")
    return errs
