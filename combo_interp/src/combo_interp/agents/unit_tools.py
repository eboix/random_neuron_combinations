#!/usr/bin/env python3
"""Helpers for exploring ONE unit's data (stdlib only). Run from the unit's folder: python unit_tools.py <cmd>."""
import argparse
import json
import re
import sys


def load():
    with open("data.json") as f:
        return json.load(f)


def pick(d, pole, tier):
    if tier == "top":
        return d["examples"]["primary" if pole == "primary" else "other"]
    if tier == "random":
        return d["random"]
    return d["tiers"][tier]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="print examples")
    s.add_argument("--pole", choices=["primary", "other"], default="primary")
    s.add_argument("--tier", choices=["top", "strong", "moderate", "weak", "random"], default="top")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--long", action="store_true", help="longer context (only the strongest primary examples have it)")
    t = sub.add_parser("tokens", help="firing / next-token statistics")
    t.add_argument("--pole", choices=["primary", "other"], default="primary")
    g = sub.add_parser("grep", help="search all example texts with a regex")
    g.add_argument("pattern")
    sub.add_parser("stats", help="summary numbers")
    a = ap.parse_args()
    d = load()
    if a.cmd == "show":
        ex = pick(d, a.pole, a.tier)[a.start:a.start + a.n]
        if not ex:
            print("(no examples)")
        for e in ex:
            txt = e["long"] if (a.long and e.get("long")) else e["text"]
            print(f"{e['i']:>2}. [{e['frac']:.2f}] {txt}\n")
    elif a.cmd == "tokens":
        ts = d["token_stats"][a.pole]
        print("fires on:", ts["fires_on"]); print("next token:", ts["next_token"])
    elif a.cmd == "grep":
        rx = re.compile(a.pattern)
        n = 0
        for name, lst in (("primary", d["examples"]["primary"]), ("other", d["examples"]["other"]), ("random", d["random"]),
                          *[(k, v) for k, v in d["tiers"].items()]):
            for e in lst:
                if rx.search(e["text"]) or (e.get("long") and rx.search(e["long"])):
                    print(f"[{name} {e['i']} frac={e['frac']:.2f}] {e['text']}\n"); n += 1
        print(f"{n} matches")
    elif a.cmd == "stats":
        print(json.dumps({"layer": d["layer"], "primary_pole": d["primary_pole"],
                          "n_examples": {k: len(v) for k, v in d["tiers"].items()} |
                          {"primary": len(d["examples"]["primary"]), "other": len(d["examples"]["other"]),
                           "random": len(d["random"])}}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
