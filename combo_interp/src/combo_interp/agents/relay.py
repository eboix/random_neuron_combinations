"""File-based relay between explainer workspaces and the query service.

Headless agents may not be allowed to run any command or tool beyond reading and writing files in their own folder
(the case for Enterprise claude.ai accounts that require approval for every shell / MCP call). So the explainer asks
by writing a file, and this relay, running in the 05_run_agents.py process during the explain stage, answers:

    <workspace>/queries/001.txt       one text per line (\\n inside a line = a line break in the text)
    <workspace>/queries/001.out.md    written by the relay: activations in the view.md format + remaining budget

The unit is the workspace folder name, so an agent can only query its own unit. Budget and logging are enforced by
the query service (query_server.py); the relay only forwards.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_request(text: str) -> list[str]:
    return [line.replace("\\n", "\n") for line in text.splitlines() if line.strip()]


def render(tokens, fracs) -> str:
    """view.md markers: [[tok|f]] = strongest scored token, «tok|f» = other tokens >= 0.10 of the corpus max."""
    scored = [j for j, x in enumerate(fracs) if x is not None]
    best = max(scored, key=lambda j: fracs[j]) if scored else None
    out = []
    for j, (t, x) in enumerate(zip(tokens, fracs)):
        t = t.replace("\n", "↵")
        if j == best:
            out.append(f"[[{t}|{x:.2f}]]")
        elif x is not None and x >= 0.1:
            out.append(f"«{t}|{x:.2f}»")
        else:
            out.append(t)
    return "".join(out)


def format_response(texts: list[str], resp: dict) -> str:
    lines = []
    for i, r in enumerate(resp.get("results", [])):
        mx = "n/a" if r["max_frac"] is None else f"{r['max_frac']:.2f}"
        lines.append(f"{i}. [max {mx}] {render(r['tokens'], r['fracs'])}")
        if r.get("truncated_from"):
            lines.append(f"   (TRUNCATED: only the first {len(r['tokens'])} of {r['truncated_from']} tokens were run)")
        lines.append("")
    if resp.get("error"):
        lines.append(f"NOTE: {resp['error']}")
    if "remaining" in resp:
        lines.append(f"Texts left in this unit's query budget: {resp['remaining']}")
    return "\n".join(lines) + "\n"


def post(url: str, blind: str, texts: list[str], timeout: float = 600) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/query", data=json.dumps({"blind_id": blind, "texts": texts}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return {"error": json.loads(e.read()).get("error", str(e))}
        except ValueError:
            return {"error": str(e)}
    except OSError as e:
        return {"error": f"query service unreachable ({e}); try again later"}


class QueryRelay:
    """Polls <root>/<blind_id>/queries/*.txt and writes the matching .out.md files."""

    def __init__(self, root, url: str, poll: float = 0.3, settle: float = 0.3):
        self.root, self.url, self.poll, self.settle = Path(root), url, poll, settle
        self._stop = threading.Event()
        self._thread = None
        self.n_answered = 0

    def process_once(self):
        for req in sorted(self.root.glob("*/queries/*.txt")):
            out = req.with_name(req.stem + ".out.md")
            if out.exists() or time.time() - req.stat().st_mtime < self.settle:   # answered / maybe still being written
                continue
            texts = parse_request(req.read_text())
            resp = post(self.url, req.parent.parent.name, texts) if texts else {"error": "no texts in this file"}
            tmp = out.with_name(out.name + ".tmp")
            tmp.write_text(format_response(texts, resp))
            tmp.replace(out)
            self.n_answered += 1

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.process_once()
            except OSError:
                pass                       # a workspace was removed / recreated mid-scan; retry next round
            self._stop.wait(self.poll)

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="query-relay", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
