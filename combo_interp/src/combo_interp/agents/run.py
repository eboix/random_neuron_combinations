"""Run the three agent stages (explain -> write, simulate; then one GPU probe round) over the selected units.

Backends: 'claude' (headless Claude Code CLI, `claude -p`) and 'fake' (deterministic stand-in for plumbing tests).
Each unit gets an isolated workspace folder containing ONLY that unit's blinded files. Results are validated and
copied to paths.agent_results(stage)/<blind_id>.json. Resumable (existing results are skipped).
"""
from __future__ import annotations

import json
import re
import shutil
import zlib
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..io_utils import read_json, write_json
from . import prompts
from .validate import validate_explain, validate_simulate, validate_write

_MARK = [(re.compile(r"\[\[(.*?)\|[-0-9.nae]+\]\]"), r"\1"), (re.compile(r"«(.*?)\|[-0-9.nae]+»"), r"\1"),
         (re.compile(r"\[\[(.*?)\]\]"), r"\1")]


def strip_marks(t: str) -> str:
    for rx, rep in _MARK:
        t = rx.sub(rep, t)
    return t.replace("↵", "\n")


# ------------------------------------------------------------------ workspaces
def render_long_contexts(d) -> str:
    """Longer context (several hundred tokens before the firing token) for the strongest primary examples."""
    lines = [f"# Unit {d['blind_id']}: longer contexts for the strongest firings (primary pole)",
             "Same markers as view.md; only the selected token carries its activation here.", ""]
    for e in d["examples"]["primary"]:
        if e.get("long"):
            lines += [f"## Example {e['i']} [{e['frac']:.2f}]", e["long"], ""]
    return "\n".join(lines) + "\n"


def _query_cfg(cfg):
    q = (cfg.agents.get("query") if cfg is not None else None) or {}
    return q if q.get("enabled") else None


def _prepare(stage, blind, paths, cfg=None):
    work = paths.agent_work(stage) / blind
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    if stage == "explain":
        shutil.copy(paths.agent_cache / f"{blind}.json", work / "data.json")
        shutil.copy(paths.agent_cache / f"{blind}.md", work / "view.md")
        (work / "long_contexts.md").write_text(render_long_contexts(read_json(work / "data.json")))
        task = prompts.EXPLAIN
        q = _query_cfg(cfg)
        if q:
            (work / "queries").mkdir()
            task += prompts.QUERY_ADDENDUM.format(budget=int(q["budget_texts"]), per_call=int(q["max_texts_per_call"]))
        (work / "TASK.md").write_text(task)
        return work, None
    ex = read_json(paths.agent_results("explain") / f"{blind}.json")
    if stage == "write":
        (work / "TASK.md").write_text(prompts.WRITE.format(explanation=ex["explanation"], fires_on=ex["fires_on"]))
        return work, None
    items = read_json(paths.agent_cache / "holdout" / f"{blind}.json")
    write_json(work / "items.json", items, indent=1)
    (work / "TASK.md").write_text(prompts.SIMULATE.format(n=len(items), explanation=ex["explanation"], fires_on=ex["fires_on"]))
    return work, len(items)


def _validate(stage, obj, n_items):
    if stage == "explain":
        return validate_explain(obj)
    if stage == "write":
        return validate_write(obj)
    return validate_simulate(obj, n_items)


# ------------------------------------------------------------------ backends
def _fake(stage, work, paths, blind, rng):
    """Plumbing stand-in. Unlike a real writer, the fake one peeks at the unit's own examples (so scores are not
    pure noise in tests); real writers see only the explanation."""
    if stage == "explain":
        d = read_json(work / "data.json")
        toks = [t for t, _ in d["token_stats"]["primary"]["fires_on"][:3]]
        notes = "fake backend"
        if (work / "queries").is_dir():                     # query through the relay like a real agent would
            top = [strip_marks(e["text"]).replace("\n", "\\n") for e in d["examples"]["primary"][:2]]
            (work / "queries" / "001.txt").write_text("\n".join(top) + "\n")
            out = work / "queries" / "001.out.md"
            for _ in range(200):
                if out.exists():
                    break
                time.sleep(0.05)
            notes += " | query answer: " + (out.read_text()[-80:] if out.exists() else "none")
        obj = {"explanation": f"Fires on tokens such as {toks}.", "fires_on": "on the listed tokens",
               "parts": [repr(t) for t in toks[:2]] or ["unclear"],     # repr: whitespace tokens stay non-empty
               "other_pole": "", "confidence": 3, "notes": notes}
    elif stage == "write":
        d = read_json(paths.agent_cache / f"{blind}.json")
        top = [strip_marks(e["text"]) for e in d["examples"]["primary"]]
        rnd = [strip_marks(e["text"]) for e in d["random"]]
        pad = lambda lst, n, fill: (lst + [fill] * n)[:n]
        obj = {"max_texts": pad(top, 3, "alpha beta gamma"), "predicted_fire": pad(top[3:], 5, "alpha beta"),
               "near_miss": pad(rnd, 5, "gamma delta"),
               "distant": ["The recipe calls for two cups of flour.", "Quantum chromodynamics is hard.",
                           "He sailed across the bay at dawn.", "Invoice number 4471 is overdue.",
                           "The choir rehearsed on Thursday."]}
    else:
        items = read_json(work / "items.json")
        obj = {"scores": {str(i["item"]): int(rng.integers(0, 11)) for i in items}}
    (work / "result.json").write_text(json.dumps(obj))
    return {"cost": 0.0}


def _claude(cfg, stage, work):
    A = cfg.agents
    cmd = [A.claude_cmd, "-p", prompts.RUN_PROMPT, "--model", A.model, "--max-turns", str(A.max_turns[stage]),
           "--output-format", "json", "--permission-mode", "acceptEdits",   # without it Write is denied in -p mode
           "--allowedTools", *A.allowed_tools]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=int(A.timeout_s),
                       stdin=subprocess.DEVNULL)    # otherwise the CLI waits 3 s for piped input

    info = {"returncode": r.returncode, "seconds": round(time.time() - t0, 1), "stderr": r.stderr[-500:]}
    try:
        j = json.loads(r.stdout)
        info.update({"cost": j.get("total_cost_usd"), "turns": j.get("num_turns"), "is_error": j.get("is_error")})
    except Exception:
        info["stdout_tail"] = r.stdout[-500:]
    return info


def run_one(cfg, paths, stage, blind):
    out = paths.agent_results(stage) / f"{blind}.json"
    if out.exists():
        return blind, "skipped", {}
    work, n_items = _prepare(stage, blind, paths, cfg)
    rng = np.random.default_rng(zlib.crc32(f"{cfg.seed}|{stage}|{blind}".encode()))
    info = {}
    for attempt in range(int(cfg.agents.retries) + 1):
        (work / "result.json").unlink(missing_ok=True)
        try:
            info = _fake(stage, work, paths, blind, rng) if cfg.agents.backend == "fake" else _claude(cfg, stage, work)
        except subprocess.TimeoutExpired:
            info = {"error": "timeout"}
        rp = work / "result.json"
        if rp.exists():
            try:
                obj = json.loads(rp.read_text())
                errs = _validate(stage, obj, n_items)
            except Exception as e:  # noqa: BLE001
                obj, errs = None, [f"invalid JSON: {e}"]
            if not errs:
                write_json(out, obj, indent=1)
                return blind, "ok", info
            info["validation_errors"] = errs
    return blind, "failed", info


def instruction_files_above(path) -> list[Path]:
    """CLAUDE.md-style files that `claude -p` started in `path` would load into the agent's context."""
    found = []
    for d in [Path(path).resolve(), *Path(path).resolve().parents]:
        for name in ("CLAUDE.md", "CLAUDE.local.md", ".claude/CLAUDE.md"):
            if (d / name).is_file():
                found.append(d / name)
    return found


def run_stage(cfg, paths, stage, limit=None, only=None, log=print, dry_run=False):
    if cfg.agents.backend == "claude":
        paths.agent_work(stage).mkdir(parents=True, exist_ok=True)
        leaks = instruction_files_above(paths.agent_work(stage))
        if leaks:
            raise RuntimeError(f"agent workspaces under {paths.agent_work(stage)} would load {[str(p) for p in leaks]} "
                               f"into every agent (breaks blinding). Set COMBO_AGENT_WORK (or paths.agent_work_root) "
                               f"to a folder with no CLAUDE.md above it, e.g. ~/combo_agent_work.")
    q = _query_cfg(cfg)
    relay = None
    if stage == "explain" and q and not dry_run:
        from .query_server import check_service
        from .relay import QueryRelay
        if not check_service(q["url"]):
            raise RuntimeError(f"agents.query is enabled but no query service answers at {q['url']}: start "
                               f"scripts/05b_query_server.py (or the SSH tunnel to it) first")
        paths.agent_work(stage).mkdir(parents=True, exist_ok=True)
        relay = QueryRelay(paths.agent_work(stage), q["url"]).start()
    try:
        _run_selected(cfg, paths, stage, limit, only, log, dry_run)
    finally:
        if relay:
            relay.stop()
            log(f"[agents:{stage}] query relay answered {relay.n_answered} query files")


def _run_selected(cfg, paths, stage, limit, only, log, dry_run):
    sel = read_json(paths.agent_cache / "selection.json")
    if only:
        sel = [b for b in sel if b in set(only)]
    if stage != "explain":
        sel = [b for b in sel if (paths.agent_results("explain") / f"{b}.json").exists()]
    if limit:
        sel = sel[: int(limit)]
    if dry_run:
        for b in sel:
            _prepare(stage, b, paths, cfg)
        log(f"[agents:{stage}] prepared {len(sel)} workspaces under {paths.agent_work(stage)} (dry run)")
        return
    paths.agent_results(stage).mkdir(parents=True, exist_ok=True)
    log_path = paths.agent_results(stage) / "_run_log.jsonl"
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    cost = 0.0
    with ThreadPoolExecutor(max_workers=int(cfg.agents.concurrency)) as ex, open(log_path, "a") as lf:
        futs = [ex.submit(run_one, cfg, paths, stage, b) for b in sel]
        for i, f in enumerate(as_completed(futs)):
            b, status, info = f.result()
            counts[status] += 1
            cost += float(info.get("cost") or 0)
            lf.write(json.dumps({"blind_id": b, "status": status, **info}) + "\n"); lf.flush()
            if (i + 1) % 10 == 0:
                log(f"[agents:{stage}] {i + 1}/{len(sel)} {counts} cost=${cost:.2f}")
    log(f"[agents:{stage}] done {counts} total cost=${cost:.2f}")
