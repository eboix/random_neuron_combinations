"""Run the three agent stages (explain -> [GPU probe] -> blindmax / simulate) over the selected units.

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
from .validate import validate_blindmax, validate_explain, validate_simulate

_MARK = [(re.compile(r"\[\[(.*?)\|[-0-9.nae]+\]\]"), r"\1"), (re.compile(r"«(.*?)\|[-0-9.nae]+»"), r"\1"),
         (re.compile(r"\[\[(.*?)\]\]"), r"\1")]


def strip_marks(t: str) -> str:
    for rx, rep in _MARK:
        t = rx.sub(rep, t)
    return t.replace("↵", "\n")


# ------------------------------------------------------------------ workspaces
def _prepare(stage, blind, paths):
    work = paths.agent_work(stage) / blind
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    if stage == "explain":
        shutil.copy(paths.agent_cache / f"{blind}.json", work / "data.json")
        shutil.copy(paths.agent_cache / f"{blind}.md", work / "view.md")
        shutil.copy(Path(__file__).parent / "unit_tools.py", work / "unit_tools.py")
        (work / "TASK.md").write_text(prompts.EXPLAIN)
        return work, None
    ex = read_json(paths.agent_results("explain") / f"{blind}.json")
    if stage == "blindmax":
        (work / "TASK.md").write_text(prompts.BLINDMAX.format(explanation=ex["explanation"], fires_on=ex["fires_on"]))
        return work, None
    items = read_json(paths.agent_cache / "holdout" / f"{blind}.json")
    write_json(work / "items.json", items, indent=1)
    (work / "TASK.md").write_text(prompts.SIMULATE.format(n=len(items), explanation=ex["explanation"], fires_on=ex["fires_on"]))
    return work, len(items)


def _validate(stage, obj, n_items):
    if stage == "explain":
        return validate_explain(obj)
    if stage == "blindmax":
        return validate_blindmax(obj)
    return validate_simulate(obj, n_items)


# ------------------------------------------------------------------ backends
def _fake(stage, work, paths, blind, rng):
    """Plumbing stand-in: 'explains' a unit from its own examples (so scores are not pure noise)."""
    if stage == "explain":
        d = read_json(work / "data.json")
        top = [strip_marks(e["text"]) for e in d["examples"]["primary"]][:5]
        rnd = [strip_marks(e["text"]) for e in d["random"]][:5]
        toks = [t for t, _ in d["token_stats"]["primary"]["fires_on"][:3]]
        pad = lambda lst, fill: (lst + [fill] * 5)[:5]
        obj = {"explanation": f"Fires on tokens such as {toks}.", "fires_on": "on the listed tokens",
               "parts": [repr(t) for t in toks[:2]] or ["unclear"],     # repr: whitespace tokens stay non-empty
               "other_pole": "", "confidence": 3,
               "predicted_fire": pad(top, "alpha beta"), "near_miss": pad(rnd, "gamma delta"),
               "distant": ["The recipe calls for two cups of flour.", "Quantum chromodynamics is hard.",
                           "He sailed across the bay at dawn.", "Invoice number 4471 is overdue.",
                           "The choir rehearsed on Thursday."], "notes": "fake backend"}
    elif stage == "blindmax":
        d = read_json(paths.agent_cache / f"{blind}.json")
        top = [strip_marks(e["text"]) for e in d["examples"]["primary"]][:3]
        obj = {"texts": (top + ["alpha beta gamma"] * 3)[:3]}
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
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=int(A.timeout_s))
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
    work, n_items = _prepare(stage, blind, paths)
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
    sel = read_json(paths.agent_cache / "selection.json")
    if only:
        sel = [b for b in sel if b in set(only)]
    if stage != "explain":
        sel = [b for b in sel if (paths.agent_results("explain") / f"{b}.json").exists()]
    if limit:
        sel = sel[: int(limit)]
    if dry_run:
        for b in sel:
            _prepare(stage, b, paths)
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
