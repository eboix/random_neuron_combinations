"""Model query service for explainer agents (agents.query.enabled).

POST /query {"blind_id": "u00123", "texts": [...]} -> that unit's activation on every token of each text, as a
fraction of the unit's corpus maximum along its primary pole (the scale of view.md), never raw values. A budget of
`agents.query.budget_texts` texts per unit is enforced here and survives restarts: every query is appended to
data/query_log/<blind_id>.jsonl, which scoring also uses to drop explainer texts that were queried before submission.

Needs the private unit reference, so it runs where private/ lives: on the laptop for local studies, or as a GPU job on
the cluster reached through an SSH tunnel (slurm/submit.sh serve).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..io_utils import read_json


class UnitQueryEngine:
    def __init__(self, cfg, paths, model=None, tok=None, device=None):
        import torch
        from ..model_utils import load_model
        from ..units import load_unit_set
        if model is None:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            model, tok = load_model(cfg, device)
        self.model, self.tok = model, tok
        self.device = next(model.parameters()).device
        self.ref = read_json(paths.unit_ref)
        self.us = {int(L): load_unit_set(paths, int(L)) for L in cfg.model.layers}
        self.col = {m["unit_id"]: m["col"] for us in self.us.values() for m in us["meta"]}
        Q = cfg.agents.query
        self.budget, self.per_call = int(Q.budget_texts), int(Q.max_texts_per_call)
        self.max_tokens = int(Q.max_tokens_per_text)
        self.skip = int(cfg.cache.skip_first_tokens)
        self.early_stop = bool(cfg.get("runtime", {}).get("early_stop", True))
        self.log_dir = paths.query_log
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def used(self, blind: str) -> int:
        p = self.log_dir / f"{blind}.jsonl"
        return sum(1 for _ in open(p)) if p.exists() else 0

    def activations(self, layer: int, col: int, text: str):
        import torch
        from ..hooks import MLPCapture
        ids = self.tok(text, add_special_tokens=False)["input_ids"][: self.max_tokens]
        if not ids:
            return [], []
        a = torch.as_tensor(self.us[layer]["A"][:, col], device=self.device, dtype=torch.float32)
        b = float(self.us[layer]["b"][col])
        with MLPCapture(self.model, [layer], stop_after_last=self.early_stop) as cap:
            cap.run(torch.tensor([ids], device=self.device))
            z = (cap.p[layer][0].float() @ a + b).cpu().numpy()
        return ids, z

    def query(self, blind, texts) -> dict:
        if blind not in self.ref:
            raise KeyError(f"unknown unit {blind!r}")
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("texts must be a non-empty list of non-empty strings")
        if len(texts) > self.per_call:
            raise ValueError(f"at most {self.per_call} texts per call")
        R = self.ref[blind]
        L, col, s, ref = int(R["layer"]), self.col[R["unit_id"]], float(R["primary"]), float(R["ref"])
        with self.lock:
            left = self.budget - self.used(blind)
            if left <= 0:
                return {"error": "query budget for this unit is used up", "results": [], "remaining": 0}
            results = []
            with open(self.log_dir / f"{blind}.jsonl", "a") as log:
                for t in texts[:left]:
                    ids, z = self.activations(L, col, t)
                    fr = [None if i < self.skip else round(float(s * x / ref), 3) for i, x in enumerate(z)]
                    scored = [x for x in fr if x is not None]
                    mx = max(scored) if scored else None
                    n_all = len(self.tok(t, add_special_tokens=False)["input_ids"])
                    results.append({"tokens": [self.tok.decode([i]) for i in ids], "fracs": fr, "max_frac": mx,
                                    "truncated_from": n_all if n_all > len(ids) else None})
                    log.write(json.dumps({"text": t, "max_frac": mx}) + "\n")
            out = {"results": results, "remaining": self.budget - self.used(blind)}
            if len(texts) > left:
                out["error"] = f"budget reached: only the first {left} text(s) were run"
            return out


def make_server(engine: UnitQueryEngine, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "budget_texts": engine.budget})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/query":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, engine.query(req.get("blind_id"), req.get("texts")))
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def check_service(url: str, timeout: float = 5.0) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return bool(json.load(r).get("ok"))
    except OSError:
        return False
