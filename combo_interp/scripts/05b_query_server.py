#!/usr/bin/env python3
"""Serve model queries for explainer agents (needed while `05_run_agents.py --stage explain` runs with
agents.query.enabled). Runs where private/ is; on a cluster, reach it from the laptop through an SSH tunnel."""
from _common import setup


def extra(ap):
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 on a cluster node behind an SSH tunnel")
    ap.add_argument("--port", type=int, default=8765)


cfg, paths, args = setup(__doc__, extra)
import socket  # noqa: E402

from combo_interp.agents.query_server import UnitQueryEngine, make_server  # noqa: E402

engine = UnitQueryEngine(cfg, paths)
srv = make_server(engine, args.host, args.port)
print(f"[query] serving {len(engine.ref)} units on {socket.gethostname()}:{args.port} "
      f"(budget {engine.budget} texts per unit, device {engine.device})", flush=True)
srv.serve_forever()
