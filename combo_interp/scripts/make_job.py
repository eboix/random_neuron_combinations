#!/usr/bin/env python3
"""Create a cluster job folder (sbatch script + request + the exact commands you run yourself).

    python scripts/make_job.py stats_units --name stats_units
    python scripts/make_job.py cache_train --name cache_train
    python scripts/make_job.py cache_holdout --name cache_holdout
    python scripts/make_job.py probe --name probe_r1 --input data/probe/round1_in.jsonl

Nothing here touches the cluster. Review jobs/pending/NNN_name/{run.sbatch,request.md}, then run COMMANDS.txt by hand.
"""
import shutil
import sys
from pathlib import Path

from _common import setup

STAGES = {
    "stats_units": dict(
        cmd="python scripts/01_stats_and_units.py --config {config}",
        need=["data/corpus/train"], out=["data/stats", "data/units", "private/blind_key.json"],
        desc="Pass 1: activation mean/covariance per layer + generate all units (neurons and combinations). ~10-30 min on 1 GPU."),
    "cache_train": dict(
        cmd="python scripts/02_cache.py --config {config} --split train",
        need=["data/corpus/train", "data/units"], out=["data/cache/train"],
        desc="Pass 2 on the train corpus: top-k / windows / moments / uniform sample for every unit. Hours; resumable (re-run the same sbatch after a timeout)."),
    "cache_holdout": dict(
        cmd="python scripts/02_cache.py --config {config} --split holdout",
        need=["data/corpus/holdout", "data/units"], out=["data/cache/holdout"],
        desc="Pass 2 on the held-out corpus (for the simulation score). ~10% of the train cost."),
    "probe": dict(
        cmd="python scripts/07_probe.py --config {config} --job-file {jobdir}/probe_in.jsonl --out {jobdir}/probe_out.jsonl",
        need=["data/units"], out=[],
        desc="Score agent-written texts on the units. Minutes."),
}

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --output={jobdir}/slurm-%j.out
set -euo pipefail
cd {remote_dir}
{setup}
export PYTHONPATH=$PWD/src
export COMBO_DATA_ROOT=$PWD/data
export COMBO_PRIVATE_ROOT=$PWD/private
{command}
"""


def extra(ap):
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--name", required=True)
    ap.add_argument("--input", default=None, help="probe stage: path to the probe input jsonl")


cfg, paths, args = setup(__doc__, extra)
C, S = cfg.cluster, STAGES[args.stage]
jobs = paths.jobs
existing = [p for d in ("pending", "done") for p in (jobs / d).glob("[0-9][0-9][0-9]_*")]
num = max([int(p.name[:3]) for p in existing] + [0]) + 1
jobname = f"{num:03d}_{args.name}"
jdir = jobs / "pending" / jobname
jdir.mkdir(parents=True)
rel_job = f"jobs/pending/{jobname}"

out_files = list(S["out"])
if args.stage == "probe":
    if not args.input:
        sys.exit("--input required for the probe stage")
    shutil.copy(args.input, jdir / "probe_in.jsonl")
    out_files = [f"{rel_job}/probe_out.jsonl"]

command = S["cmd"].format(config=args.config, jobdir=rel_job)
(jdir / "run.sbatch").write_text(TEMPLATE.format(
    name=args.name, partition=C.partition, gres=C.gres, time=C.time, mem=C.mem, cpus=C.cpus, jobdir=rel_job,
    remote_dir=C.remote_dir, setup=C.setup.rstrip(), command=command))
(jdir / "request.md").write_text(
    f"# Job {jobname}  ({args.stage})\n\n{S['desc']}\n\n"
    f"- command: `{command}`\n- needs on cluster: {S['need']}\n- produces: {out_files}\n"
    f"- resources: gres={C.gres} time={C.time} mem={C.mem} cpus={C.cpus}\n"
    f"- model: {cfg.model.name}, layers {list(cfg.model.layers)}\n\n"
    f"Review `run.sbatch` before submitting.\n")

R, H = C.remote_dir, C.host
cmds = ["# --- 0. one-time: code on the cluster (skip if already synced; data/ and private/ are excluded on purpose)",
        f"rsync -av --exclude data --exclude private --exclude jobs/done --exclude '*.pyc' ./ {H}:{R}/",
        "", "# --- 1. inputs this job needs"]
cmds += [f"rsync -av --relative ./{n} {H}:{R}/" for n in S["need"]]
cmds += ["", "# --- 2. submit (you will be asked for 2FA)", f"ssh {H} 'cd {R} && sbatch {rel_job}/run.sbatch'",
         "", "# --- 3. watch", f"ssh {H} 'squeue -u $USER; tail -n 20 {R}/{rel_job}/slurm-*.out'",
         "", "# --- 4. when finished, fetch outputs"]
cmds += [f"rsync -av --relative {H}:{R}/./{n} ./" for n in out_files if n.startswith(("data", "private", "jobs"))]
cmds += [f"rsync -av {H}:{R}/{rel_job}/ {jobs}/done/{jobname}/"]
if args.stage == "cache_train":
    cmds += ["", "# If the job hits the time limit it leaves data/cache/train/ckpt.pt; just re-submit the same sbatch to resume."]
(jdir / "COMMANDS.txt").write_text("\n".join(cmds) + "\n")
print(f"created {jdir}\n  review: {jdir}/request.md, {jdir}/run.sbatch\n  then run the commands in {jdir}/COMMANDS.txt yourself")
