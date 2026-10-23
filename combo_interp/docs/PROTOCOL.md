# Manual cluster handoff protocol

The cluster needs 2FA, so a human is the only bridge. Agents (and you) never run anything on the cluster implicitly.

1. **Request.** `python scripts/make_job.py <stage> --name <name> [--input file]` creates `jobs/pending/NNN_<name>/` with
   `run.sbatch`, `request.md` (what/why/inputs/outputs/resources) and `COMMANDS.txt` (the exact rsync + sbatch + fetch commands).
2. **Review.** Read `request.md` and `run.sbatch`. Check `partition`, `gres`, `time`, the `module load` / venv lines (all from
   `cluster:` in the config, defaults are placeholders).
3. **Run.** Execute the commands in `COMMANDS.txt` yourself. Optional once-per-day convenience: `ControlMaster auto`,
   `ControlPath ~/.ssh/cm-%r@%h:%p`, `ControlPersist 8h` in `~/.ssh/config` so later ssh/rsync reuse your authenticated
   session (check your cluster's acceptable-use policy; never give anyone your password or 2FA codes).
4. **Fetch.** The last step of `COMMANDS.txt` rsyncs the job folder to `jobs/done/NNN_<name>/` and the stage outputs into `data/`.
5. **Continue.** Tell the agent (or yourself) to resume; results appear as files, nothing is assumed finished until they exist.

Job order: `stats_units` → `cache_holdout`, `cache_train`, `calibrate` → (local export, pilot; explain, write, simulate agents) → `probe`.
`cache_train` writes `data/cache/train/ckpt.pt`; if it hits the walltime, resubmit the *same* sbatch to resume (a fingerprint check
refuses to resume under a changed config). `private/blind_key.json` is created by `stats_units` and fetched back with it.
