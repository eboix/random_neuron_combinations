"""YAML config loading (with `base:` inheritance, dotted CLI overrides) and path helpers."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def _wrap(x):
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _load_raw(path) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    if base:
        return _deep_update(_load_raw(path.parent / base), raw)
    return raw


def load_config(path, overrides=None) -> Cfg:
    """overrides: list like ["model.name=Qwen/Qwen3-4B", "units.per_condition=50"]."""
    raw = _load_raw(path)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        cur = raw
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(val)
    return _wrap(raw)


class Paths:
    """All on-disk locations, derived from cfg.paths (env vars COMBO_DATA_ROOT / COMBO_PRIVATE_ROOT win)."""

    def __init__(self, cfg: Cfg):
        self.data = Path(os.environ.get("COMBO_DATA_ROOT", cfg.paths.data_root))
        self.private = Path(os.environ.get("COMBO_PRIVATE_ROOT", cfg.paths.private_root))
        self.jobs = Path(cfg.paths.get("jobs_dir", "jobs"))

    def corpus(self, split: str) -> Path:
        return self.data / "corpus" / split

    def stats(self, layer: int) -> Path:
        return self.data / "stats" / f"L{layer}"

    def units(self, layer: int) -> Path:
        return self.data / "units" / f"L{layer}"

    def cache(self, split: str, layer: int) -> Path:
        return self.data / "cache" / split / f"L{layer}"

    @property
    def agent_cache(self) -> Path:
        return self.data / "agent_cache"

    def agent_work(self, stage: str) -> Path:
        return self.data / "agent_work" / stage

    def agent_results(self, stage: str) -> Path:
        return self.data / "agent_results" / stage

    @property
    def scores(self) -> Path:
        return self.data / "scores"

    @property
    def results(self) -> Path:
        return self.data / "results"

    # private (never shown to agents)
    @property
    def blind_key(self) -> Path:
        return self.private / "blind_key.json"

    @property
    def unit_ref(self) -> Path:
        return self.private / "unit_ref.json"

    @property
    def holdout_truth(self) -> Path:
        return self.private / "holdout_truth.json"
