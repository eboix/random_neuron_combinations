"""Tokenized corpus store (document-aligned, right-padded) and the builder that creates it.

Design: one document per row, truncated to `seq_len`, right-padded. Because attention is causal, right
padding does not change activations of real tokens, and documents never attend across each other.
Global token id (gid) = seq_id * seq_len + position.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io_utils import write_json


class CorpusStore:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.seq_len = int(self.manifest["seq_len"])
        self.pad_id = int(self.manifest["pad_id"])
        self.shards = self.manifest["shards"]
        self.n_seqs = int(self.manifest["n_seqs"])
        self._offsets = np.array([s["offset"] for s in self.shards] + [self.n_seqs], dtype=np.int64)
        self._tok, self._len = {}, {}

    def _load(self, i):
        if i not in self._tok:
            s = self.shards[i]
            self._tok[i] = np.load(self.root / s["tokens"], mmap_mode="r")
            self._len[i] = np.load(self.root / s["lengths"])
        return self._tok[i], self._len[i]

    def locate(self, seq_id: int):
        i = int(np.searchsorted(self._offsets, seq_id, side="right") - 1)
        return i, int(seq_id - self._offsets[i])

    def seq_length(self, seq_id: int) -> int:
        i, r = self.locate(seq_id)
        return int(self._load(i)[1][r])

    def get_tokens(self, seq_id: int, lo: int, hi: int) -> np.ndarray:
        """Token ids of positions [lo, hi) of sequence seq_id, clipped to the real length."""
        i, r = self.locate(seq_id)
        toks, lens = self._load(i)
        lo = max(int(lo), 0)
        hi = min(int(hi), int(lens[r]))
        if hi <= lo:
            return np.zeros(0, dtype=np.int64)
        return np.asarray(toks[r, lo:hi]).astype(np.int64)

    def iter_batches(self, max_tokens: int, n_tokens: int | None = None, shard_order=None):
        """Yield dict(input_ids (B,T) int64, lengths (B,), seq_ids (B,)). Deterministic order.
        Within a shard sequences are sorted by length (desc) so batches have little padding."""
        consumed = 0
        order = range(len(self.shards)) if shard_order is None else shard_order
        for i in order:
            toks, lens = self._load(i)
            idx = np.argsort(-lens, kind="stable")
            pos, n = 0, len(idx)
            off = int(self._offsets[i])
            while pos < n:
                T = int(lens[idx[pos]])
                T = min(self.seq_len, max(8, -(-T // 8) * 8))
                B = max(1, max_tokens // T)
                sel = idx[pos:pos + B]
                pos += len(sel)
                ids = np.asarray(toks[sel, :T]).astype(np.int64)
                yield {"input_ids": ids, "lengths": lens[sel].astype(np.int64),
                       "seq_ids": (off + sel).astype(np.int64)}
                consumed += int(lens[sel].sum())
                if n_tokens is not None and consumed >= n_tokens:
                    return


def write_corpus(root, docs, seq_len, pad_id, shard_seqs=4000, meta=None):
    """docs: iterable of 1-D int arrays (already truncated to seq_len). Writes shards + manifest."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    shards, buf, n_seqs, n_tokens = [], [], 0, 0

    def flush():
        nonlocal buf, n_seqs
        if not buf:
            return
        i = len(shards)
        arr = np.full((len(buf), seq_len), pad_id, dtype=np.int32)
        lens = np.zeros(len(buf), dtype=np.int32)
        for r, ids in enumerate(buf):
            arr[r, :len(ids)] = ids
            lens[r] = len(ids)
        np.save(root / f"shard_{i:04d}_tok.npy", arr)
        np.save(root / f"shard_{i:04d}_len.npy", lens)
        shards.append({"tokens": f"shard_{i:04d}_tok.npy", "lengths": f"shard_{i:04d}_len.npy",
                       "n": len(buf), "offset": n_seqs})
        n_seqs += len(buf)
        buf = []

    for ids in docs:
        ids = np.asarray(ids)[:seq_len]
        buf.append(ids)
        n_tokens += len(ids)
        if len(buf) >= shard_seqs:
            flush()
    flush()
    man = {"seq_len": seq_len, "pad_id": int(pad_id), "n_seqs": n_seqs, "n_tokens": int(n_tokens),
           "shards": shards, "meta": meta or {}}
    write_json(root / "manifest.json", man, indent=1)
    return man


# ----------------------------------------------------------------------------- builder (needs `datasets`)
def _tok_batch(texts, tokenizer, seq_len, min_len):
    enc = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=seq_len)["input_ids"]
    for ids in enc:
        if len(ids) >= min_len:
            yield np.asarray(ids, dtype=np.int32)


def _doc_stream(src, split, modulus, tokenizer, seq_len, min_len, cap):
    kind = src.get("kind", "hf")
    if kind == "hf":
        from datasets import load_dataset
        ds = load_dataset(src["hf_id"], src.get("subset"), split=src.get("split", "train"), streaming=True)
        it = (ex[src.get("text_field", "text")] for ex in ds)
    elif kind == "local_jsonl":
        def _it():
            with open(src["path"]) as f:
                for line in f:
                    yield json.loads(line)[src.get("text_field", "text")]
        it = _it()
    else:
        raise ValueError(kind)
    buf = []
    for i, text in enumerate(it):
        if (i % modulus == 0) != (split == "holdout"):
            continue
        if not text:
            continue
        buf.append(text[:cap])
        if len(buf) >= 128:
            yield from _tok_batch(buf, tokenizer, seq_len, min_len)
            buf = []
    if buf:
        yield from _tok_batch(buf, tokenizer, seq_len, min_len)


def build_corpus(cfg, paths, split, tokenizer, log=print):
    """Weighted round-robin mixture of cfg.corpus.sources -> shards under paths.corpus(split)."""
    c = cfg.corpus
    total = int(c.splits[split]["tokens"])
    srcs = list(c.sources)
    w = np.array([s["weight"] for s in srcs], dtype=float)
    w = w / w.sum()
    weights = {s["name"]: float(x) for s, x in zip(srcs, w)}
    iters = {s["name"]: _doc_stream(s, split, c.holdout_modulus, tokenizer, c.seq_len, c.min_len,
                                    c.text_cap_chars) for s in srcs}
    produced = {s["name"]: 0 for s in srcs}
    active = set(produced)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def docs():
        n = 0
        while active and n < total:
            name = min(active, key=lambda k: produced[k] / weights[k])
            try:
                ids = next(iters[name])
            except StopIteration:
                log(f"[corpus:{split}] source {name} exhausted")
                active.discard(name)
                continue
            produced[name] += len(ids)
            n += len(ids)
            if produced[name] >= weights[name] * total:
                active.discard(name)
            yield ids

    man = write_corpus(paths.corpus(split), docs(), c.seq_len, pad_id, c.shard_seqs,
                       meta={"tokens_per_source": produced, "split": split})
    log(f"[corpus:{split}] {man['n_seqs']} docs, {man['n_tokens']} tokens -> {paths.corpus(split)}")
    return man
