"""Raw caches -> (a) blinded per-unit agent views, (b) private ground truth / reference scales.

Agents see: layer, primary pole, examples (with per-token activation tags as FRACTIONS of the unit's corpus max,
never raw values), tiered examples, random baselines, token statistics. They never see the condition, k, the
neurons/weights, kurtosis or raw scales (those could reveal the unit type). Ground truth lives in paths.private.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .corpus import CorpusStore
from .io_utils import read_json, write_json
from .units import load_unit_set, make_blind_key


# ------------------------------------------------------------------ pure helpers (unit tested)
def dedupe_hits(vals, gids, seq_len, max_per_seq=1, min_gap=48, limit=None):
    """vals sorted descending. Returns [(rank, val, gid)] keeping at most `max_per_seq` hits per document,
    at least `min_gap` tokens apart. Skips empty slots (gid < 0 or non-finite value)."""
    seen, out = {}, []
    for rank, (v, g) in enumerate(zip(vals, gids)):
        if g < 0 or not np.isfinite(v):
            continue
        s, pos = divmod(int(g), int(seq_len))
        lst = seen.setdefault(s, [])
        if len(lst) >= max_per_seq or any(abs(pos - q) < min_gap for q in lst):
            continue
        lst.append(pos)
        out.append((rank, float(v), int(g)))
        if limit is not None and len(out) >= limit:
            break
    return out


def pick_spread(items, n):
    if len(items) <= n:
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


def render_marked(toks, fracs, center=None, thresh=0.1):
    """Inline activation tags. center token -> [[tok|f]]; other tokens with frac >= thresh -> «tok|f»."""
    parts = []
    for i, t in enumerate(toks):
        t = t.replace("\n", "↵")
        f = None if fracs is None else fracs[i]
        ok = f is not None and np.isfinite(f)
        if i == center:
            parts.append(f"[[{t}|{f:.2f}]]" if ok else f"[[{t}]]")
        elif ok and f >= thresh:
            parts.append(f"«{t}|{f:.2f}»")
        else:
            parts.append(t)
    return "".join(parts)


class Decoder:
    def __init__(self, tok):
        self.tok, self._c = tok, {}

    def many(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i not in self._c:
                self._c[i] = self.tok.decode([i])
            out.append(self._c[i])
        return out


def tier_of(frac, edges):
    for name, e in zip(("strong", "moderate", "weak"), edges):
        if frac >= e:
            return name
    return None


# ------------------------------------------------------------------ rendering of one example
def _render_window(store, dec, seq, pos, before, after, fracs_fn, thresh, center_frac=None):
    lo = max(0, pos - before)
    hi = min(store.seq_length(seq), pos + after + 1)
    ids = store.get_tokens(seq, lo, hi)
    toks = dec.many(ids)
    center = pos - lo
    fr = fracs_fn(lo, len(ids)) if fracs_fn else None
    if fr is None and center_frac is not None:
        fr = [None] * len(ids)
        fr[center] = center_frac
    return render_marked(toks, fr, center, thresh)


def moments_stats(mom, col):
    n = float(mom["n"])
    m1 = mom["s1"][col] / n
    m2 = mom["s2"][col] / n - m1 ** 2
    m4 = mom["s4"][col] / n - 4 * m1 * mom["s3"][col] / n + 6 * m1 ** 2 * mom["s2"][col] / n - 3 * m1 ** 4
    kurt = float(m4 / (m2 ** 2) - 3) if m2 > 0 else float("nan")
    return {"n": n, "mean": float(m1), "var": float(m2), "kurtosis": kurt,
            "frac_gt3": float((mom["gt3"][col] + mom["lt3"][col]) / n),
            "frac_gt5": float((mom["gt5"][col] + mom["lt5"][col]) / n)}


def build_unit_data(col, blind, layer, s, ref, ref_other, big, prof, uacts, ugid, store, dec, E, rng):
    """Everything an agent may see about one unit. s = +1 / -1 primary pole sign."""
    pole = "pos" if s > 0 else "neg"
    other = "neg" if s > 0 else "pos"
    sp = {"pos": 1.0, "neg": -1.0}
    Lq = store.seq_len
    thresh = float(E["mark_threshold"])
    offs = prof["offsets"]
    before, after = int(-offs[0]), int(offs[-1])

    shown = set()          # contexts already displayed as top examples (excluded from the tiers)

    def examples(p, R, n, n_long):
        vals, gid = prof[f"{p}_vals"][col], prof[f"{p}_gid"][col]
        hits = dedupe_hits(sp[p] * vals, gid, Lq, E["dedupe_max_per_seq"], limit=n)
        out = []
        for j, (rank, v, g) in enumerate(hits):
            seq, pos = divmod(g, Lq)
            shown.add(g)
            win = prof[f"{p}_win"][col][rank]

            def fr(lo, n_ids, win=win, pos=pos, p=p, R=R):
                st = lo - (pos - before)
                return list((sp[p] * win[st:st + n_ids]) / R)
            text = _render_window(store, dec, seq, pos, before, after, fr, thresh)
            long = None
            if j < n_long:
                long = _render_window(store, dec, seq, pos, int(E["long_before"]), int(E["long_after"]), None, thresh,
                                      center_frac=sp[p] * v / R)
            out.append({"i": j, "frac": round(sp[p] * v / R, 3), "text": text, "long": long})
        return out

    data = {"blind_id": blind, "layer": int(layer), "primary_pole": pole}
    data["examples"] = {"primary": examples(pole, ref, int(E["k_examples"]), int(E["long_n"])),
                        "other": examples(other, ref_other, int(E["other_pole_n"]), 0) if ref_other > 0 else []}

    # tiers (from the big list; weak tier topped up with the uniform sample)
    vals, gid = big[f"{pole}_vals"][col], big[f"{pole}_gid"][col]
    sv = sp[pole] * vals
    hits = dedupe_hits(sv, gid, Lq, E["dedupe_max_per_seq"])
    edges = E["tier_edges"]
    cand = {"strong": [], "moderate": [], "weak": []}
    for rank, v, g in hits:
        t = tier_of(v / ref, edges)
        if t and g not in shown:
            cand[t].append((v / ref, g))
    usv = sp[pole] * uacts[:, col].astype(np.float64) / ref
    for r in np.where((usv >= edges[2]) & (usv < edges[1]))[0]:
        if int(ugid[r]) not in shown:
            cand["weak"].append((float(usv[r]), int(ugid[r])))
    tiers = {}
    for t, lst in cand.items():
        lst = sorted(lst, key=lambda x: -x[0])
        tiers[t] = []
        for j, (f, g) in enumerate(pick_spread(lst, int(E["tier_n"]))):
            seq, pos = divmod(g, Lq)
            text = _render_window(store, dec, seq, pos, int(E["context_before_tiers"]), int(E["context_after_tiers"]),
                                  None, thresh, center_frac=f)
            tiers[t].append({"i": j, "frac": round(f, 3), "text": text, "long": None})
    data["tiers"] = tiers

    rows = rng.choice(len(ugid), size=min(int(E["random_n"]), len(ugid)), replace=False)
    data["random"] = []
    for j, r in enumerate(rows):
        seq, pos = divmod(int(ugid[r]), Lq)
        text = _render_window(store, dec, seq, pos, int(E["context_before_tiers"]), int(E["context_after_tiers"]),
                              None, thresh, center_frac=float(usv[r]))
        data["random"].append({"i": j, "frac": round(float(usv[r]), 3), "text": text, "long": None})

    # token statistics over the 100 strongest (deduped) firings of each pole
    ts = {}
    for key, p, R in (("primary", pole, ref), ("other", other, ref_other)):
        if R <= 0:
            ts[key] = {"fires_on": [], "next_token": []}
            continue
        h = dedupe_hits(sp[p] * big[f"{p}_vals"][col], big[f"{p}_gid"][col], Lq, E["dedupe_max_per_seq"], limit=100)
        a, b = Counter(), Counter()
        for _, _, g in h:
            seq, pos = divmod(g, Lq)
            t0 = store.get_tokens(seq, pos, pos + 1)
            t1 = store.get_tokens(seq, pos + 1, pos + 2)
            if len(t0):
                a[dec.many(t0)[0].replace("\n", "↵")] += 1
            if len(t1):
                b[dec.many(t1)[0].replace("\n", "↵")] += 1
        ts[key] = {"fires_on": a.most_common(12), "next_token": b.most_common(8)}
    data["token_stats"] = ts
    raw100 = [int(g) // Lq for g in gid[:100] if g >= 0]
    data["n_distinct_docs_top100"] = len(set(raw100))
    data["rarity"] = {"frac_tokens_ge_10pct_of_max": float((usv >= 0.1).mean()),
                      "frac_tokens_ge_30pct_of_max": float((usv >= 0.3).mean())}
    return data


def render_view_md(d):
    lines = [f"# Unit {d['blind_id']}  (layer {d['layer']})",
             f"Primary pole: **{d['primary_pole']}** (the direction with the strongest corpus firings; explain THIS pole).", "",
             "Marker legend: `[[tok|f]]` = the token this example was selected for; `«tok|f»` = other tokens in the window "
             "that also fire (f = fraction of the unit's corpus maximum on this pole, so 1.00 = strongest ever seen). "
             "`↵` = newline. Windows are short; use `python unit_tools.py show --long` for longer context.", ""]
    if d["n_distinct_docs_top100"] is not None:
        lines.append(f"Among the 100 strongest firings, {d['n_distinct_docs_top100']} distinct documents. "
                     f"Fraction of random tokens firing >=10% of max: {d['rarity']['frac_tokens_ge_10pct_of_max']:.4f}; "
                     f">=30%: {d['rarity']['frac_tokens_ge_30pct_of_max']:.4f}.")
    lines += ["", "## Strongest firings, primary pole (one per document)"]
    for e in d["examples"]["primary"]:
        lines.append(f"{e['i']:>2}. [{e['frac']:.2f}] {e['text']}")
    lines += ["", "## Token statistics, primary pole", f"fires on: {d['token_stats']['primary']['fires_on']}",
              f"next token: {d['token_stats']['primary']['next_token']}"]
    for t in ("strong", "moderate", "weak"):
        if d["tiers"][t]:
            lines += ["", f"## {t.capitalize()} firings (primary pole)"]
            for e in d["tiers"][t]:
                lines.append(f"{e['i']:>2}. [{e['frac']:.2f}] {e['text']}")
    lines += ["", "## Random tokens (baseline: what ordinary text looks like for this unit)"]
    for e in d["random"]:
        lines.append(f"{e['i']:>2}. [{e['frac']:.2f}] {e['text']}")
    if d["examples"]["other"]:
        lines += ["", "## Strongest firings of the OTHER pole (fractions relative to that pole's own max)"]
        for e in d["examples"]["other"]:
            lines.append(f"{e['i']:>2}. [{e['frac']:.2f}] {e['text']}")
        lines += ["", f"other pole fires on: {d['token_stats']['other']['fires_on']}",
                  f"other pole next token: {d['token_stats']['other']['next_token']}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ driver
def build_agent_cache(cfg, paths, tok, log=print):
    E = cfg.export
    layers = [int(l) for l in cfg.model.layers]
    tr_store, ho_store = CorpusStore(paths.corpus("train")), CorpusStore(paths.corpus("holdout"))
    dec = Decoder(tok)
    if paths.blind_key.exists():
        key = read_json(paths.blind_key)
    else:
        ids = [m["unit_id"] for L in layers for m in load_unit_set(paths, L)["meta"]]
        key = make_blind_key(ids, cfg.seed)
        write_json(paths.blind_key, key, indent=0)
    inv = {u: b for b, u in key.items()}
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 4242]))
    unit_ref, truth, index, by_cell = {}, {}, [], {}
    out_dir = paths.agent_cache
    (out_dir / "holdout").mkdir(parents=True, exist_ok=True)

    for L in layers:
        us = load_unit_set(paths, L)
        d_tr = paths.cache("train", L)
        big, prof = np.load(d_tr / "topk_big.npz"), np.load(d_tr / "topk_prof.npz")
        uni, mom = np.load(d_tr / "uniform.npz"), np.load(d_tr / "moments.npz")
        d_ho = paths.cache("holdout", L)
        hbig, huni = np.load(d_ho / "topk_big.npz"), np.load(d_ho / "uniform.npz")
        big = {k: big[k] for k in big.files}
        prof = {k: prof[k] for k in prof.files}
        hbig = {k: hbig[k] for k in hbig.files}
        mom = {k: mom[k] for k in mom.files}
        uacts, ugid = uni["acts"], uni["gid"]
        hacts, hgid = huni["acts"], huni["gid"]
        mean_u, std_u = us["mean_u"], np.maximum(us["std_u"], 1e-12)
        zpos = (big["pos_vals"][:, 0] - mean_u) / std_u
        zneg = (mean_u - big["neg_vals"][:, 0]) / std_u
        for col, m in enumerate(us["meta"]):
            uid, blind = m["unit_id"], inv[m["unit_id"]]
            s = 1 if zpos[col] >= zneg[col] else -1
            ref = float(big["pos_vals"][col, 0]) if s > 0 else float(-big["neg_vals"][col, 0])
            ref_o = float(-big["neg_vals"][col, 0]) if s > 0 else float(big["pos_vals"][col, 0])
            if not np.isfinite(ref) or ref <= 0:
                log(f"[export] skipping dead unit {uid}")
                continue
            data = build_unit_data(col, blind, L, s, ref, ref_o if np.isfinite(ref_o) else 0.0, big, prof,
                                   uacts, ugid, tr_store, dec, E, rng)
            write_json(out_dir / f"{blind}.json", data)
            (out_dir / f"{blind}.md").write_text(render_view_md(data))
            # holdout items for the simulation score
            pole = "pos" if s > 0 else "neg"
            hv, hg = hbig[f"{pole}_vals"][col], hbig[f"{pole}_gid"][col]
            hits = pick_spread(dedupe_hits(s * hv, hg, ho_store.seq_len, E["dedupe_max_per_seq"], limit=60),
                               int(E["holdout_top_items"]))
            items = []
            for _, v, g in hits:
                seq, pos = divmod(g, ho_store.seq_len)
                items.append((_render_window(ho_store, dec, seq, pos, int(E["holdout_before"]), int(E["holdout_after"]), None, 0.1), s * v / ref))
            if len(hgid):
                rr = rng.choice(len(hgid), size=min(int(E["holdout_random_items"]), len(hgid)), replace=False)
                for r in rr:
                    seq, pos = divmod(int(hgid[r]), ho_store.seq_len)
                    items.append((_render_window(ho_store, dec, seq, pos, int(E["holdout_before"]), int(E["holdout_after"]), None, 0.1),
                                  s * float(hacts[r, col]) / ref))
            order = rng.permutation(len(items))
            items = [items[i] for i in order]
            write_json(out_dir / "holdout" / f"{blind}.json", [{"item": i, "text": t} for i, (t, _) in enumerate(items)])
            truth[blind] = [r for _, r in items]
            ms = moments_stats(mom, col)
            unit_ref[blind] = {"unit_id": uid, "layer": L, "cond": m["cond"], "kind": m["kind"], "k": m["k"],
                               "primary": s, "ref": ref, "ref_other": ref_o, "std": float(std_u[col]), **{
                                   k: ms[k] for k in ("kurtosis", "frac_gt3", "frac_gt5")}}
            index.append({"blind_id": blind, "layer": L})
            by_cell.setdefault((L, m["cond"]), []).append(blind)
        log(f"[export] layer {L} done ({len(index)} units so far)")

    n_sel = int(cfg.agents.units_per_condition_per_layer)
    selection = []
    for cell in sorted(by_cell):
        ids = by_cell[cell]
        pick = rng.choice(len(ids), size=min(n_sel, len(ids)), replace=False)
        selection += [ids[i] for i in pick]
    selection = [selection[i] for i in rng.permutation(len(selection))]
    write_json(out_dir / "index.json", index)
    write_json(out_dir / "selection.json", selection)
    write_json(paths.unit_ref, unit_ref)
    write_json(paths.holdout_truth, truth)
    log(f"[export] {len(index)} units exported, {len(selection)} selected for agents -> {out_dir}")
