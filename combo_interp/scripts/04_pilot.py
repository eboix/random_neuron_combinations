#!/usr/bin/env python3
"""CPU: agent-free pilot. Activation-shape and concentration metrics per condition (no LLM calls)."""
from _common import setup


def extra(ap):
    ap.add_argument("--embed", action="store_true", help="also compute embedding coherence (needs sentence-transformers)")


cfg, paths, args = setup(__doc__, extra)
from combo_interp.pilot import pilot_metrics, plot_pilot, summarize_pilot  # noqa: E402

tok = None
if args.embed:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.name)
df = pilot_metrics(cfg, paths, tok=tok, embed=args.embed)
out = paths.results
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "pilot_units.csv", index=False)
s = summarize_pilot(df)
s.to_csv(out / "pilot_summary.csv", index=False)
plot_pilot(df, out)
print(s.pivot_table(index="cond", columns="metric", values="median", sort=False).round(3).to_string())
