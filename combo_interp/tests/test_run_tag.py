"""agents.run_tag gives a separate agent run its own outputs; data shared by all runs stays where it is."""
from pathlib import Path

from combo_interp.config import Paths, load_config

CFG = Path(__file__).resolve().parents[1] / "configs"


def test_noquery_run_has_own_agent_outputs():
    plain = Paths(load_config(CFG / "cluster_qwen3_8b.yaml"))
    noq_cfg = load_config(CFG / "cluster_qwen3_8b_noquery.yaml")
    noq = Paths(noq_cfg)
    assert not noq_cfg.agents.query.enabled and plain.run_tag == "" and noq.run_tag == "noquery"
    # plain layout unchanged
    assert plain.agent_results("explain") == plain.data / "agent_results" / "explain"
    assert plain.probe == plain.data / "probe" and plain.scores == plain.data / "scores"
    assert plain.agent_analysis == plain.results and plain.query_log == plain.data / "query_log"
    # tagged run: every agent-run output differs, shared data does not
    for f in (lambda p: p.agent_results("explain"), lambda p: p.agent_work("write"), lambda p: p.probe,
              lambda p: p.scores, lambda p: p.agent_analysis, lambda p: p.query_log):
        assert f(noq) != f(plain) and "noquery" in f(noq).parts
    assert noq.agent_cache == plain.agent_cache and noq.cache("holdout", 9) == plain.cache("holdout", 9)
    assert noq.blind_key == plain.blind_key and noq.results == plain.results
