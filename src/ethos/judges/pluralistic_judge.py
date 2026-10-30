"""Pluralistic LLM judge (config type "ollama_dmp"), after Dynamic Moral Profiling.

Same loop as the base judge, but each comment gets a value profile (ethos.llm.dmp) that is
filled into the prompt's {profile} slot and recorded in the `profile` column. All versions
of a comment get the same profile.
"""
from ethos.judges.core import ROOT
from ethos.judges.llm_judge import judge_texts
from ethos.llm.dmp import format_profile, load_values, sample_profile


def run(cfg, sample_dir, limit=None):
    template = (ROOT / cfg["prompt"]).read_text().strip()
    values = load_values(cfg["values"])

    def prompt_for(row):
        profile = format_profile(sample_profile(row.id, values, cfg.get("alpha", 10),
                                                cfg.get("top_k", 3), cfg.get("seed", 0)))
        return template.replace("{profile}", profile), profile

    judge_texts(cfg, sample_dir, limit, prompt_for)
