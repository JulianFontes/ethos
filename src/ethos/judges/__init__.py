"""Judges that label each comment version allow/remove, and score it where they can (see scripts/judge.py).

Each judge is an entry in configs/judges/judges.yaml; its `type` picks the implementation.
"""
from ethos.judges import detoxify_judge, human_judge, llm_judge

RUNNERS = {"human": human_judge.run, "detoxify": detoxify_judge.run,
           "ollama_minimal": llm_judge.run, "openai_minimal": llm_judge.run,
           "ollama_minimal_dmp": llm_judge.run_dmp, "openai_minimal_dmp": llm_judge.run_dmp}


def run_judge(cfg, sample_dir, limit=None):
    if cfg["type"] not in RUNNERS:
        raise ValueError(f"unknown judge type {cfg['type']!r}; expected one of {sorted(RUNNERS)}")
    RUNNERS[cfg["type"]](cfg, sample_dir, limit)
