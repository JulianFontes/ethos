"""Judges that label each comment version allow/remove, and score it where they can (see scripts/judge.py).

Each judge is an entry in configs/judges.yaml; its `type` picks the implementation.
"""
from ethos.judges import annotator_judge, detoxify_judge, human_judge, llm_judge, pluralistic_judge

RUNNERS = {"human": human_judge.run, "detoxify": detoxify_judge.run,
           "ollama": llm_judge.run, "ollama_dmp": pluralistic_judge.run,
           "openai": llm_judge.run, "openai_dmp": pluralistic_judge.run,
           "ollama_annotator": annotator_judge.run, "openai_annotator": annotator_judge.run,
           "ollama_annotator_dmp": annotator_judge.run_dmp, "openai_annotator_dmp": annotator_judge.run_dmp,
           "ollama_minimal": annotator_judge.run_minimal, "openai_minimal": annotator_judge.run_minimal}


def run_judge(cfg, sample_dir, limit=None):
    if cfg["type"] not in RUNNERS:
        raise ValueError(f"unknown judge type {cfg['type']!r}; expected one of {sorted(RUNNERS)}")
    RUNNERS[cfg["type"]](cfg, sample_dir, limit)
