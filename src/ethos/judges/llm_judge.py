"""LLM judge: asks an LLM whether each comment is TOXIC or NOT TOXIC, `samples` times, like a panel of annotators.

Prompt: prompts/judge_minimal.txt (short, no annotator instructions). Each call returns {"evaluation", "rationale"}.
For each comment:

    score    = share of the `samples` ratings that are TOXIC   (comparable to the human `toxicity` score)
    decision = "remove" if score >= threshold (default 0.5)
    ratings  = JSON list of every rating, for later analysis

Two variants (config `type`; ollama_* runs locally through Ollama, openai_* through the OpenAI API):
  ollama_minimal / openai_minimal          the base judge, no profile
  ollama_minimal_dmp / openai_minimal_dmp  Dynamic Moral Profiling: each rating gets its own random value profile
                                           (ethos.llm.dmp), or a fixed `profile` from the config, filled into
                                           prompts/judge_minimal_pluralistic.txt's {profile}

A comment is written only when all its ratings succeed; otherwise it is retried on the next run.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from tqdm import tqdm

from ethos.judges.core import ROOT, append_rows, output_path, todo
from ethos.llm import ollama_client, openai_client
from ethos.llm.dmp import format_profile, load_values, sample_profile
from ethos.llm.ollama_client import DEFAULT_BASE_URL
from ethos.scores import toxicity_score

SCHEMA = {
    "type": "object",
    "properties": {"evaluation": {"type": "string", "enum": ["TOXIC", "NOT TOXIC"]},
                   "rationale": {"type": "string"}},
    "required": ["evaluation", "rationale"],
    "additionalProperties": False,
}


def rate_texts(cfg, sample_dir, limit, prompt_for):
    """Rate every remaining comment `samples` times. `prompt_for(row, i)` returns (system_prompt, profile) for
    rating i; profile is "" when there is none."""
    name, n = cfg["name"], cfg.get("samples", 10)
    threshold, temperature = cfg.get("threshold", 0.5), cfg.get("temperature", 1.0)
    llm = openai_client if cfg["type"].startswith("openai") else ollama_client
    client = llm.get_client(cfg["model"], cfg.get("base_url", DEFAULT_BASE_URL))  # fails fast if unavailable
    texts = todo(sample_dir, name, limit, cfg.get("query"))
    if texts.empty:
        print(f"{name}: nothing to do")
        return

    def rate(row):
        ratings, profiles = [], []
        for i in range(n):
            system, profile = prompt_for(row, i)
            profiles.append(profile)
            try:
                ratings.append(llm.chat_json(client, cfg["model"], system, f"Comment:\n{row.text}", SCHEMA,
                                             temperature=temperature, max_tokens=cfg.get("max_tokens"),
                                             reasoning_effort=cfg.get("reasoning_effort")))
            except Exception as e:  # connection error, bad JSON, refusal, rate limit
                return row, "", e
        return row, json.dumps(profiles) if any(profiles) else "", ratings

    # `workers` comments in flight at once (default 1); rows are written from this thread only, as they finish.
    out, failed = output_path(sample_dir), 0
    with ThreadPoolExecutor(max_workers=cfg.get("workers", 1)) as pool:
        futures = [pool.submit(rate, row) for row in texts.itertuples(index=False)]
        for future in tqdm(as_completed(futures), total=len(futures), desc=name, unit="text"):
            row, profile, ratings = future.result()
            if isinstance(ratings, Exception):
                failed += 1
                tqdm.write(f"{name}: failed {row.id}/{row.version}: {str(ratings)[:120]}")
                continue
            score = toxicity_score(r["evaluation"] for r in ratings)
            append_rows(out, [{"id": row.id, "version": row.version, "text": row.text, "judge": name,
                               "score": round(score, 4), "decision": "remove" if score >= threshold else "allow",
                               "rationale": "", "profile": profile, "ratings": json.dumps(ratings)}])
    if failed:
        print(f"{name}: {failed} failed; re-run to retry them")


def run(cfg, sample_dir, limit=None):
    """Base judge: the same prompt for every rating."""
    system = (ROOT / cfg["prompt"]).read_text().strip()
    rate_texts(cfg, sample_dir, limit, lambda row, i: (system, ""))


def load_profile(path, values):
    """A fixed profile from a YAML file in prompts/profiles/ (`values: {value: weight, ...}`), as
    [(value, weight), ...] in file order with weights rescaled to sum to 1. Every value must be in `values`."""
    spec = yaml.safe_load((ROOT / path).read_text())["values"]
    missing = [v for v in spec if v not in values]
    if missing:
        raise ValueError(f"{path}: values not in the taxonomy: {missing}")
    total = sum(spec.values())
    return [(v, w / total) for v, w in spec.items()]


def run_dmp(cfg, sample_dir, limit=None):
    """DMP judge. By default rating i gets a random profile of `top_k` values (default 10) from `values`, seeded by
    the comment id and seed + i. With `profile: <file>` (a YAML file in prompts/profiles/ with value: weight) every rating gets that fixed
    profile instead. `equal_weights: true` gives every value the same weight."""
    template = (ROOT / cfg["prompt"]).read_text().strip()
    values = load_values(cfg["values"])
    fixed = load_profile(cfg["profile"], values) if cfg.get("profile") else None

    def prompt_for(row, i):
        profile = fixed or sample_profile(row.id, values, cfg.get("alpha", 10), cfg.get("top_k", 10),
                                          cfg.get("seed", 0) + i)
        if cfg.get("equal_weights", False):
            profile = [(v, 1 / len(profile)) for v, _ in profile]
        profile = format_profile(profile)
        return template.replace("{profile}", profile), profile

    rate_texts(cfg, sample_dir, limit, prompt_for)
