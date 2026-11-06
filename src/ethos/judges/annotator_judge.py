"""LLM annotator judge: an LLM rates each comment the way the Civil Comments annotators did, several times over.

The prompt (prompts/judge_annotator.txt) is the original annotator instructions. Each call returns a toxicity level
(Very Toxic / Toxic / Slightly Toxic or Hard to Say / Not Toxic) and the five attribute answers (Yes / Hard to say / No).
The comment is rated `samples` times (default 10, sampled at `temperature`, default 1.0), like a panel of annotators:

    score    = share of ratings that are Toxic or Very Toxic   (how Civil Comments computes the human `toxicity` score)
    decision = "remove" if score >= threshold (default 0.5), as for the human judge
    ratings  = JSON list of every rating, for later analysis (e.g. the model's own consensus, max(score, 1 - score))

Works with Ollama (config type "ollama_annotator") or the OpenAI API ("openai_annotator").

Pluralistic variant (types "ollama_annotator_dmp" / "openai_annotator_dmp", after Dynamic Moral Profiling): each of
the `samples` ratings gets its own value profile (ethos.llm.dmp, seeded by comment id and rating number), filled into
the prompt's {profile} slot, like a panel of annotators with different values. `profile` holds the list of profiles.
A comment is written only when all its ratings succeed; otherwise it is retried on the next run.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ethos.judges.core import ROOT, append_rows, output_path, todo
from ethos.llm import ollama_client, openai_client
from ethos.llm.dmp import format_profile, load_values, sample_profile
from ethos.llm.ollama_client import DEFAULT_BASE_URL

TOXICITY_LEVELS = ["Very Toxic", "Toxic", "Slightly Toxic or Hard to Say", "Not Toxic"]
TOXIC_LEVELS = {"Very Toxic", "Toxic"}           # counted as toxic, as in the human `toxicity` score
ATTRIBUTES = ["profanity_obscenity", "sexually_explicit", "identity_attack", "insult", "threat"]
ATTRIBUTE_ANSWERS = ["Yes", "Hard to say", "No"]

RATING_SCHEMA = {
    "type": "object",
    "properties": {
        "toxicity": {"type": "string", "enum": TOXICITY_LEVELS},
        **{a: {"type": "string", "enum": ATTRIBUTE_ANSWERS} for a in ATTRIBUTES},
    },
    "required": ["toxicity", *ATTRIBUTES],
    "additionalProperties": False,
}


# Minimal judge (types "ollama_minimal" / "openai_minimal"): a short TOXIC / NOT TOXIC prompt with a rationale,
# no annotator instructions (prompts/judge_minimal.txt), rated `samples` times like the annotator judge.
MINIMAL_SCHEMA = {
    "type": "object",
    "properties": {"evaluation": {"type": "string", "enum": ["TOXIC", "NOT TOXIC"]},
                   "rationale": {"type": "string"}},
    "required": ["evaluation", "rationale"],
    "additionalProperties": False,
}


def annotate_texts(cfg, sample_dir, limit, prompt_for, schema=RATING_SCHEMA,
                   is_toxic=lambda r: r["toxicity"] in TOXIC_LEVELS):
    """Rate every remaining comment version `samples` times.

    `prompt_for(row, i)` returns (system_prompt, profile) for rating i; profile is "" when there is none.
    """
    name, n = cfg["name"], cfg.get("samples", 10)
    threshold, temperature = cfg.get("threshold", 0.5), cfg.get("temperature", 1.0)
    llm = openai_client if cfg["type"].startswith("openai") else ollama_client
    client = llm.get_client(cfg["model"], cfg.get("base_url", DEFAULT_BASE_URL))  # fails fast if unavailable
    texts = todo(sample_dir, name, limit, cfg.get("query"))
    out = output_path(sample_dir)
    if texts.empty:
        print(f"{name}: nothing to do")
        return

    def rate(row):
        ratings, profiles = [], []
        for i in range(n):
            system, profile = prompt_for(row, i)
            profiles.append(profile)
            try:
                ratings.append(llm.chat_json(client, cfg["model"], system, f"Comment:\n{row.text}", schema,
                                             temperature=temperature, max_tokens=cfg.get("max_tokens"),
                                             reasoning_effort=cfg.get("reasoning_effort")))
            except Exception as e:  # connection error, bad JSON, refusal, rate limit
                return row, "", e
        return row, json.dumps(profiles) if any(profiles) else "", ratings

    # `workers` comments in flight at once (default 1); rows are written from this thread only, as they finish.
    failed = 0
    with ThreadPoolExecutor(max_workers=cfg.get("workers", 1)) as pool:
        futures = [pool.submit(rate, row) for row in texts.itertuples(index=False)]
        for future in tqdm(as_completed(futures), total=len(futures), desc=name, unit="text"):
            row, profile, ratings = future.result()
            if isinstance(ratings, Exception):
                failed += 1
                tqdm.write(f"{name}: failed {row.id}/{row.version}: {str(ratings)[:120]}")
                continue
            score = sum(is_toxic(r) for r in ratings) / len(ratings)
            append_rows(out, [{"id": row.id, "version": row.version, "text": row.text, "judge": name,
                               "score": round(score, 4), "decision": "remove" if score >= threshold else "allow",
                               "rationale": "", "profile": profile, "ratings": json.dumps(ratings)}])
    if failed:
        print(f"{name}: {failed} failed; re-run to retry them")


def run(cfg, sample_dir, limit=None):
    system = (ROOT / cfg["prompt"]).read_text().strip()
    annotate_texts(cfg, sample_dir, limit, lambda row, i: (system, ""))


def run_minimal(cfg, sample_dir, limit=None):
    """score = share of the `samples` ratings that are TOXIC."""
    system = (ROOT / cfg["prompt"]).read_text().strip()
    annotate_texts(cfg, sample_dir, limit, lambda row, i: (system, ""), schema=MINIMAL_SCHEMA,
                   is_toxic=lambda r: r["evaluation"] == "TOXIC")


def run_dmp(cfg, sample_dir, limit=None):
    """Each rating gets its own value profile: Dirichlet(alpha * uniform) over the taxonomy, top_k values kept."""
    template = (ROOT / cfg["prompt"]).read_text().strip()
    values = load_values(cfg["values"])

    def prompt_for(row, i):
        profile = format_profile(sample_profile(row.id, values, cfg.get("alpha", 10), cfg.get("top_k", 3),
                                                cfg.get("seed", 0) + i))
        return template.replace("{profile}", profile), profile

    annotate_texts(cfg, sample_dir, limit, prompt_for)
