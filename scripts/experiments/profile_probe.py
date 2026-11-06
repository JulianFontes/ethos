"""Profile probe: on a few comments, does a random moral profile (DMP) change the LLM's TOXIC / NOT TOXIC answers?

A small, standalone experiment (it does not touch data/judgements/). Default comments: 6 low-consensus comments
(3 from consensus 0.5-0.6, 3 from 0.6-0.7) and 2 controls (one clearly toxic, one clearly clean). Each comment is
rated with the minimal prompt (the same prompt and answer format as ethos.judges.llm_judge):

  --set minimal          M    no profile (prompts/judge_minimal.txt)                                   10 ratings
  --set minimal_dmp      M-B  random DMP: a new random 10-value profile per rating
                              (prompts/judge_minimal_pluralistic.txt)                                  10 ratings
  --set minimal_compare  both, in one output folder

score = share of ratings that are TOXIC. Older probes (annotator instructions, reasoning, thinking) are in .artifact/.

Writes to the --out folder (default data/experiments/profile_probe_<set>/):
  ratings.csv   one row per rating (id, model, condition, rating number, profile, evaluation, rationale)
  summary.csv   score per comment and condition, with the human score and Detoxify
Re-running skips ratings already in ratings.csv.

Usage:
    python scripts/experiments/profile_probe.py --set minimal_compare --equal-weights --model gpt-6-luna --workers 8
    python scripts/experiments/profile_probe.py --set minimal --ids 7182289 --scale 0.1 --out /tmp/probe   # smoke test
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ethos.judges.llm_judge import SCHEMA
from ethos.llm import ollama_client, openai_client
from ethos.llm.dmp import format_profile, load_values, sample_profile

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/processed/consensus_n125/sample.csv"
JUDGEMENTS = ROOT / "data/judgements/consensus_n125/judgements.csv"
BASE_URL, TEMPERATURE = "http://localhost:11434/v1", 1.0
MODELS = ["qwen3:8b", "gpt-6-luna"]   # qwen3:8b through local Ollama, gpt-6-luna through the OpenAI API

COMMENTS = {   # id: role
    7182289: "0.5-0.6", 7055589: "0.5-0.6", 7037307: "0.5-0.6",
    7186928: "0.6-0.7", 7041111: "0.6-0.7", 7121149: "0.6-0.7",
    7006136: "control: toxic", 7095114: "control: clean",
}
MINIMAL = "prompts/judge_minimal.txt"
MINIMAL_DMP = "prompts/judge_minimal_pluralistic.txt"
# set: {condition: (prompt, random profile?, ratings)}
BASE = {"M no profile, minimal prompt": (MINIMAL, False, 10)}
DMP = {"M-B random DMP, minimal": (MINIMAL_DMP, True, 10)}
SETS = {"minimal": BASE, "minimal_dmp": DMP, "minimal_compare": {**BASE, **DMP}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=list(SETS), default="minimal_compare", help="which conditions to run")
    p.add_argument("--ids", nargs="+", type=int, default=list(COMMENTS), help="comment ids (default: all 8)")
    p.add_argument("--model", choices=MODELS, default="qwen3:8b",
                   help="qwen3:8b (local Ollama) or gpt-6-luna (OpenAI API, needs OPENAI_API_KEY in .env)")
    p.add_argument("--equal-weights", action="store_true", help="give every profile value the same weight")
    p.add_argument("--scale", type=float, default=1.0, help="multiply ratings per condition (smoke test: 0.1)")
    p.add_argument("--workers", type=int, default=4, help="calls in flight")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: data/experiments/profile_probe_<set>)")
    args = p.parse_args()
    conditions = SETS[args.set]
    out = args.out or ROOT / "data/experiments" / f"profile_probe_{args.set}"

    values = load_values("prompts/taxonomy-values.txt")
    prompts = {f: (ROOT / f).read_text().strip() for f in {c[0] for c in conditions.values()}}
    texts = pd.read_csv(SAMPLE).set_index("id").comment_text

    out.mkdir(parents=True, exist_ok=True)
    path = out / "ratings.csv"
    done = set()
    if path.exists():
        r = pd.read_csv(path)
        done = set(zip(r.id, r.condition, r.rating))

    jobs = []
    for cid in args.ids:
        for cond, (prompt, random_profile, n) in conditions.items():
            for i in range(max(1, round(n * args.scale))):
                if (cid, cond, i) in done:
                    continue
                profile = ""
                if random_profile:
                    prof = sample_profile(cid, values, 10, 10, i)
                    if args.equal_weights:
                        prof = [(v, 1 / len(prof)) for v, _ in prof]
                    profile = format_profile(prof)
                jobs.append((cid, cond, i, prompts[prompt].replace("{profile}", profile), profile))
    print(f"{len(jobs)} ratings to do ({len(done)} already in {path})")

    if jobs:
        if args.model.startswith("gpt"):
            llm, client = openai_client, openai_client.get_client(args.model)
        else:
            llm, client = ollama_client, ollama_client.get_client(args.model, BASE_URL)

        def rate(job):
            cid, cond, i, system, profile = job
            try:
                return job, llm.chat_json(client, args.model, system, f"Comment:\n{texts[cid]}", SCHEMA,
                                          temperature=TEMPERATURE, reasoning_effort="none")
            except Exception as e:  # connection error, bad JSON, refusal, rate limit
                return job, e

        failed = 0
        with ThreadPoolExecutor(args.workers) as pool:
            for fut in tqdm(as_completed([pool.submit(rate, j) for j in jobs]), total=len(jobs), unit="rating"):
                (cid, cond, i, _, profile), answer = fut.result()
                if isinstance(answer, Exception):
                    failed += 1
                    continue
                row = {"id": cid, "model": args.model, "condition": cond, "rating": i, "profile": profile, **answer}
                pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
        if failed:
            print(f"{failed} ratings failed; re-run to retry them")
    summarize(path, out, conditions, args.model)


def summarize(path, out, conditions, model):
    """Share of TOXIC ratings per comment and condition, next to the human score and Detoxify."""
    r = pd.read_csv(path).assign(toxic=lambda d: d.evaluation == "TOXIC")
    score = r.pivot_table(index="id", columns="condition", values="toxic", aggfunc="mean")
    score = score[[c for c in conditions if c in score]]
    j = pd.read_csv(JUDGEMENTS)
    detoxify = j[(j.judge == "detoxify") & (j.version == "original")].set_index("id").score
    human = pd.read_csv(SAMPLE).set_index("id").toxicity
    table = pd.DataFrame({"role": pd.Series(COMMENTS), "human": human, "detoxify": detoxify}).loc[score.index]
    table = table.join(score)
    table.to_csv(out / "summary.csv")
    pd.set_option("display.width", 250)
    print(f"\nShare of {model} ratings TOXIC:")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
