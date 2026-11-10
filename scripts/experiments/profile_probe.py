"""Profile probe: on a few comments, does a random moral profile (DMP) change the LLM's TOXIC / NOT TOXIC answers?

A small, standalone experiment (it does not touch data/judgements/). Default comments: 6 low-consensus comments
(3 from consensus 0.5-0.6, 3 from 0.6-0.7) and 2 controls (one clearly toxic, one clearly clean). Each comment is
rated with the minimal prompt (the same prompt and answer format as ethos.judges.llm_judge):

  --set minimal          M    no profile (prompts/judge_minimal.txt)                                   10 ratings
  --set minimal_dmp      M-B  random DMP: a new random 10-value profile per rating
                              (prompts/judge_minimal_pluralistic.txt)                                  10 ratings
  --set minimal_compare  both, in one output folder
  --set steer            M, plus two fixed 10-value profiles (same prompt as random DMP), to test steering both ways:
                         M-C expression (Freedom, Honesty, Autonomy, ...)   should lower TOXIC answers
                         M-D protective (Respect, Compassion, ...)          should raise TOXIC answers      10 each

score = share of ratings that are TOXIC. Older probes (annotator instructions, reasoning, thinking) are in .archive/.

Data (--data, default data/processed/consensus_n125_test/profile_probe/<set>/):
  ratings.csv   one row per rating (id, model, condition, rating number, profile, evaluation, rationale)
Results (--results, default data/experiments/01_profile_probe/<set>/):
  summary.csv   score per comment and condition, with the human score and Detoxify
  sources.yaml  which data the summary came from
Re-running skips ratings already in ratings.csv.

Usage:
    python scripts/experiments/profile_probe.py --set minimal_compare --equal-weights --model gpt-6-luna --workers 8
    python scripts/experiments/profile_probe.py --set minimal --ids 7182289 --scale 0.1 --data /tmp/probe --results /tmp/probe   # smoke test
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ethos.judges.llm_judge import SCHEMA, load_profile
from ethos.llm import ollama_client, openai_client
from ethos.llm.dmp import format_profile, load_values, sample_profile
from ethos.paths import ROOT
from ethos.scores import toxicity_score
from ethos.sources import write_sources

SAMPLE = ROOT / "data/processed/consensus_n125_test/sample.csv"
JUDGEMENTS = ROOT / "data/judgements/consensus_n125_test/judgements.csv"
BASE_URL, TEMPERATURE = "http://localhost:11434/v1", 1.0
MODELS = ["qwen3:8b", "gpt-6-luna"]   # qwen3:8b through local Ollama, gpt-6-luna through the OpenAI API

COMMENTS = {   # id: role
    7182289: "0.5-0.6", 7055589: "0.5-0.6", 7037307: "0.5-0.6",
    7186928: "0.6-0.7", 7041111: "0.6-0.7", 7121149: "0.6-0.7",
    7006136: "control: toxic", 7095114: "control: clean",
}
MINIMAL = "prompts/judge_minimal.txt"
MINIMAL_DMP = "prompts/judge_minimal_pluralistic.txt"
FIXED = {"expression": "prompts/profiles/expression.yaml", "protective": "prompts/profiles/protective.yaml"}
# set: {condition: (prompt, profile: "none" / "random" / a FIXED key, ratings)}
BASE = {"M no profile, minimal prompt": (MINIMAL, "none", 10)}
DMP = {"M-B random DMP, minimal": (MINIMAL_DMP, "random", 10)}
STEER = {"M-C expression, minimal": (MINIMAL_DMP, "expression", 10),
         "M-D protective, minimal": (MINIMAL_DMP, "protective", 10)}
SETS = {"minimal": BASE, "minimal_dmp": DMP, "minimal_compare": {**BASE, **DMP}, "steer": {**BASE, **STEER}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=list(SETS), default="minimal_compare", help="which conditions to run")
    p.add_argument("--ids", nargs="+", type=int, default=list(COMMENTS), help="comment ids (default: all 8)")
    p.add_argument("--model", choices=MODELS, default="qwen3:8b",
                   help="qwen3:8b (local Ollama) or gpt-6-luna (OpenAI API, needs OPENAI_API_KEY in .env)")
    p.add_argument("--equal-weights", action="store_true", help="give every profile value the same weight")
    p.add_argument("--scale", type=float, default=1.0, help="multiply ratings per condition (smoke test: 0.1)")
    p.add_argument("--workers", type=int, default=4, help="calls in flight")
    p.add_argument("--data", type=Path, help="answers folder (default: data/processed/consensus_n125_test/profile_probe/<set>)")
    p.add_argument("--results", type=Path, help="summary folder (default: data/experiments/01_profile_probe/<set>)")
    args = p.parse_args()
    conditions = SETS[args.set]
    data = args.data or ROOT / "data/processed/consensus_n125_test/profile_probe" / args.set
    results = args.results or ROOT / "data/experiments/01_profile_probe" / args.set

    values = load_values("prompts/taxonomy-values.txt")
    fixed = {k: load_profile(path, values) for k, path in FIXED.items()}
    prompts = {f: (ROOT / f).read_text().strip() for f in {c[0] for c in conditions.values()}}
    texts = pd.read_csv(SAMPLE).set_index("id").comment_text

    data.mkdir(parents=True, exist_ok=True)
    path = data / "ratings.csv"
    done = set()
    if path.exists():
        r = pd.read_csv(path)
        done = set(zip(r.id, r.condition, r.rating))

    jobs = []
    for cid in args.ids:
        for cond, (prompt, kind, n) in conditions.items():
            for i in range(max(1, round(n * args.scale))):
                if (cid, cond, i) in done:
                    continue
                profile = ""
                if kind != "none":
                    prof = sample_profile(cid, values, 10, 10, i) if kind == "random" else fixed[kind]
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
    summarize(path, results, conditions, args.model)


def summarize(path, out, conditions, model):
    """Share of TOXIC ratings per comment and condition, next to the human score and Detoxify."""
    score = toxicity_score(pd.read_csv(path), by=["id", "condition"]).unstack("condition")
    score = score[[c for c in conditions if c in score]]
    j = pd.read_csv(JUDGEMENTS)
    detoxify = j[(j.judge == "detoxify") & (j.version == "original")].set_index("id").score
    human = pd.read_csv(SAMPLE).set_index("id").toxicity
    table = pd.DataFrame({"role": pd.Series(COMMENTS), "human": human, "detoxify": detoxify}).loc[score.index]
    table = table.join(score)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "summary.csv")
    write_sources(out, [path, SAMPLE, JUDGEMENTS])
    pd.set_option("display.width", 250)
    print(f"\nShare of {model} ratings TOXIC:")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
