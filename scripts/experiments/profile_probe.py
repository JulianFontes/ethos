"""Profile probe: can moral profiles make the LLM annotator budge on comments the annotators split on?

A small, standalone experiment (it does not touch data/judgements/). For 6 low-consensus comments (3 from consensus
0.5-0.6, 3 from 0.6-0.7, all rated 10/10 "Toxic" by the base qwen3:8b annotator) and 2 controls (one clearly toxic,
one clearly clean), qwen3:8b rates each comment with the annotator instructions under several profile conditions.
Every profile has 10 values from prompts/taxonomy-values.txt.

  B  random DMP     a new random profile per rating: Dirichlet(alpha=10 * uniform), top 10 values   10 ratings
  C  expression     fixed: Freedom, Honesty, Autonomy, ...                                            5 ratings
  D  protective     fixed: Respect, Compassion, Inclusivity, ...                                      5 ratings
  F-C, F-D          C and D with the stronger wording (prompts/judge_annotator_pluralistic_strong.txt)  5 each
  (A, no profile, is the base annotator's 10 ratings, read from data/judgements/consensus_n125/judgements.csv.)

score = share of ratings that are Toxic or Very Toxic, as for the human score.

--set reasoning runs a second probe (output: data/experiments/profile_probe_reasoning/): the answer starts with a
"reasoning" field (1-3 sentences) that the model writes before the toxicity level, so the label can follow from it.
  A+R  no profile, reasoning only (prompts/judge_annotator.txt)       5 ratings
  B+R  random DMP + reasoning                                         10 ratings
  C+R  expression + reasoning                                          5 ratings
  D+R  protective + reasoning                                          5 ratings
All use the standard profile wording, so the only change from the first probe is the reasoning step.

--set thinking (output: data/experiments/profile_probe_thinking/) uses the same four conditions (A+T, B+T, C+T, D+T)
with qwen3's thinking mode on instead of the reasoning field: the model thinks freely before answering. Its thinking
text is saved in a `thinking` column. 10 ratings per condition. About 4x slower per rating.

Writes to data/experiments/profile_probe/:
  ratings.csv   one row per rating (id, condition, rating number, profile, toxicity level, attributes)
  summary.csv   score per comment and condition, with the human score and Detoxify
Re-running skips ratings already in ratings.csv.

Usage:
    python scripts/experiments/profile_probe.py
    python scripts/experiments/profile_probe.py --set reasoning
    python scripts/experiments/profile_probe.py --set thinking --ids 7182289
    python scripts/experiments/profile_probe.py --set thinking --ids 7182289 --model gpt-6-luna --workers 8
    python scripts/experiments/profile_probe.py --set minimal --ids 7182289 7006136 7095114 --model gpt-6-luna --workers 8
    python scripts/experiments/profile_probe.py --ids 7182289 --scale 0.2 --out /tmp/probe   # quick smoke test
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ethos.judges.annotator_judge import RATING_SCHEMA, TOXIC_LEVELS
from ethos.llm import ollama_client, openai_client
from ethos.llm.dmp import format_profile, load_values, sample_profile

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/processed/consensus_n125/sample.csv"
JUDGEMENTS = ROOT / "data/judgements/consensus_n125/judgements.csv"
BASE_URL, TEMPERATURE = "http://localhost:11434/v1", 1.0
# reasoning_effort per model: (thinking off, thinking on). None = leave unset (qwen3 thinks by default in Ollama).
EFFORT = {"qwen3:8b": ("none", None), "gpt-6-luna": ("none", "medium")}

COMMENTS = {   # id: role
    7182289: "0.5-0.6", 7055589: "0.5-0.6", 7037307: "0.5-0.6",
    7186928: "0.6-0.7", 7041111: "0.6-0.7", 7121149: "0.6-0.7",
    7006136: "control: toxic", 7095114: "control: clean",
}
WEIGHTS = [0.18, 0.15, 0.13, 0.11, 0.10, 0.09, 0.08, 0.06, 0.05, 0.05]
EXPRESSION = ["Freedom", "Honesty", "Autonomy", "Individualism", "Justice", "Integrity", "Self-Reliance",
              "Pragmatism", "Knowledge", "Curiosity"]
PROTECTIVE = ["Respect", "Compassion", "Inclusivity", "Cultural Respect", "Sensitivity", "Consideration", "Harmony",
              "Nonviolence", "Care", "Emotional Intelligence"]
PROMPT = "prompts/judge_annotator_pluralistic.txt"
STRONG = "prompts/judge_annotator_pluralistic_strong.txt"
NO_PROFILE = "prompts/judge_annotator.txt"
MINIMAL = "prompts/judge_minimal.txt"   # short TOXIC / NOT TOXIC prompt, no annotator instructions
MINIMAL_DMP = "prompts/judge_minimal_pluralistic.txt"   # the same, with a moral profile
MINIMAL_SCHEMA = {"type": "object",
                  "properties": {"evaluation": {"type": "string", "enum": ["TOXIC", "NOT TOXIC"]},
                                 "rationale": {"type": "string"}},
                  "required": ["evaluation", "rationale"], "additionalProperties": False}
# set: {condition: (prompt, profile kind, ratings, reasoning field first?, thinking on?)}
SETS = {
    "profiles": {"A no profile": (NO_PROFILE, "none", 10, False, False),
                 "B random DMP": (PROMPT, "random", 10, False, False),
                 "C expression": (PROMPT, "expression", 5, False, False),
                 "D protective": (PROMPT, "protective", 5, False, False),
                 "F-C expression, strong wording": (STRONG, "expression", 5, False, False),
                 "F-D protective, strong wording": (STRONG, "protective", 5, False, False)},
    "reasoning": {"A+R no profile, reasoning": (NO_PROFILE, "none", 5, True, False),
                  "B+R random DMP, reasoning": (PROMPT, "random", 10, True, False),
                  "C+R expression, reasoning": (PROMPT, "expression", 5, True, False),
                  "D+R protective, reasoning": (PROMPT, "protective", 5, True, False)},
    "minimal": {"M no profile, minimal prompt": (MINIMAL, "none", 10, False, False)},
    "minimal_dmp": {"M-B random DMP, minimal": (MINIMAL_DMP, "random", 10, False, False)},
    # base minimal prompt and random DMP side by side, in one output folder
    "minimal_compare": {"M no profile, minimal prompt": (MINIMAL, "none", 10, False, False),
                        "M-B random DMP, minimal": (MINIMAL_DMP, "random", 10, False, False)},
    "thinking": {"A+T no profile, thinking": (NO_PROFILE, "none", 10, False, True),
                 "B+T random DMP, thinking": (PROMPT, "random", 10, False, True),
                 "C+T expression, thinking": (PROMPT, "expression", 10, False, True),
                 "D+T protective, thinking": (PROMPT, "protective", 10, False, True)},
}
# Added at the end of the prompt when the answer starts with "reasoning" (with / without a profile).
REASON_PROFILE = ('Before choosing the toxicity level, write 1-3 sentences in "reasoning": explain how the values in '
                  'your moral profile bear on whether this comment is toxic, and where you draw the line for it. '
                  'Then give your ratings.')
REASON_PLAIN = ('Before choosing the toxicity level, write 1-3 sentences in "reasoning": explain whether and why this '
                'comment is toxic, and where you draw the line for it. Then give your ratings.')
# Same as RATING_SCHEMA, with "reasoning" first so it is written before the toxicity level.
REASONING_SCHEMA = {**RATING_SCHEMA,
                    "properties": {"reasoning": {"type": "string"}, **RATING_SCHEMA["properties"]},
                    "required": ["reasoning", *RATING_SCHEMA["required"]]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="+", type=int, default=list(COMMENTS), help="comment ids (default: all 8)")
    p.add_argument("--scale", type=float, default=1.0, help="multiply ratings per condition (smoke test: 0.2)")
    p.add_argument("--workers", type=int, default=4, help="calls in flight (needs OLLAMA_NUM_PARALLEL)")
    p.add_argument("--set", choices=list(SETS), default="profiles", help="which conditions to run")
    p.add_argument("--equal-weights", action="store_true",
                   help="give every value in a profile the same weight (0.10 each for 10 values)")
    p.add_argument("--model", choices=list(EFFORT), default="qwen3:8b",
                   help="qwen3:8b (local Ollama) or gpt-6-luna (OpenAI API, needs OPENAI_API_KEY in .env)")
    p.add_argument("--out", type=Path, default=None,
                   help="output folder (default: data/experiments/profile_probe, or ..._<set> for other sets)")
    args = p.parse_args()
    conditions = SETS[args.set]
    if args.out is None:
        args.out = ROOT / "data/experiments" / ("profile_probe" if args.set == "profiles" else f"profile_probe_{args.set}")

    values = load_values("prompts/taxonomy-values.txt")
    for v in EXPRESSION + PROTECTIVE:
        assert v in values, f"{v} is not in the taxonomy"
    fixed = {"expression": list(zip(EXPRESSION, WEIGHTS)), "protective": list(zip(PROTECTIVE, WEIGHTS))}
    prompts = {f: (ROOT / f).read_text().strip() for f in {c[0] for c in conditions.values()}}   # only this set's
    texts = pd.read_csv(SAMPLE).set_index("id").comment_text

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "ratings.csv"
    done = set()
    if path.exists():
        r = pd.read_csv(path)
        done = set(zip(r.id, r.condition, r.rating))

    jobs = []
    for cid in args.ids:
        for cond, (prompt, kind, n, reason, think) in conditions.items():
            for i in range(max(1, round(n * args.scale))):
                if (cid, cond, i) in done:
                    continue
                prof = [] if kind == "none" else sample_profile(cid, values, 10, 10, i) if kind == "random" else fixed[kind]
                if args.equal_weights:
                    prof = [(v, 1 / len(prof)) for v, _ in prof]
                prof = format_profile(prof) if prof else ""
                jobs.append((cid, cond, i, prompt, prof, reason, think))
    print(f"{len(jobs)} ratings to do ({len(done)} already in {path})")
    if not jobs:
        return summarize(path, args.out, conditions, args.model)

    model = args.model
    off, on = EFFORT[model]
    if model.startswith("gpt"):
        load_env()
        client = openai_client.get_client(model)
    else:
        client = ollama_client.get_client(model, BASE_URL)

    def rate(job):
        cid, cond, i, prompt, profile, reason, think = job
        system = prompts[prompt].replace("{profile}", profile)
        if reason:
            system += "\n\n" + (REASON_PROFILE if profile else REASON_PLAIN)
        schema = MINIMAL_SCHEMA if prompt in (MINIMAL, MINIMAL_DMP) else REASONING_SCHEMA if reason else RATING_SCHEMA
        effort = on if think else off
        try:
            # One call for every case; Ollama returns qwen3's thinking in `reasoning`, OpenAI returns none.
            resp = client.chat.completions.create(
                model=model, temperature=TEMPERATURE,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": f"Comment:\n{texts[cid]}"}],
                response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": schema, "strict": True}},
                **({"extra_body": {"reasoning_effort": effort}} if effort else {}))
            msg = resp.choices[0].message
            out = json.loads(msg.content)
            if think:
                out["thinking"] = (msg.model_extra or {}).get("reasoning", "")
            return job, out
        except Exception as e:
            return job, e

    failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for fut in tqdm(as_completed([pool.submit(rate, j) for j in jobs]), total=len(jobs), unit="rating"):
            (cid, cond, i, prompt, profile, reason, think), out = fut.result()
            if isinstance(out, Exception):
                failed += 1
                continue
            row = pd.DataFrame([{"id": cid, "model": model, "condition": cond, "rating": i, "profile": profile, **out}])
            row.to_csv(path, mode="a", header=not path.exists(), index=False)
    if failed:
        print(f"{failed} ratings failed; re-run to retry them")
    summarize(path, args.out, conditions, model)


def load_env():
    """Put KEY=value lines from the project's .env into the environment (for OPENAI_API_KEY)."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not line.lstrip().startswith("#") and key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


def summarize(path, out, conditions, model):
    r = pd.read_csv(path)
    r["toxic"] = r.toxicity.isin(TOXIC_LEVELS) if "toxicity" in r else False
    if "evaluation" in r:   # minimal prompt: TOXIC / NOT TOXIC
        r.loc[r.evaluation.notna(), "toxic"] = r.evaluation == "TOXIC"
    score = r.pivot_table(index="id", columns="condition", values="toxic", aggfunc="mean")
    score = score[[c for c in conditions if c in score]]
    n = r.pivot_table(index="id", columns="condition", values="toxic", aggfunc="size")
    j = pd.read_csv(JUDGEMENTS)
    j = j[j.version == "original"].pivot_table(index="id", columns="judge", values="score")
    s = pd.read_csv(SAMPLE).set_index("id")
    cols = {"role": pd.Series(COMMENTS), "human": s.toxicity, "detoxify": j.detoxify}
    if model == "qwen3:8b":   # the qwen base annotator's earlier score; other models have only their own "A" condition
        cols["qwen3:8b base annotator (judgements)"] = j["qwen3:8b-annotator"]
    table = pd.DataFrame(cols).loc[score.index].join(score)
    table.to_csv(out / "summary.csv")
    pd.set_option("display.width", 250)
    print(f"\nShare of {model} ratings Toxic or Very Toxic:")
    print(table.round(2).to_string())
    print("\nRatings per cell:", n.stack().min(), "to", n.stack().max())


if __name__ == "__main__":
    main()
