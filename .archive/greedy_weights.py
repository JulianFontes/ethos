"""Greedy search for profile weights that steer the LLM: expression toward NOT TOXIC, protective toward TOXIC.

Tunes on contested comments (consensus < 0.7) from the train sample (data/processed/consensus_n125_train). The 10
values in each profile (prompts/profiles/<name>.yaml) stay fixed; only their weights change.

  score(weights) = share of TOXIC answers over `n_comments` comments x `samples` answers each
  start from equal weights; each round try shifting `step` of the weight onto each value in turn
  (w -> (1 - step) * w + step on that value), keep the best move if it improves the score, stop when none does

Goal: expression minimises the score, protective maximises it.

Writes to <out>/:
  evaluations.csv              every weight setting tried, with its score
  <profile>_tuned.yaml         the best weights, in the same format as prompts/profiles/

Usage:
    python scripts/experiments/greedy_weights.py                                    # both profiles
    python scripts/experiments/greedy_weights.py --profiles expression --rounds 1   # quick check
    python scripts/experiments/greedy_weights.py --model qwen3:8b --workers 4      # local, free
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ethos.judges.llm_judge import SCHEMA, load_profile
from ethos.llm import ollama_client, openai_client
from ethos.llm.dmp import format_profile, load_values

ROOT = Path(__file__).resolve().parents[2]
PROMPT = ROOT / "prompts/judge_minimal_pluralistic.txt"
GOAL = {"expression": "min", "protective": "max"}   # expression: fewer TOXIC answers, protective: more


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profiles", nargs="+", default=list(GOAL), choices=list(GOAL))
    p.add_argument("--sample", default="data/processed/consensus_n125_train/sample.csv")
    p.add_argument("--n-comments", type=int, default=12, help="contested train comments to tune on")
    p.add_argument("--samples", type=int, default=5, help="answers per comment per evaluation")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--step", type=float, default=0.1, help="share of weight moved onto one value per move")
    p.add_argument("--model", default="gpt-6-luna", help="gpt-* through the OpenAI API, anything else through Ollama")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "data/experiments/greedy_weights")
    args = p.parse_args()

    comments = pd.read_csv(ROOT / args.sample).query("consensus < 0.7")
    texts = comments.sample(args.n_comments, random_state=args.seed).comment_text.tolist()
    template = PROMPT.read_text().strip()
    llm = openai_client if args.model.startswith("gpt") else ollama_client
    client = llm.get_client(args.model)
    taxonomy = load_values("prompts/taxonomy-values.txt")
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "evaluations.csv"

    def answer(system, text):
        r = llm.chat_json(client, args.model, system, f"Comment:\n{text}", SCHEMA, temperature=1.0,
                                    reasoning_effort="none")
        return r["evaluation"] == "TOXIC"

    def score(values, weights):
        """Share of TOXIC answers with this profile, over all comments and samples."""
        system = template.replace("{profile}", format_profile(list(zip(values, weights))))
        with ThreadPoolExecutor(args.workers) as pool:
            toxic = list(pool.map(lambda t: answer(system, t), [t for t in texts for _ in range(args.samples)]))
        return float(np.mean(toxic))

    for name in args.profiles:
        values = [v for v, _ in load_profile(f"prompts/profiles/{name}.yaml", taxonomy)]
        pick = min if GOAL[name] == "min" else max
        progress = Progress(name, total=1 + args.rounds * len(values))
        best_w = np.full(len(values), 1 / len(values))
        best = score(values, best_w)
        record(log, name, 0, "start (equal weights)", values, best_w, best)
        progress.step("start (equal weights)", best, best)
        print(f"{name}: start {best:.2f}", flush=True)
        for rnd in range(1, args.rounds + 1):
            tried = []
            for k, v in enumerate(values):
                w = (1 - args.step) * best_w
                w[k] += args.step
                s = score(values, w)
                record(log, name, rnd, v, values, w, s)
                progress.step(f"round {rnd}/{args.rounds}, {k + 1}/{len(values)} {v}", s, best)
                tried.append((s, v, w))
            s, v, w = pick(tried, key=lambda t: t[0])
            print(f"{name}: round {rnd} best move -> {v} ({s:.2f}, current {best:.2f})", flush=True)
            if pick(s, best) == best:   # no improvement
                print(f"{name}: no move improves; stopping", flush=True)
                break
            best, best_w = s, w
        tuned = {"values": {v: round(float(w), 3) for v, w in zip(values, best_w)}}
        (args.out / f"{name}_tuned.yaml").write_text(
            f"# Tuned by scripts/experiments/greedy_weights.py; score {best:.2f} ({GOAL[name]} share TOXIC)\n"
            + yaml.safe_dump(tuned, sort_keys=False))
        print(f"{name}: final {best:.2f} -> {format_profile(list(zip(values, best_w)))} "
              f"({progress.done} evaluations, {progress.elapsed()})", flush=True)


class Progress:
    """Prints one line per evaluation: count against the most this profile can take, scores, time left.

    `total` is the worst case (every round runs); the search stops earlier when no move improves.
    """
    def __init__(self, name, total):
        self.name, self.total, self.done, self.t0 = name, total, 0, time.time()

    def elapsed(self):
        return fmt(time.time() - self.t0)

    def step(self, move, score, best):
        self.done += 1
        left = (time.time() - self.t0) / self.done * (self.total - self.done)
        print(f"  [{self.name}] {self.done}/{self.total} max | {move}: {score:.2f} (best {best:.2f}) | "
              f"{self.elapsed()} elapsed, <= {fmt(left)} left", flush=True)


def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def record(path, name, rnd, move, values, weights, score):
    row = {"profile": name, "round": rnd, "move": move, "score": score,
           "weights": json.dumps({v: round(float(w), 3) for v, w in zip(values, weights)})}
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


if __name__ == "__main__":
    main()
