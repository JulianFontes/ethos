"""Profile signal: do the identity and the weighting of a profile's values change the LLM's TOXIC answers?

Every profile in prompts/profiles/*.yaml is run three ways (the same values, different weights):
  as-is      the file's weights
  equal      every value the same weight
  reversed   the file's weights in reverse order (the top value gets the smallest weight)
plus a no-profile baseline, on the same uncertain train comments (comments whose answers varied under random DMP,
read from data/judgements/consensus_n125_train/judgements.csv), `samples` answers each, minimal prompt.

Tests (permutation: shuffle answers between the compared conditions within each comment):
  identity   do the profiles (as-is) differ from each other more than chance?
  weighting  within each profile, do as-is / equal / reversed differ more than chance?
  profile    does adding a profile change answers vs. no profile?

Writes to <out>/: ratings.csv (one row per answer; re-running skips answers already there) and summary.csv.

Usage:
    python scripts/experiments/profile_signal.py --model gpt-6-luna --workers 16
    python scripts/experiments/profile_signal.py --model qwen3:8b --workers 4
    python scripts/experiments/profile_signal.py --all --weightings as-is --samples 10 --out data/experiments/profile_signal/train_all
    python scripts/experiments/profile_signal.py --ids 5705144 --samples 10 --out data/experiments/profile_signal/one_comment
    python scripts/experiments/profile_signal.py --analyze-only --out data/experiments/profile_signal/gpt
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ethos.judges.llm_judge import SCHEMA, load_profile
from ethos.llm import ollama_client, openai_client
from ethos.llm.dmp import format_profile, load_values

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/processed/consensus_n125_train/sample.csv"
RANDOM_DMP = ROOT / "data/judgements/consensus_n125_train/judgements.csv"
BASE_PROMPT = ROOT / "prompts/judge_minimal.txt"
DMP_PROMPT = ROOT / "prompts/judge_minimal_pluralistic.txt"


def uncertain_comments(n):
    """The n train comments whose random-DMP answers were most mixed (closest to a 50/50 split)."""
    j = pd.read_csv(RANDOM_DMP)
    s = j[j.judge == "gpt-6-luna-dmp-random"].set_index("id").score
    return (s - 0.5).abs().sort_values().index[:n].tolist()


def conditions(taxonomy, weightings=("as-is", "equal", "reversed")):
    """{condition name: profile string}; "" = no profile."""
    out = {"none": ""}
    for path in sorted((ROOT / "prompts/profiles").glob("*.yaml")):
        prof = load_profile(path.relative_to(ROOT), taxonomy)
        values, weights = [v for v, _ in prof], [w for _, w in prof]
        variants = {"as-is": prof, "equal": [(v, 1 / len(values)) for v in values],
                    "reversed": list(zip(values, weights[::-1]))}
        for w in weightings:
            out[f"{path.stem}|{w}"] = format_profile(variants[w])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt-6-luna", help="gpt-* through the OpenAI API, anything else through Ollama")
    p.add_argument("--n-comments", type=int, default=20)
    p.add_argument("--ids", nargs="+", type=int, default=None, help="specific comment ids instead of the most uncertain")
    p.add_argument("--all", action="store_true", help="every comment in --sample instead of the most uncertain")
    p.add_argument("--sample", type=Path, default=SAMPLE, help="sample.csv to draw comments from")
    p.add_argument("--weightings", nargs="+", default=["as-is", "equal", "reversed"],
                   choices=["as-is", "equal", "reversed"], help="which weightings of each profile to run")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=Path, default=None, help="default: data/experiments/profile_signal/<model>")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    out = args.out or ROOT / "data/experiments/profile_signal" / args.model.split(":")[0]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ratings.csv"

    if not args.analyze_only:
        conds = conditions(load_values("prompts/taxonomy-values.txt"), args.weightings)
        ids = args.ids or (pd.read_csv(args.sample).id.tolist() if args.all else uncertain_comments(args.n_comments))
        texts = pd.read_csv(args.sample).set_index("id").comment_text
        done = set()
        if path.exists():
            r = pd.read_csv(path)
            done = set(zip(r.id, r.condition, r.rating))
        jobs = [(cid, c, i) for cid in ids for c in conds for i in range(args.samples) if (cid, c, i) not in done]
        print(f"{len(conds)} conditions x {len(ids)} comments x {args.samples} answers: {len(jobs)} to do")
        llm = openai_client if args.model.startswith("gpt") else ollama_client
        client = llm.get_client(args.model)
        base, dmp = BASE_PROMPT.read_text().strip(), DMP_PROMPT.read_text().strip()

        def rate(job):
            cid, c, i = job
            system = dmp.replace("{profile}", conds[c]) if conds[c] else base
            try:
                return job, llm.chat_json(client, args.model, system, f"Comment:\n{texts[cid]}", SCHEMA,
                                          temperature=1.0, reasoning_effort="none")
            except Exception as e:  # retried on the next run
                return job, e

        with ThreadPoolExecutor(args.workers) as pool:
            for fut in tqdm(as_completed([pool.submit(rate, j) for j in jobs]), total=len(jobs), unit="answer"):
                (cid, c, i), a = fut.result()
                if isinstance(a, Exception):
                    continue
                row = {"id": cid, "condition": c, "rating": i, "profile": conds[c], "toxic": a["evaluation"] == "TOXIC",
                       "rationale": a["rationale"]}
                pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    analyze(path, out)


def spread(r, conds, rng, n_perm=2000):
    """Variance of the conditions' TOXIC shares (within-comment centred) and its permutation p-value."""
    x = r[r.condition.isin(conds)].copy()
    x["c"] = x.toxic - x.groupby("id").toxic.transform("mean")
    stat = lambda d: d.groupby("condition").c.mean().var()
    obs, null = stat(x), []
    for _ in range(n_perm):
        x["condition"] = x.groupby("id").condition.transform(lambda s: rng.permutation(s.values))
        null.append(stat(x))
    return obs, float(np.mean(np.array(null) >= obs))


def analyze(path, out):
    r = pd.read_csv(path)
    r["toxic"] = r.toxic.astype(bool)
    rng = np.random.default_rng(0)
    base = r[r.condition == "none"].groupby("id").toxic.mean()
    table = r.groupby("condition").toxic.mean().rename("share_toxic").to_frame()
    table["shift_vs_none"] = table.share_toxic - base.mean()
    table.to_csv(out / "summary.csv")
    pd.set_option("display.width", 200)
    print(f"\n{r.id.nunique()} comments, {len(r)} answers. Share TOXIC per condition (none = no profile):")
    print(table.sort_values("share_toxic").round(3).to_string())

    profiles = sorted({c.split("|")[0] for c in r.condition if "|" in c})
    print("\nTests (p = share of within-comment shuffles at least as extreme):")
    _, p = spread(r, ["none"] + [f"{q}|as-is" for q in profiles], rng)
    print(f"  any profile vs none + identity: p = {p:.3f}")
    _, p = spread(r, [f"{q}|as-is" for q in profiles], rng)
    print(f"  identity (the {len(profiles)} profiles, as-is, differ from each other): p = {p:.3f}")
    for q in profiles:
        variants = [f"{q}|{w}" for w in ("as-is", "equal", "reversed") if f"{q}|{w}" in set(r.condition)]
        if len(variants) < 2:
            continue
        _, p = spread(r, variants, rng, n_perm=1000)
        print(f"  weighting within {q}: p = {p:.3f}")


if __name__ == "__main__":
    main()
