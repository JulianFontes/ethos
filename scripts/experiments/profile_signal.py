"""Profile signal: do moral profiles change the LLM's TOXIC / NOT TOXIC answers?

Conditions (each asked `--samples` times per comment, minimal prompt, plus a no-profile baseline "none"):
  default          every profile in prompts/profiles/*.yaml, under each `--weightings`:
                     as-is     the file's weights
                     equal     every value the same weight
                     reversed  the file's weights in reverse order
  --single-values  each value of the taxonomy alone at weight 1 ("<value>|as-is")

Comments: the `--n-comments` train comments whose random-DMP answers were most mixed, or `--all` of them, or `--ids`.

Data (`--data`, a folder in the sample, e.g. data/processed/consensus_n125_train/single_values):
  ratings.csv   one row per answer; re-running skips answers already there
  scores.csv    one row per comment and condition: LLM toxicity score next to the human score
Distributed runs (one Slurm array task per job; --job / --num-jobs default to SLURM_ARRAY_TASK_ID / _COUNT):
  comments are put in a fixed, band-balanced random order (order.csv) and job i takes every Nth one, writing
  jobs/job_<i>of<N>.csv in comment order. --merge combines the jobs into ratings.csv and scores.csv (complete
  comments only) and writes checkpoints/n<k>/scores.csv once the first k comments in the order are complete
  (k = --checkpoint-every, 2x, ...): each is a balanced sample, so a policy can be trained on any of them.
Results (`--results`, an experiment folder, optional):
  summary.csv   share TOXIC per condition
  sources.yaml  which data the results came from
and prints permutation tests (answers shuffled between conditions within each comment): whether profiles differ from
no profile and from each other, and whether weightings differ within a profile.
Plot a run with scripts/plots/plot_profiles.py --data <data> --out <results>.

Usage:
    python scripts/experiments/profile_signal.py --data data/processed/consensus_n125_train/profiles_probe_20 \
        --results data/experiments/02_profile_signal/probe_20_comments
    python scripts/experiments/profile_signal.py --all --weightings as-is reversed \
        --data data/processed/consensus_n125_train/profiles --results data/experiments/02_profile_signal/train_125_comments
    python scripts/experiments/profile_signal.py --all --single-values \
        --data data/processed/consensus_n125_train/single_values --results data/experiments/03_single_values
    python scripts/experiments/profile_signal.py --ids 5705144 --data /tmp/one_comment          # smoke test, no results
    python scripts/experiments/profile_signal.py --all --single-values --model vllm:Qwen/Qwen3-8B --num-jobs 2 --job 0 \
        --sample data/processed/consensus_n1000_train/sample.csv --data data/processed/consensus_n1000_train/single_values_qwen3_8b
    python scripts/experiments/profile_signal.py --merge --sample data/processed/consensus_n1000_train/sample.csv \
        --data data/processed/consensus_n1000_train/single_values_qwen3_8b
    python scripts/experiments/profile_signal.py --analyze-only --data data/processed/consensus_n125_train/single_values \
        --results data/experiments/03_single_values
"""
import argparse
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ethos.judges.llm_judge import SCHEMA, load_profile
from ethos.llm import connect
from ethos.llm.dmp import format_profile, load_values
from ethos.paths import ROOT, TRAIN_SAMPLE
from ethos.scores import score_table, toxicity_score
from ethos.sources import write_sources

RANDOM_DMP = ROOT / "data/judgements/consensus_n125_train/judgements.csv"
BASE_PROMPT = ROOT / "prompts/judge_minimal.txt"
DMP_PROMPT = ROOT / "prompts/judge_minimal_pluralistic.txt"
WEIGHTINGS = ["as-is", "equal", "reversed"]


# --- setup -----------------------------------------------------------------------------------------------------

def uncertain_comments(n):
    """The n train comments whose random-DMP answers were most mixed (closest to a 50/50 split)."""
    j = pd.read_csv(RANDOM_DMP)
    score = j[j.judge == "gpt-6-luna-dmp-random"].set_index("id").score
    return (score - 0.5).abs().sort_values().index[:n].tolist()


def conditions(taxonomy, weightings=WEIGHTINGS, single_values=False):
    """{condition name: profile text for the prompt}; "none" (empty text) = no profile."""
    out = {"none": ""}
    if single_values:
        return out | {f"{v}|as-is": format_profile([(v, 1.0)]) for v in taxonomy}
    for path in sorted((ROOT / "prompts/profiles").glob("*.yaml")):
        profile = load_profile(path.relative_to(ROOT), taxonomy)
        values, weights = [v for v, _ in profile], [w for _, w in profile]
        variants = {"as-is": profile, "equal": [(v, 1 / len(values)) for v in values],
                    "reversed": list(zip(values, weights[::-1]))}
        for w in weightings:
            out[f"{path.stem}|{w}"] = format_profile(variants[w])
    return out


def comment_order(sample, ids, seed=0):
    """`ids` in a fixed random order that interleaves the consensus bands (one per band in turn), so any first k
    comments are a balanced sample."""
    s = sample.set_index("id").loc[ids]
    bands = [b.sample(frac=1, random_state=seed).index.tolist() for _, b in s.groupby("consensus_bin", sort=True)]
    return [i for group in zip(*[b + [None] * (max(map(len, bands)) - len(b)) for b in bands]) for i in group
            if i is not None]


# --- collecting answers ----------------------------------------------------------------------------------------

def collect(path, conds, texts, samples, model, workers, reasoning_effort="none", flush_every=500):
    """Ask the model for every (comment, condition, answer number) not yet in `path`, appending the answers in batches
    of `flush_every`. Failed calls are skipped and retried on the next run. Prints the speed (answers per second)."""
    done = set()
    if path.exists():
        r = pd.read_csv(path)
        done = set(zip(r.id, r.condition, r.rating))
    jobs = [(cid, c, i) for cid in texts.index for c in conds for i in range(samples) if (cid, c, i) not in done]
    print(f"{len(conds)} conditions x {len(texts)} comments x {samples} answers: {len(jobs)} to do")

    llm, client, name = connect(model)
    base, dmp = BASE_PROMPT.read_text().strip(), DMP_PROMPT.read_text().strip()

    def ask(job):
        cid, c, _ = job
        system = dmp.replace("{profile}", conds[c]) if conds[c] else base
        try:
            return job, llm.chat_json(client, name, system, f"Comment:\n{texts[cid]}", SCHEMA, temperature=1.0,
                                      reasoning_effort=reasoning_effort)
        except Exception as e:
            return job, e

    def flush(rows):
        if rows:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
        return []

    # Keep about 4 x workers requests in flight and send the next in order as each finishes: the server stays busy and
    # answers complete roughly in comment order (so checkpoints fill up one after another).
    rows, failed, start, todo = [], 0, time.perf_counter(), iter(jobs)
    with ThreadPoolExecutor(workers) as pool, tqdm(total=len(jobs), unit="answer") as bar:
        pending = {pool.submit(ask, j) for j in [j for _, j in zip(range(4 * workers), todo)]}
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                (cid, c, i), answer = future.result()
                bar.update(1)
                if (job := next(todo, None)) is not None:
                    pending.add(pool.submit(ask, job))
                if isinstance(answer, Exception):
                    failed += 1
                    continue
                rows.append({"id": cid, "condition": c, "rating": i, "profile": conds[c],
                             "toxic": answer["evaluation"] == "TOXIC", "rationale": answer["rationale"]})
                if len(rows) >= flush_every:
                    rows = flush(rows)
    flush(rows)
    seconds = time.perf_counter() - start
    if jobs:
        print(f"{len(jobs) - failed:,} answers in {seconds:.0f} s = {(len(jobs) - failed) / seconds:.1f} answers/s "
              f"({model}, {workers} workers){f'; {failed} failed' if failed else ''}")
    return failed


# --- merging jobs and checkpoints ----------------------------------------------------------------------------

def merge(data, sample_path, every):
    """Combine <data>/jobs/*.csv into ratings.csv and scores.csv (complete comments only) and write
    checkpoints/n<k>/scores.csv for every k (every, 2 x every, ..., all) whose first k comments in order.csv are complete.
    Safe to run while jobs are still writing."""
    parts = [pd.read_csv(f, on_bad_lines="skip") for f in sorted((data / "jobs").glob("job_*.csv"))]
    r = pd.concat(parts).dropna(subset=["id", "condition", "rating"]).drop_duplicates(["id", "condition", "rating"])
    r["toxic"] = r.toxic.astype(str).str.lower() == "true"
    r.to_csv(data / "ratings.csv", index=False)
    order = pd.read_csv(data / "order.csv").id.tolist()
    expected = r.condition.nunique() * r.rating.nunique()
    complete = set(r.groupby("id").size().loc[lambda n: n == expected].index)
    sample = pd.read_csv(sample_path)
    score_table(r[r.id.isin(complete)], sample).to_csv(data / "scores.csv", index=False)

    ks = list(range(every, len(order), every)) + [len(order)]
    log = []
    for k in ks:
        ready = set(order[:k]) <= complete
        if ready:
            out = data / "checkpoints" / f"n{k:04d}"
            out.mkdir(parents=True, exist_ok=True)
            score_table(r[r.id.isin(order[:k])], sample).to_csv(out / "scores.csv", index=False)
        log.append({"comments": k, "complete": ready})
    pd.DataFrame(log).to_csv(data / "checkpoints.csv", index=False)
    done = [x["comments"] for x in log if x["complete"]]
    print(f"{len(r):,} answers, {len(complete)} of {len(order)} comments complete; "
          f"checkpoints written: {done if done else 'none yet'}")


# --- analysis --------------------------------------------------------------------------------------------------

def permutation_p(r, conds, rng, n_perm=2000):
    """p-value for "the conditions' TOXIC shares differ": the variance of their within-comment-centred shares,
    against shuffles of the condition labels within each comment."""
    x = r[r.condition.isin(conds)].copy()
    x["c"] = x.toxic - x.groupby("id").toxic.transform("mean")
    stat = lambda d: d.groupby("condition").c.mean().var()
    observed, null = stat(x), []
    for _ in range(n_perm):
        x["condition"] = x.groupby("id").condition.transform(lambda s: rng.permutation(s.values))
        null.append(stat(x))
    return float(np.mean(np.array(null) >= observed))


def analyze(data, results=None, sample_path=TRAIN_SAMPLE / "sample.csv"):
    """Write <data>/scores.csv; with `results`, also <results>/summary.csv and sources.yaml and print the permutation
    tests (slow on large runs, so a data-only run, without `results`, skips them)."""
    r = pd.read_csv(data / "ratings.csv")
    r["toxic"] = r.toxic.astype(bool)
    score_table(r, pd.read_csv(sample_path)).to_csv(data / "scores.csv", index=False)
    table = toxicity_score(r, by="condition").rename("share_toxic").to_frame()
    table["shift_vs_none"] = table.share_toxic - table.share_toxic["none"]
    if results:
        results.mkdir(parents=True, exist_ok=True)
        table.to_csv(results / "summary.csv")
        write_sources(results, [data / "ratings.csv", sample_path])
    pd.set_option("display.width", 200)
    print(f"\n{r.id.nunique()} comments, {len(r)} answers. Share TOXIC per condition (none = no profile):")
    print(table.sort_values("share_toxic").round(3).to_string())

    if not results:
        return
    rng = np.random.default_rng(0)
    names = sorted({c.split("|")[0] for c in r.condition if c != "none"})
    as_written = [f"{q}|as-is" for q in names]
    print("\nPermutation tests (p = share of within-comment shuffles at least as extreme):")
    print(f"  profiles and no profile differ:   p = {permutation_p(r, ['none'] + as_written, rng):.3f}")
    print(f"  profiles differ from each other:  p = {permutation_p(r, as_written, rng):.3f}")
    for q in names:
        variants = [f"{q}|{w}" for w in WEIGHTINGS if f"{q}|{w}" in set(r.condition)]
        if len(variants) > 1:
            print(f"  weightings differ within {q}: p = {permutation_p(r, variants, rng, n_perm=1000):.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True,
                   help="where the answers go, e.g. data/processed/consensus_n125_train/single_values")
    p.add_argument("--results", type=Path, help="experiment folder for summary.csv, e.g. data/experiments/03_single_values")
    p.add_argument("--model", default="gpt-6-luna",
                   help="gpt-* (OpenAI API), vllm:<hf id> (vLLM server, e.g. vllm:Qwen/Qwen3-8B) or an Ollama model")
    p.add_argument("--sample", type=Path, default=TRAIN_SAMPLE / "sample.csv", help="sample.csv with the comments")
    p.add_argument("--n-comments", type=int, default=20, help="the most uncertain comments (default)")
    p.add_argument("--all", action="store_true", help="every comment in --sample")
    p.add_argument("--ids", nargs="+", type=int, help="specific comment ids")
    p.add_argument("--weightings", nargs="+", default=WEIGHTINGS, choices=WEIGHTINGS)
    p.add_argument("--single-values", action="store_true", help="each taxonomy value alone instead of the profiles")
    p.add_argument("--samples", type=int, default=10, help="answers per comment and condition")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--reasoning-effort", default="none",
                   help="none (thinking off; Qwen3, gpt-6-luna) or low / medium / high (models like gpt-oss that always reason)")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--job", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-jobs", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    p.add_argument("--merge", action="store_true", help="combine jobs into ratings.csv, scores.csv, checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=100, help="comments per checkpoint (with --merge)")
    args = p.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    if args.merge:
        return merge(args.data, args.sample, args.checkpoint_every)

    if not args.analyze_only:
        sample = pd.read_csv(args.sample)
        texts = sample.set_index("id").comment_text
        ids = args.ids or (texts.index.tolist() if args.all else uncertain_comments(args.n_comments))
        conds = conditions(load_values("prompts/taxonomy-values.txt"), args.weightings, args.single_values)
        path = args.data / "ratings.csv"
        if args.num_jobs > 1:                      # this job's share of a fixed, band-balanced order
            order = comment_order(sample, ids)
            tmp = args.data / f"order.{args.job}.tmp"
            pd.DataFrame({"id": order}).to_csv(tmp, index=False)
            tmp.replace(args.data / "order.csv")     # every job writes the same order
            ids = order[args.job::args.num_jobs]
            path = args.data / "jobs" / f"job_{args.job}of{args.num_jobs}.csv"
            path.parent.mkdir(exist_ok=True)
            print(f"job {args.job} of {args.num_jobs}: {len(ids)} comments")
        for attempt in range(3):                     # retry failed calls (up to 3 passes)
            if not collect(path, conds, texts.loc[ids], args.samples, args.model, args.workers, args.reasoning_effort):
                break
        if args.num_jobs > 1:
            return print(f"job done; combine with: --merge --sample {args.sample} --data {args.data}")
    analyze(args.data, args.results, args.sample)


if __name__ == "__main__":
    main()
