"""Multi-value profiles: does adding values similar to an anchor change the LLM's TOXIC / NOT TOXIC answers?

For each anchor value (e.g. Compassion) and each size n (default 3, 4, 5), the conditions are:

  none      no profile (baseline)
  alone     the anchor by itself (weight 1)
  similar   the anchor + its n-1 closest values from the same similarity-bank cluster
  random    the anchor + n-1 random values from other clusters   (--n-random independent draws)
  distant   the anchor + the n-1 values least similar to it
See src/ethos/multi_profiles.py for how profiles are built (nested sizes, fixed weights, small-cluster fallback).

Each condition is asked `--samples` times per comment with the same minimal prompt and judge as profile_signal.py
(it reuses that script's collector). Writes to <out>/:
  conditions.csv  every condition: anchor, kind, size, profile text, coherence, how many values are outside the cluster
  ratings.csv     one row per answer (re-running skips answers already there)
  summary.csv     share TOXIC per condition and its shift vs "none"
  contrasts.csv   paired comparisons per anchor and size, with 95% bootstrap CIs and sign-flip p-values:
                    similar - alone, similar - random, similar - distant, random - alone, similar - none, alone - none

Cost = (#conditions) x (#comments) x (--samples) answers: check it with --dry-run first.

Usage:
    python scripts/experiments/profile_multi.py --list                      # which anchors can fill which sizes
    python scripts/experiments/profile_multi.py --anchors Compassion Autonomy --dry-run --out /tmp/x
    python scripts/experiments/profile_multi.py --anchors Compassion Autonomy Justice --n-comments 20 \\
        --out data/experiments/04_multi_value/probe
    python scripts/experiments/profile_multi.py --anchors Compassion --scope cluster --all --out data/experiments/04_multi_value/strict
    python scripts/experiments/profile_multi.py --anchors Compassion Autonomy --analyze-only --out data/experiments/04_multi_value/probe
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from profile_signal import collect, uncertain_comments  # sibling script: same prompts, judge and collector
from ethos.multi_profiles import build_conditions, feasibility, name_of
from ethos.paths import TRAIN_SAMPLE
from ethos.value_banks import SimilarityBank

CONTRASTS = [  # (label, left, right): mean per-comment TOXIC share of left minus right
    ("alone - none", "alone", "none"),
    ("similar - none", "similar", "none"),
    ("similar - alone", "similar", "alone"),
    ("similar - random", "similar", "random"),
    ("similar - distant", "similar", "distant"),
    ("random - alone", "random", "alone"),
]


def list_anchors(bank, sizes):
    print(f"{'value':<24}{'cluster':<24}{'size':>5}   fills (cluster only)")
    for v, ok in feasibility(bank, sizes).items():
        cid = bank.cluster_of(v)
        label = next(c["label"] for c in bank.clusters if c["id"] == cid)
        print(f"{v:<24}{label:<24}{len(bank.members(cid)):>5}   " + " ".join(f"n{n}" if f else "--" for n, f in ok.items()))


# --- analysis --------------------------------------------------------------------------------------------------

def paired_stats(diff, rng, n_boot=2000):
    """Mean of per-comment differences, 95% bootstrap CI over comments, sign-flip permutation p-value."""
    d = np.asarray(diff.dropna(), dtype=float)
    if len(d) == 0:
        return dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    flips = (rng.choice([-1.0, 1.0], (n_boot, len(d))) * d).mean(1)
    return dict(n=len(d), mean=d.mean(), lo=np.quantile(boots, 0.025), hi=np.quantile(boots, 0.975),
                p=float(np.mean(np.abs(flips) >= abs(d.mean()) - 1e-12)))


def analyze(path, out):
    r = pd.read_csv(path)
    r["toxic"] = r.toxic.astype(bool)
    meta = pd.read_csv(out / "conditions.csv").set_index("condition")
    share = r.groupby("condition").toxic.mean().rename("share_toxic").to_frame()
    share["shift_vs_none"] = share.share_toxic - share.share_toxic.get("none", np.nan)
    summary = share.join(meta[["anchor", "kind", "size", "n_outside", "coherence", "anchor_similarity", "n_clusters"]])
    summary.to_csv(out / "summary.csv")
    pd.set_option("display.width", 220)
    print(f"\n{r.id.nunique()} comments, {len(r)} answers. Share TOXIC (none = no profile, "
          f"{share.share_toxic.get('none', float('nan')):.3f}):")
    show = summary.drop(index="none", errors="ignore").sort_values(["anchor", "size", "kind"])
    print(show[["share_toxic", "shift_vs_none", "coherence", "n_outside"]].round(3).to_string())

    cs = r.groupby(["id", "condition"]).toxic.mean().unstack()  # per-comment TOXIC share, comments x conditions
    rng = np.random.default_rng(0)
    rows = []

    def col(anchor, kind, size):
        if kind == "none":
            return cs["none"] if "none" in cs else None
        names = [c for c in cs.columns if c in meta.index and meta.at[c, "anchor"] == anchor
                 and meta.at[c, "kind"] == kind and meta.at[c, "size"] == (1 if kind == "alone" else size)]
        return cs[names].mean(axis=1) if names else None  # random: average over its independent draws

    for anchor in sorted(meta.anchor.unique()):
        sizes = sorted(s for s in meta[meta.anchor == anchor]["size"].unique() if s > 1)
        for size in sizes:
            for label, left, right in CONTRASTS:
                if label == "alone - none" and size != sizes[0]:
                    continue  # does not depend on size: report once per anchor
                a, b = col(anchor, left, size), col(anchor, right, size)
                if a is not None and b is not None:
                    rows.append({"anchor": anchor, "size": size, "contrast": label} | paired_stats(a - b, rng))
    c = pd.DataFrame(rows)
    c.to_csv(out / "contrasts.csv", index=False)
    if c.empty:
        print("\nNo contrasts yet (a condition has no answers).")
        return
    print("\nPaired contrasts (mean difference in TOXIC share across comments; CI = 95% bootstrap over comments;")
    print("p = sign-flip test, UNCORRECTED for the number of anchors x sizes x contrasts):")
    print(c.round(3).to_string(index=False))
    if c.anchor.nunique() > 1:
        print("\nAcross anchors (how consistent is each contrast?):")
        g = c.groupby(["contrast", "size"])["mean"].agg(mean="mean", n_anchors="count",
                                                      n_positive=lambda x: int((x > 0).sum()))
        print(g.round(3).to_string())


# --- main ------------------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, help="run folder, e.g. data/experiments/04_multi_value/probe")
    p.add_argument("--anchors", nargs="+", help="anchor values (names as in prompts/taxonomy-values.txt)")
    p.add_argument("--sizes", nargs="+", type=int, default=[3, 4, 5], help="profile sizes incl. the anchor")
    p.add_argument("--n-random", type=int, default=3, help="independent random-value draws per size")
    p.add_argument("--scope", default="tree", choices=["tree", "cluster"],
                   help="cluster: skip sizes the anchor's cluster cannot fill; tree: fill from the nearest bigger group")
    p.add_argument("--weighting", default="anchor", choices=["anchor", "equal"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="gpt-6-luna", help="gpt-* through the OpenAI API, anything else through Ollama")
    p.add_argument("--sample", type=Path, default=TRAIN_SAMPLE / "sample.csv")
    p.add_argument("--n-comments", type=int, default=20, help="the most uncertain comments (default)")
    p.add_argument("--all", action="store_true", help="every comment in --sample")
    p.add_argument("--ids", nargs="+", type=int)
    p.add_argument("--samples", type=int, default=10, help="answers per comment and condition")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--list", action="store_true", help="show which anchors can fill which sizes, then stop")
    p.add_argument("--dry-run", action="store_true", help="build the conditions and count the answers, ask nothing")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()

    bank = SimilarityBank.load()
    if args.list:
        return list_anchors(bank, args.sizes)
    if not args.anchors or not args.out:
        p.error("--anchors and --out are required (use --list to see the options)")

    meta, skipped = build_conditions(bank, args.anchors, args.sizes, args.n_random, args.scope, args.weighting, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{"condition": k, **{f: e[f] for f in ("anchor", "kind", "size", "draw", "scope", "n_outside",
                           "coherence", "anchor_similarity", "n_clusters", "text")}} for k, e in meta.items()])
    table.to_csv(args.out / "conditions.csv", index=False)

    print(f"{len(args.anchors)} anchors x sizes {args.sizes}: {len(meta)} profile conditions (+ none)")
    if skipped:
        print("skipped (cluster too small, --scope cluster): " + ", ".join(f"{a} n{n}" for a, n in skipped))
    outside = table[(table.kind == "similar") & (table.n_outside > 0)]
    if len(outside):
        print("similar profiles that reach outside the cluster (report separately): "
              + ", ".join(f"{r.anchor} n{r.size} (+{r.n_outside})" for r in outside.itertuples()))

    if not args.analyze_only:
        texts = pd.read_csv(args.sample).set_index("id").comment_text
        ids = args.ids or (texts.index.tolist() if args.all else uncertain_comments(args.n_comments))
        conds = {"none": ""} | {k: e["text"] for k, e in meta.items()}
        n = len(conds) * len(ids) * args.samples
        if args.dry_run:
            print(f"\n{len(conds)} conditions x {len(ids)} comments x {args.samples} answers = {n} model calls")
            for k in list(conds)[:12]:
                print(f"  {k:<34}{conds[k] or '(no profile)'}")
            return
        collect(args.out / "ratings.csv", conds, texts.loc[ids], args.samples, args.model, args.workers)
    analyze(args.out / "ratings.csv", args.out)


if __name__ == "__main__":
    main()
