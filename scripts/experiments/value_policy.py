"""Value policy with a decision tree: for each comment, pick the one moral value (of 60), or no value at all (the
61st option, "none"), whose LLM answers land closest to the human toxicity score.

The tree learns to predict distance = |LLM score - human score| from the comment's embedding and the value (one-hot);
the policy picks the value with the lowest predicted distance. Evaluated with cross-validation on the train sample
(split by comment, balanced by consensus band, 5 folds x 5 repeats) at each depth, against the baselines.

Data: a single-value scores.csv (--scores; default data/processed/consensus_n125_train/single_values/scores.csv, or a
checkpoint such as .../single_values_qwen3_8b/checkpoints/n0200/scores.csv) and the comment embeddings.
Results (--out, default data/experiments/04_value_policy/decision_tree/):
  cv_results.csv    one row per round and depth: train/val MSE, policy distance, tie share, leaves
  baselines.csv     one row per round and split: no value, random, best fixed value, oracle
  summary.csv       mean and std per depth, validation
  depth_curves.png  loss and policy distance by depth
  sources.yaml      which data the results came from
cv_results.csv, summary.csv and depth_curves.png are rewritten after every round, so they can be watched while it runs.
On the cluster, scripts/shell/submit_value_policy.sh runs it as a CPU job with the log in <out>/logs/.

Usage:
    python scripts/experiments/value_policy.py
    python scripts/experiments/value_policy.py --depths 1 2 3 --repeats 1      # quick run
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ethos.data import feature_matrix, load_embeddings, load_scores
from ethos.paths import EMBEDDINGS, EXPERIMENTS, ROOT, SINGLE_VALUES
from ethos.plotting.policy import plot_depth_curves
from ethos.policy import N_FOLDS, baselines, best_fixed_value, cv_splits, run_tree_cv
from ethos.sources import write_sources


logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # missing-font warnings on every redraw


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", type=Path, default=SINGLE_VALUES / "scores.csv", help="single-value scores.csv to train on")
    p.add_argument("--depths", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--repeats", type=int, default=5, help="cross-validation repeats (5 folds each)")
    p.add_argument("--out", type=Path, default=EXPERIMENTS / "04_value_policy" / "decision_tree")
    p.add_argument("--choose-none", action=argparse.BooleanOptionalAction, default=True,
                   help="let the policy also choose no value (61 options); --no-choose-none: only the 60 values")
    args = p.parse_args()
    out = ROOT / args.out

    print("Loading data ...")
    rows, base = load_scores(args.scores, include_none=args.choose_none)
    comment = load_embeddings()                      # all comments; feature_matrix picks the rows by id
    X = feature_matrix(rows, comment)
    print(f"  {rows.id.nunique()} comments x {rows.moral_value.nunique()} options"
          f"{' (60 values + none)' if args.choose_none else ' (values)'} = {len(rows):,} rows, "
          f"{X.shape[1]} features ({comment.shape[1]} embedding + "
          f"{sum(c.startswith('value_') for c in X)} value)")
    splits = list(cv_splits(rows, n_repeats=args.repeats))

    print("Baselines ...")
    base_recs = []
    for repeat, fold, tr, va in splits:
        v = best_fixed_value(rows, tr)
        for split, idx in (("train", tr), ("val", va)):
            base_recs.append({"repeat": repeat, "fold": fold, "split": split, "fixed_value_name": v,
                              **baselines(rows, base, idx, v)})
    base_df = pd.DataFrame(base_recs)
    val = base_df[base_df.split == "val"]
    print("  validation: " + ", ".join(f"{k} {val[k].mean():.3f}" for k in ["no_value", "random", "fixed_value", "oracle"])
          + f"  (fixed value: {val.fixed_value_name.mode()[0]})")

    out.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(out / "baselines.csv", index=False)
    write_sources(out, [ROOT / args.scores, EMBEDDINGS / "embeddings.parquet"])
    title = "Decision Tree Value Policy by Tree Depth"

    def save(results):
        """Write the results so far (called after every round, and at the end)."""
        n = results.groupby(["repeat", "fold"]).ngroups
        results.to_csv(out / "cv_results.csv", index=False)
        cols = ["train_mse", "val_mse", "train_policy", "val_policy", "val_tied", "leaves"]
        results.groupby("depth")[cols].agg(["mean", "std"]).round(4).to_csv(out / "summary.csv")
        plot_depth_curves(results, base_df, out / "depth_curves.png",
                          title if n == len(splits) else f"{title} (in progress: {n} of {len(splits)} rounds)")

    print(f"Decision tree: {len(splits)} rounds ({N_FOLDS} folds x {args.repeats} repeats) x {len(args.depths)} depths")
    with tqdm(total=len(splits), unit="round") as bar:
        results = run_tree_cv(rows, X, args.depths, splits, progress=bar, on_round=save)
    save(results)

    pd.set_option("display.width", 200)
    print("\nBy depth (mean over rounds; policy = real distance of the chosen values, lower is better):")
    show = results.groupby("depth")[["train_mse", "val_mse", "train_policy", "val_policy", "val_tied", "leaves"]].mean()
    print(show.round(3).to_string())
    best = show.val_policy.idxmin()
    print(f"\nBest depth on validation: {best} -> {show.val_policy[best]:.3f} "
          f"(best fixed value {val.fixed_value.mean():.3f}, no value {val.no_value.mean():.3f})")

    print(f"wrote cv_results.csv, baselines.csv, summary.csv, depth_curves.png, sources.yaml to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
