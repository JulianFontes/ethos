"""Combine activation steering runs on the same comments into one run folder (more steering strengths per comment).

The runs must share the sample, model vectors, value and layers, and must not repeat a setting. Writes to --out:
ratings.csv (all answers), scores.csv (rebuilt from them with ethos.scores.score_table), steering.json (all alphas, and
which runs they came from) and README.md. The source runs are not changed. Rerun with more --runs to add strengths.

Usage:
    python scripts/experiments/combine_steering_runs.py --sample data/processed/civil-comments/boundary_n500_train/sample.csv \\
        --runs <run folder> <run folder> ... --out <new run folder>
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ethos.paths import ROOT
from ethos.scores import score_table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="steering run folders (ratings.csv, steering.json)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    ratings, steerings = [], []
    for run in args.runs:
        ratings.append(pd.read_csv(ROOT / run / "ratings.csv"))
        steerings.append(json.loads((ROOT / run / "steering.json").read_text()))
    for s in steerings[1:]:
        for key in ("vectors", "layers", "values"):
            assert s[key] == steerings[0][key], f"runs differ in {key}"
    conditions = [set(r.condition) for r in ratings]
    assert all(not (a & b) for i, a in enumerate(conditions) for b in conditions[i + 1:]), "a setting appears in two runs"
    r = pd.concat(ratings, ignore_index=True)
    r.to_csv(out / "ratings.csv", index=False)
    scores = score_table(r, pd.read_csv(ROOT / args.sample))
    scores.to_csv(out / "scores.csv", index=False)
    alphas = sorted({a for s in steerings for a in s["alphas"]})
    (out / "steering.json").write_text(json.dumps(steerings[0] | {"alphas": alphas,
                                                                   "combined_from": [str(run) for run in args.runs]}, indent=1))
    lines = "\n".join(f"- `{run}`: alpha {', '.join(f'{a:+g}' if a else '0' for a in s['alphas'])}"
                      for run, s in zip(args.runs, steerings))
    (out / "README.md").write_text(f"# Combined steering runs ({len(alphas)} strengths)\n\nAll answers from these runs on "
                                   f"the same comments ({args.sample}), vectors, value and layers:\n{lines}\n\nratings.csv = "
                                   "their answers; scores.csv rebuilt from them (ethos.scores.score_table); steering.json "
                                   "lists all alphas. Made by scripts/experiments/combine_steering_runs.py; the source runs "
                                   "are unchanged.\n")
    print(scores.groupby("weighting").agg(comments=("id", "nunique"), answers=("n_answers", "min"),
                                          toxic=("llm_toxicity", "mean")).round(3).to_string())
    print(f"wrote {args.out}: {len(alphas)} strengths")


if __name__ == "__main__":
    main()
