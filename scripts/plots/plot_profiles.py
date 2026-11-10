"""Plot a run of scripts/experiments/profile_signal.py into <results>/figures/:

  toxic_change_count.png        change in TOXIC answers vs. no profile (counts)
  toxic_change_pp.png           the same in percentage points
  distance_by_consensus.png     average distance from the human score per consensus bin, one dot per profile / value
  profile_shift_weighting.png   score shift vs. no profile by profile and weighting (runs with several weightings)
  profile_distance.png          per-comment distance from the human score (runs with <= 15 conditions)

Usage:
    python scripts/plots/plot_profiles.py                     # data/experiments/02_profile_signal/train_125_comments
    python scripts/plots/plot_profiles.py --results data/experiments/03_single_values
"""
import argparse
from pathlib import Path

import pandas as pd

from ethos.paths import EXPERIMENTS, TRAIN_SAMPLE
from ethos.plotting.profiles import (
    plot_distance_by_consensus,
    plot_profile_distance,
    plot_profile_shift_weighting,
    plot_toxic_change,
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=EXPERIMENTS / "02_profile_signal/train_125_comments")
    p.add_argument("--sample", type=Path, default=TRAIN_SAMPLE / "sample.csv",
                   help="sample.csv with the comments' human toxicity scores and consensus bins")
    p.add_argument("--model", default="gpt-6-luna", help="model name for titles")
    args = p.parse_args()

    ratings = pd.read_csv(args.results / "ratings.csv")
    sample = pd.read_csv(args.sample).set_index("id")
    n_answers = int(ratings.groupby(["id", "condition"]).size().max())
    scope = f"{args.model}, {ratings.id.nunique()} comments, {n_answers} answers each"
    out = args.results / "figures"

    plot_toxic_change(ratings, out / "toxic_change_count.png", scope)
    plot_toxic_change(ratings, out / "toxic_change_pp.png", scope, percent=True)
    plot_distance_by_consensus(ratings, sample.toxicity, sample.consensus_bin, out / "distance_by_consensus.png",
                               model=args.model)
    if ratings.condition.str.endswith(("|equal", "|reversed")).any():
        plot_profile_shift_weighting(ratings, out / "profile_shift_weighting.png", scope)
    if ratings.condition.nunique() <= 15:
        plot_profile_distance(ratings, sample.toxicity, out / "profile_distance.png", scope)
    print(f"wrote {sorted(f.name for f in out.glob('*.png'))} to {out}")


if __name__ == "__main__":
    main()
