"""Plot a profile_signal.py run: reads <data>/ratings.csv, writes to <out>/figures/ plus <out>/sources.yaml:

  toxic_change_count.png        change in TOXIC answers vs. no profile (counts)
  toxic_change_pp.png           the same in percentage points
  distance_by_consensus.png     average distance from the human score per consensus bin, one dot per profile / value
  profile_shift_weighting.png   score shift vs. no profile by profile and weighting (runs with several weightings)
  profile_distance.png          per-comment distance from the human score (runs with <= 15 conditions)

Usage:
    python scripts/plots/plot_profiles.py \\
        --data data/processed/civil-comments/consensus_n125_train/single_values --out data/experiments/civil-comments/03_single_values
    python scripts/plots/plot_profiles.py \\
        --data data/processed/civil-comments/consensus_n125_train/profiles --out data/experiments/civil-comments/02_profile_signal/train_125_comments
"""
import argparse
from pathlib import Path

import pandas as pd

from ethos.paths import ROOT, TRAIN_SAMPLE
from ethos.plotting.profiles import (
    plot_distance_by_consensus,
    plot_profile_distance,
    plot_profile_shift_weighting,
    plot_toxic_change,
)
from ethos.sources import write_sources


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="folder with ratings.csv (in data/processed/...)")
    p.add_argument("--out", type=Path, required=True, help="experiment folder (in data/experiments/...)")
    p.add_argument("--sample", type=Path, default=TRAIN_SAMPLE / "sample.csv",
                   help="sample.csv with the comments' human toxicity scores and consensus bins")
    p.add_argument("--model", default="gpt-6-luna", help="model name for titles")
    args = p.parse_args()
    data, out = ROOT / args.data, ROOT / args.out

    ratings = pd.read_csv(data / "ratings.csv")
    sample = pd.read_csv(args.sample).set_index("id")
    n_answers = int(ratings.groupby(["id", "condition"]).size().max())
    scope = f"{args.model}, {ratings.id.nunique()} comments, {n_answers} answers each"
    fig = out / "figures"

    plot_toxic_change(ratings, fig / "toxic_change_count.png", scope)
    plot_toxic_change(ratings, fig / "toxic_change_pp.png", scope, percent=True)
    plot_distance_by_consensus(ratings, sample.toxicity, sample.consensus_bin, fig / "distance_by_consensus.png",
                               model=args.model)
    if ratings.condition.str.endswith(("|equal", "|reversed")).any():
        plot_profile_shift_weighting(ratings, fig / "profile_shift_weighting.png", scope)
    if ratings.condition.nunique() <= 15:
        plot_profile_distance(ratings, sample.toxicity, fig / "profile_distance.png", scope)
    write_sources(out, [data / "ratings.csv", args.sample])
    print(f"wrote {sorted(f.name for f in fig.glob('*.png'))} and sources.yaml to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
