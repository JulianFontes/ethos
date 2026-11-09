"""Plot the moral-profile results of scripts/experiments/profile_signal.py.

Writes next to the results (<results>/):
  profile_shift_weighting.png   toxicity score shift vs. no profile: (A) by profile, (B) by weighting
  profile_distance.png          absolute difference from the human score, for no profile and each profile

Usage:
    python scripts/plot_profiles.py                                              # data/experiments/profile_signal/gpt
    python scripts/plot_profiles.py --results data/experiments/profile_signal/probing
"""
import argparse
from pathlib import Path

import pandas as pd

from ethos.plotting.profiles import plot_profile_distance, plot_profile_shift_weighting

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=ROOT / "data/experiments/profile_signal/gpt")
    p.add_argument("--sample", type=Path, default=ROOT / "data/processed/consensus_n125_train/sample.csv",
                   help="sample.csv with the comments' human toxicity scores")
    p.add_argument("--model", default="gpt-6-luna", help="model name for the subtitle")
    args = p.parse_args()

    ratings = pd.read_csv(args.results / "ratings.csv")
    human = pd.read_csv(args.sample).set_index("id").toxicity
    n_comments = ratings.id.nunique()
    n_answers = int(ratings.groupby(["id", "condition"]).size().max())
    scope = f"{args.model}, {n_comments} comments, {n_answers} answers each"

    if ratings.condition.str.endswith(("|equal", "|reversed")).any():
        plot_profile_shift_weighting(ratings, args.results / "profile_shift_weighting.png", scope)
        print(f"wrote {args.results / 'profile_shift_weighting.png'}")
    plot_profile_distance(ratings, human, args.results / "profile_distance.png", scope)
    print(f"wrote {args.results / 'profile_distance.png'}")


if __name__ == "__main__":
    main()
