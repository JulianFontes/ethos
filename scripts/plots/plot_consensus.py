"""Plot annotator consensus vs toxicity score for the datasets built by scripts/data/extract_splits.py.

Writes <data_dir>/figures/consensus_vs_toxicity_<split>.png and consensus_distribution_<split>.png for each split.

Usage:
    python scripts/plots/plot_consensus.py                      # train and test
    python scripts/plots/plot_consensus.py --splits test
    python scripts/plots/plot_consensus.py --splits all         # train and test together
    python scripts/plots/plot_consensus.py --data-dir data/processed/toplevel_min30
"""
import argparse

import pandas as pd

from ethos.paths import ROOT
from ethos.plotting.consensus import plot_consensus_distribution, plot_consensus_vs_toxicity


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed/toplevel_min30")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--low", type=float, default=0.6, help="consensus at or below this is shaded as low")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    comments = pd.read_csv(data_dir / "comments.csv", usecols=["split", "toxicity"])
    for split in args.splits:
        df = comments if split == "all" else comments[comments.split == split]
        scope = "train + test" if split == "all" else split
        label = f"Top-level {scope} comments with ≥30 raters"
        for name, plot in (("consensus_vs_toxicity", plot_consensus_vs_toxicity),
                           ("consensus_distribution", plot_consensus_distribution)):
            path = data_dir / "figures" / f"{name}_{split}.png"
            plot(df.toxicity, path, label, args.low)
            print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
