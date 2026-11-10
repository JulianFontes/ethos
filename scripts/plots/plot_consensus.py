"""Plot annotator consensus vs toxicity score for the datasets built by scripts/data/extract_splits.py.

Writes data/experiments/dataset/consensus/consensus_vs_toxicity_<split>.png and consensus_distribution_<split>.png
for each split, plus sources.yaml.

Usage:
    python scripts/plots/plot_consensus.py                      # train and test
    python scripts/plots/plot_consensus.py --splits test
    python scripts/plots/plot_consensus.py --splits all         # train and test together
    python scripts/plots/plot_consensus.py --data-dir data/processed/comments
"""
import argparse

import pandas as pd

from ethos.paths import EXPERIMENTS, ROOT
from ethos.plotting.consensus import plot_consensus_distribution, plot_consensus_vs_toxicity
from ethos.sources import write_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed/comments")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--out", default=EXPERIMENTS / "dataset" / "consensus")
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
            path = ROOT / args.out / f"{name}_{split}.png"
            plot(df.toxicity, path, label, args.low)
            print(f"wrote {path.relative_to(ROOT)}")
    write_sources(ROOT / args.out, [data_dir / "comments.csv"])


if __name__ == "__main__":
    main()
