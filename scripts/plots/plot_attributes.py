"""Plot each comment's toxicity score against how many annotators selected each attribute (raw, one dot per comment).

Reads <data_dir>/comments.csv (built by scripts/data/extract_splits.py) and writes to
data/experiments/civil-comments/dataset/attributes/ (plus sources.yaml), for each split:
  attributes_vs_toxicity_<split>.png              x = toxicity score
  attributes_vs_toxicity_consensus_<split>.png    the same, plus a row with x = consensus
  attributes_by_bin_<split>.png                    average per consensus bin (top) and toxicity bin
  attributes_by_bin_<split>.csv                    the numbers behind it
and, from the individual annotator votes (train only, <data_dir>/train_annotations.csv):
  attribute_votes_by_consensus_train.png           votes per consensus bin: said toxic / not toxic,
                                                                      split by whether the attribute was selected
  attribute_votes_by_consensus_train.csv           the counts behind it

Usage:
    python scripts/plots/plot_attributes.py                     # train and test together
    python scripts/plots/plot_attributes.py --splits train test all
"""
import argparse

import pandas as pd

from ethos.paths import EXPERIMENTS, ROOT
from ethos.plotting.attributes import (
    ATTRIBUTES,
    attribute_votes_by_consensus,
    attributes_by_bin,
    plot_attribute_votes_by_consensus,
    plot_attributes_by_bin,
    plot_attributes_vs_toxicity,
)
from ethos.sources import write_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed/civil-comments/comments")
    parser.add_argument("--out", default=EXPERIMENTS / "dataset" / "attributes")
    parser.add_argument("--splits", nargs="+", default=["all"], help="train, test and/or all (both together)")
    args = parser.parse_args()

    data_dir, out = ROOT / args.data_dir, ROOT / args.out
    comments = pd.read_csv(data_dir / "comments.csv", usecols=["split", "toxicity", "consensus", *ATTRIBUTES])
    for split in args.splits:
        df = comments if split == "all" else comments[comments.split == split]
        scope = "top-level comments with ≥30 annotators" + (", train + test" if split == "all" else f", {split}")
        for name, x_cols in (("attributes_vs_toxicity", ("toxicity",)),
                             ("attributes_vs_toxicity_consensus", ("toxicity", "consensus"))):
            path = out / f"{name}_{split}.png"
            plot_attributes_vs_toxicity(df, path, scope, x_cols)
            print(f"wrote {path.relative_to(ROOT)}")
        path = out / f"attributes_by_bin_{split}.png"
        plot_attributes_by_bin(df, path, scope)
        by_consensus, by_toxicity = attributes_by_bin(df)
        pd.concat({"consensus": by_consensus.reset_index().rename(columns={"cbin": "bin"}),
                   "toxicity": by_toxicity.reset_index().rename(columns={"tbin": "bin"})},
                  names=["x_axis"]).reset_index(level=0).to_csv(path.with_suffix(".csv"), index=False)
        print(f"wrote {path.relative_to(ROOT)} and .csv")

    votes_path = data_dir / "train_annotations.csv"
    if votes_path.exists():
        consensus = pd.read_csv(data_dir / "comments.csv", usecols=["id", "consensus"]).set_index("id").consensus
        votes = pd.read_csv(votes_path, usecols=["id", "toxic", *ATTRIBUTES])
        counts = attribute_votes_by_consensus(votes, consensus)
        path = out / "attribute_votes_by_consensus_train.png"
        counts.to_csv(path.with_suffix(".csv"), index=False)
        plot_attribute_votes_by_consensus(counts, path)
        print(f"wrote {path.relative_to(ROOT)} and .csv")
    write_sources(out, [data_dir / "comments.csv"] + ([votes_path] if votes_path.exists() else []))


if __name__ == "__main__":
    main()
