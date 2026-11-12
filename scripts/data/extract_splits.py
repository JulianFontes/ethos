"""Build one dataset of Civil Comments train and test comments, filtered the same way, with a split label.

Keeps Jigsaw's own train/test split (Detoxify was trained on train, so test stays held out), then for each split:
top-level (parent) comments only, >= min_annotators toxicity annotators, and deduplicated by stripped text.
Optionally drops test comments whose text also appears in train. The logic lives in ethos.data.

Writes to <output_dir>/:
  comments.csv           comment-level rows for both splits: all_data.csv columns (its `split` column says
                         train or test), plus consensus = max(toxicity, 1 - toxicity)
  train_annotations.csv  annotator-level toxicity votes for train (Jigsaw only released these for train)
  config.yaml            the config used

Usage:
    python scripts/data/extract_splits.py
    python scripts/data/extract_splits.py --config configs/data/my_splits.yaml
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from ethos.data import build_splits, load_comments, load_votes
from ethos.paths import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "data" / "splits.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_path = parser.parse_args().config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train, test, n_dropped = build_splits(load_comments(), cfg["min_annotators"], cfg.get("top_level_only", True),
                                          cfg.get("drop_test_overlap", True))
    print(f"dropped {n_dropped} test comments whose text is also in train")
    votes = load_votes()
    votes = votes[votes.id.isin(train.id)]

    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([train, test]).to_csv(out / "comments.csv", index=False)
    votes.to_csv(out / "train_annotations.csv", index=False)
    shutil.copy(config_path, out / "config.yaml")

    for name, d in (("train", train), ("test", test)):
        print(f"{name}: {len(d)} comments, annotators median {d.toxicity_annotator_count.median():.0f}, "
              f"toxicity >= 0.5: {(d.toxicity >= 0.5).mean():.1%}")
    print(f"train_annotations: {len(votes)} votes from {votes.worker.nunique()} annotators")
    print(f"wrote to {out}")


if __name__ == "__main__":
    main()
