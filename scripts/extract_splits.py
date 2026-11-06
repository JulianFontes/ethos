"""Build one dataset of Civil Comments train and test comments, filtered the same way, with a split label.

Keeps Jigsaw's own train/test split (Detoxify was trained on train, so test stays held out),
then for each split: top-level (parent) comments only, >= min_annotators toxicity raters, and
deduplicated by stripped text. Optionally drops test comments whose text also appears in train.

Writes to <output_dir>/:
  comments.csv           comment-level rows for both splits: all_data.csv columns (its `split` column says
                         train or test), plus consensus = max(toxicity, 1 - toxicity)
  train_annotations.csv  rater-level toxicity votes for train (Jigsaw only released these for train)
  config.yaml            the config used

Usage:
    python scripts/extract_splits.py
    python scripts/extract_splits.py --config configs/my_splits.yaml
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "unprocessed"
DEFAULT_CONFIG = ROOT / "configs" / "splits.yaml"


def filter_split(df, split, cfg):
    d = df[(df.split == split) & (df.toxicity_annotator_count >= cfg["min_annotators"])]
    if cfg.get("top_level_only", True):
        d = d[d.parent_id.isna()]
    return d.assign(_text=d.comment_text.str.strip()).drop_duplicates("_text")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_path = parser.parse_args().config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(RAW / "all_data.csv")
    train, test = filter_split(df, "train", cfg), filter_split(df, "test", cfg)
    if cfg.get("drop_test_overlap", True):
        overlap = test._text.isin(train._text)
        print(f"dropping {overlap.sum()} test comments whose text is also in train")
        test = test[~overlap]
    train, test = train.drop(columns="_text"), test.drop(columns="_text")
    # consensus = share of raters in the majority: 0.5 = even split, 1 = unanimous
    train, test = (d.assign(consensus=d.toxicity.clip(lower=1 - d.toxicity)) for d in (train, test))

    votes = pd.read_csv(RAW / "annotations" / "toxicity_individual_annotations.csv")
    votes = votes[votes.id.isin(train.id)]

    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([train, test]).to_csv(out / "comments.csv", index=False)
    votes.to_csv(out / "train_annotations.csv", index=False)
    shutil.copy(config_path, out / "config.yaml")

    for name, d in (("train", train), ("test", test)):
        print(f"{name}: {len(d)} comments, raters median {d.toxicity_annotator_count.median():.0f}, "
              f"toxicity >= 0.5: {(d.toxicity >= 0.5).mean():.1%}")
    print(f"train_annotations: {len(votes)} votes from {votes.worker.nunique()} raters")
    print(f"wrote to {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
