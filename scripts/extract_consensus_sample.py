"""Draw an equal number of comments from each consensus bin (consensus = max(p, 1 - p)).

Reads the combined dataset from scripts/extract_splits.py, keeps one split (test by default), bins comments
by consensus and draws `per_bin` from each. All settings live in configs/sample_consensus.yaml.

Writes to <output_dir>/:
  sample.csv    the drawn comments (all columns of the source, plus consensus_bin)
  config.yaml   the config used

Usage:
    python scripts/extract_consensus_sample.py
    python scripts/extract_consensus_sample.py --config configs/my_consensus_sample.yaml
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sample_consensus.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_path = parser.parse_args().config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    edges = sorted(float(e) for e in cfg["bin_edges"])
    if edges[0] > 0.5 or edges[-1] < 1:
        raise SystemExit(f"bin edges must cover [0.5, 1], got {edges}")
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]

    df = pd.read_csv(ROOT / cfg["source"])
    pool = df[df.split == cfg["split"]].copy()
    # Bins are [lo, hi); nudge the top edge so a consensus of exactly 1.0 lands in the last bin.
    pool["consensus_bin"] = pd.cut(pool.consensus, edges[:-1] + [edges[-1] + 1e-9], right=False, labels=labels)

    available = pool.consensus_bin.value_counts()
    print(f"{cfg['split']} comments: {len(pool)}")
    for label in labels:
        print(f"  {label:>8}: {available[label]:>5} available, taking {cfg['per_bin']}")
    short = [label for label in labels if available[label] < cfg["per_bin"]]
    if short:
        raise SystemExit(f"not enough comments in bin(s) {short}; lower per_bin or change the bins")

    sample = pd.concat(pool[pool.consensus_bin == label].sample(cfg["per_bin"], random_state=cfg["seed"])
                       for label in labels)

    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out / "sample.csv", index=False)
    shutil.copy(config_path, out / "config.yaml")
    print(f"wrote {len(sample)} comments to {(out / 'sample.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
