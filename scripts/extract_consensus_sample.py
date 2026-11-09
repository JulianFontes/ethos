"""Draw an equal number of comments from each consensus bin (consensus = max(p, 1 - p)).

Reads the combined dataset from scripts/extract_splits.py, keeps one split (test by default), bins comments by
consensus and draws `per_bin` from each. Settings live in configs/sample_consensus.yaml; the logic lives in
ethos.data (bin_column, sample_per_bin).

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

from ethos.data import bin_column, sample_per_bin

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

    df = pd.read_csv(ROOT / cfg["source"])
    pool = df[df.split == cfg["split"]].assign(consensus_bin=lambda d: bin_column(d.consensus, edges))
    print(f"{cfg['split']} comments: {len(pool)}")
    for label, n in pool.consensus_bin.value_counts(sort=False).items():
        print(f"  {label:>8}: {n:>5} available, taking {cfg['per_bin']}")
    try:
        sample = sample_per_bin(pool, "consensus_bin", cfg["per_bin"], cfg["seed"])
    except ValueError as e:
        raise SystemExit(str(e))

    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out / "sample.csv", index=False)
    shutil.copy(config_path, out / "config.yaml")
    print(f"wrote {len(sample)} comments to {out / 'sample.csv'}")


if __name__ == "__main__":
    main()
