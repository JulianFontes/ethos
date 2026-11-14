"""Draw an equal number of comments from each consensus bin (consensus = max(p, 1 - p)).

Reads the combined dataset from scripts/data/extract_splits.py, keeps one split (test by default), bins comments by
consensus and draws `per_bin` from each, optionally leaving out the comments of other samples (`exclude`: a list of
sample.csv paths) and optionally balancing each bin's two sides (`balance_sides`: half the bin's comments with
toxicity below 0.5, half above; comments at exactly 0.5 are left out). Settings live in configs/data/sample_consensus_test.yaml; the logic lives in
ethos.data (bin_column, sample_per_bin).

Writes to <output_dir>/:
  sample.csv    the drawn comments (all columns of the source, plus consensus_bin)
  config.yaml   the config used

Usage:
    python scripts/data/extract_consensus_sample.py
    python scripts/data/extract_consensus_sample.py --config configs/data/my_consensus_sample.yaml
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from ethos.data import bin_column, sample_per_bin
from ethos.paths import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "data" / "sample_consensus_test.yaml"


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
    for other in cfg.get("exclude", []):                     # leave out comments already in these samples
        pool = pool[~pool.id.isin(pd.read_csv(ROOT / other, usecols=["id"]).id)]
    print(f"{cfg['split']} comments: {len(pool)}")
    for label, n in pool.consensus_bin.value_counts(sort=False).items():
        print(f"  {label:>8}: {n:>5} available, taking {cfg['per_bin']}")
    if cfg.get("balance_sides"):                              # within each bin, half below and half above 0.5
        pool = pool[pool.toxicity != 0.5]
        side = (pool.toxicity > 0.5).map({False: "not toxic", True: "toxic"})
        pool = pool.assign(stratum=pd.Categorical(pool.consensus_bin.astype(str) + " | " + side))
    try:
        if cfg.get("balance_sides"):
            sample = sample_per_bin(pool, "stratum", cfg["per_bin"] // 2, cfg["seed"]).drop(columns="stratum")
        else:
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
