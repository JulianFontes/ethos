"""Build a stratified sample of Civil Comments for the ETHOS pipeline.

Only test-split comments are used, since Detoxify was trained on the train split.
Comments are deduplicated, filtered by annotator count (and optionally to
top-level comments), binned by human
toxicity, and an equal number is drawn from each bin. All settings live in a
YAML config (see configs/sample.yaml). Overview figures (see
ethos.plotting.sample_overview) are saved to the sample's figures/ folder.

Usage:
    python scripts/extract_sample.py
    python scripts/extract_sample.py --config configs/my_sample.yaml
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ethos.plotting.sample_overview import plot_sample

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "unprocessed" / "all_data.csv"
DEFAULT_CONFIG = ROOT / "configs" / "sample.yaml"


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = {"n", "min_annotators", "seed", "output_dir"} - cfg.keys()
    if missing:
        raise SystemExit(f"{path} is missing required keys: {sorted(missing)}")
    if "bin_edges" not in cfg and "bin_width" not in cfg:
        raise SystemExit(f"{path} must set bin_width or bin_edges")
    return cfg


def make_edges(cfg):
    if cfg.get("bin_edges"):
        edges = sorted(float(e) for e in cfg["bin_edges"])
    else:
        edges = list(np.round(np.arange(0, 1 + 1e-9, cfg["bin_width"]), 6))
        if edges[-1] < 1:
            edges.append(1.0)
    if edges[0] > 0 or edges[-1] < 1:
        raise SystemExit(f"bin edges must cover [0, 1], got {edges}")
    return edges


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML config (default: configs/sample.yaml)")
    config_path = parser.parse_args().config
    cfg = load_config(config_path)

    edges = make_edges(cfg)
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]

    df = pd.read_csv(SRC)
    pool = (
        df[(df.split == "test") & (df.toxicity_annotator_count >= cfg["min_annotators"])]
        .assign(_text=lambda d: d.comment_text.str.strip())
        .drop_duplicates("_text")
        .drop(columns="_text")
    )
    if cfg.get("top_level_only", False):
        pool = pool[pool.parent_id.isna()]
    # Bins are [lo, hi); nudge the top edge so a score of exactly 1.0 lands in the last bin.
    cut_edges = edges[:-1] + [edges[-1] + 1e-9]
    pool["toxicity_bin"] = pd.cut(pool.toxicity, cut_edges, right=False, labels=labels)

    # Split n as evenly as possible; earlier bins take any remainder.
    n = cfg["n"]
    per_bin = [n // len(labels) + (i < n % len(labels)) for i in range(len(labels))]
    available = pool.toxicity_bin.value_counts()

    scope = "top-level test comments" if cfg.get("top_level_only", False) else "test comments"
    print(f"{scope} with >= {cfg['min_annotators']} annotators, deduplicated: {len(pool)}")
    short = []
    for label, k in zip(labels, per_bin):
        print(f"  {label:>10}: {available[label]:>6} available, taking {k}")
        if available[label] < k:
            short.append(label)
    if short:
        raise SystemExit(f"not enough comments in bin(s) {short}; lower min_annotators or change the bins")

    sample = pd.concat(
        pool[pool.toxicity_bin == label].sample(k, random_state=cfg["seed"])
        for label, k in zip(labels, per_bin)
    )

    # Each sample gets its own folder: the data, the config that produced it,
    # and the overview figures.
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_dir / "sample.csv", index=False)
    shutil.copy(config_path, out_dir / "config.yaml")
    plot_sample(sample, labels, cfg, out_dir / "figures")
    print(f"wrote {len(sample)} rows, config, and figures to {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
