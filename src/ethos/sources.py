"""Record which data an experiment's results were made from.

Data lives in data/processed/ and data/judgements/; results (figures, summaries, checks) live in data/experiments/.
Every script that writes results calls write_sources(out_dir, inputs), which writes <out_dir>/sources.yaml:

    created: 2026-11-10 14:32
    command: scripts/plots/plot_profiles.py --data data/processed/civil-comments/consensus_n125_train/single_values ...
    git_commit: a1b2c3d (uncommitted changes)
    inputs:
      - path: data/processed/civil-comments/consensus_n125_train/single_values/ratings.csv
        rows: 76250
        md5: 9f2c...

The md5 changes whenever the input changes, so a figure made from older data can be spotted.
"""
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from ethos.paths import ROOT


def _rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _rows(path):
    """Data rows in a CSV (lines minus the header) or Parquet file; None for other files."""
    if path.suffix == ".parquet":
        return pq.ParquetFile(path).metadata.num_rows
    if path.suffix == ".csv":
        import pandas as pd
        return len(pd.read_csv(path, usecols=[0]))   # counts records, not lines (comments contain newlines)
    return None


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return f"{commit} (uncommitted changes)" if dirty.strip() else commit
    except (OSError, subprocess.CalledProcessError):
        return None


def write_sources(out_dir, inputs):
    """Write <out_dir>/sources.yaml listing `inputs` (paths of the data files used) with row counts and md5."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "command": " ".join([_rel(sys.argv[0])] + sys.argv[1:]),
        "git_commit": _git_commit(),
        "inputs": [{"path": _rel(p), "rows": _rows(Path(p)), "md5": _md5(p)} for p in inputs],
    }
    (out_dir / "sources.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    return out_dir / "sources.yaml"
