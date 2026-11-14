"""Embed an LLM's unsteered rationales with the comment-embedding model (all-mpnet-base-v2) and average them per comment.

Reads <run>/ratings.csv (a steering run of profile_signal.py), keeps the unsteered answers (condition "none"), embeds
each rationale and writes <run>/rationale_embeddings.parquet: one row per comment id, emb_0 ... emb_767, the mean of its
rationales' (unit-length) embeddings. Used by notebooks/06_rationale_clusters.ipynb and as a policy input.

Usage (a CPU job: ~6 min for 20,000 rationales on 8 cores; the model must be in the Hugging Face cache):
    sbatch -p test --account=protopapas_lab -c 8 --mem=16G -t 01:00:00 -o <run>/logs/embed-rationales-%j.out \\
        --wrap "HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/experiments/embed_rationales.py --run <run>"
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from ethos.data import embed
from ethos.paths import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="a steering run's folder (ratings.csv)")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    run = ROOT / args.run
    r = pd.read_csv(run / "ratings.csv")
    u = r[r.condition == "none"].reset_index(drop=True)
    t0 = time.time()
    e = embed(u.rationale)                                   # one row per answer, unit length
    e.index = u.id.values
    mean = e.groupby(level=0).mean()                          # one row per comment: the mean of its rationales
    mean.to_parquet(run / "rationale_embeddings.parquet")
    print(f"embedded {len(u)} rationales for {len(mean)} comments in {time.time() - t0:.0f} s -> "
          f"{args.run}/rationale_embeddings.parquet")


if __name__ == "__main__":
    main()
