"""Random steering vectors: the control for a value's vector. Same file format, layers and length (1) as the value's
vectors, but a random direction per layer, so any change they cause is a generic push, not the value's.

Writes <vectors>/random_<i>.npz for i in 0..n-1 (seed = i), next to the value vectors, so a config can list them as
`values: [random_0, random_1, ...]`.

Usage:
    python scripts/experiments/make_random_vectors.py --vectors data/processed/steering_vectors/Meta-Llama-3-8B-Instruct --like toxicity
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vectors", type=Path, required=True, help="a model's steering vectors folder")
    p.add_argument("--like", default="toxicity", help="value whose layers and sizes to copy")
    p.add_argument("-n", type=int, default=3, help="how many random vectors")
    args = p.parse_args()
    folder = ROOT / args.vectors
    with np.load(folder / f"{args.like}.npz") as ref:
        ref = dict(ref)
    for i in range(args.n):
        rng = np.random.default_rng(i)
        out = {}
        for key, v in ref.items():
            d = rng.standard_normal(v.shape)
            out[key] = (d / np.linalg.norm(d) * np.linalg.norm(v)).astype(v.dtype)
        np.savez(folder / f"random_{i}.npz", **out)
        cos = np.mean([abs(out[k] @ ref[k]) / (np.linalg.norm(out[k]) * np.linalg.norm(ref[k])) for k in ref])
        print(f"wrote random_{i}.npz ({len(out)} layers; mean |cosine| with {args.like}: {cos:.3f})")


if __name__ == "__main__":
    main()
