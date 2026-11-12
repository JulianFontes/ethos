"""Fit PCA steering vectors from hidden states captured in vLLM (step 2; step 1 is capture_states.py).

Reads <states>/<value>_states.npz and states_meta.json, computes each value's direction per layer with
ethos.steering.activation.directions_from_states (VISPA's PCA recipe, checked against VISPA's code), and writes to --out:
  <value>.npz              directions, keys layer_<l> (what plugins/vllm_steer reads via ETHOS_STEER)
  meta.json                model, layers, number of layers, pairs, the mean hidden-state norm per layer (`norms`,
                           the scale for alpha_fractions), and per layer and value how often the "for" sentence
                           projects above the "against" one (`separation`; 1.0 = the direction separates every pair)
  steering_layers.png      which layers are steered

Usage (scripts/shell/fit_steering_vectors.sh runs both steps):
    python scripts/experiments/fit_steering_vectors.py --states <dir> --out data/processed/steering_vectors/gpt-oss-20b
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ethos.plotting.steering import plot_steering_layers
from ethos.steering.activation import directions_from_states, save_directions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", type=Path, required=True, help="output folder of capture_states.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0, help="pair-order shuffle, as in VISPA")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.states / "states_meta.json").read_text())
    layers = meta["layers"]

    separation, norm_sums, n_states = {}, {l: 0.0 for l in layers}, 0
    for value in meta["values"]:
        with np.load(args.states / f"{value}_states.npz") as f:
            pos = {l: f[f"pos_{l}"] for l in layers}
            neg = {l: f[f"neg_{l}"] for l in layers}
        directions = directions_from_states(pos, neg, args.seed)
        save_directions(args.out / f"{value}.npz", directions)
        separation[value] = {l: float(np.mean((pos[l] - neg[l]) @ directions[l] > 0)) for l in layers}
        for l in layers:
            norm_sums[l] += float(np.linalg.norm(pos[l], axis=1).sum() + np.linalg.norm(neg[l], axis=1).sum())
        n_states += 2 * len(pos[layers[0]])
        print(f"{value}: separation min {min(separation[value].values()):.2f} over layers {layers[0]}-{layers[-1]}")

    # typical size of the residual stream at each layer (mean norm of the sentences' last-token states): the scale
    # that steering strengths are given relative to (a config's alpha_fractions)
    norms = {l: norm_sums[l] / n_states for l in layers}
    print("mean hidden-state norm per layer: " + ", ".join(f"{l}: {n:.0f}" for l, n in norms.items()))
    name = meta["model"].split("/")[-1]
    (args.out / "meta.json").write_text(json.dumps(
        meta | {"seed": args.seed, "norms": norms, "separation": separation}, indent=1))
    plot_steering_layers([(name, meta["num_hidden_layers"], layers, f"{len(meta['values'])} values")],
                         args.out / "steering_layers.png", title=f"Steering vectors: {name}")
    print(f"wrote {len(meta['values'])} vectors, meta.json and steering_layers.png to {args.out}")


if __name__ == "__main__":
    main()
