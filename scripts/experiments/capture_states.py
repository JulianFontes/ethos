"""Capture last-token hidden states of VISPA pair sentences inside vLLM (step 1 of fitting steering vectors).

Runs in vLLM's environment with the steering plugin (plugins/vllm_steer) and PYTHONPATH=src. Step 2,
fit_steering_vectors.py, turns the states into directions in the ethos environment.

For each value: every for / against sentence is sent alone (one forward pass each, so its last token is the sentence's
last token), and the residual stream after each layer in --layers is recorded (ethos.steering.in_vllm.capture).
Writes <out>/<value>_states.npz with pos_<layer> / neg_<layer> arrays of shape (pairs, hidden), and states_meta.json.

Usage (scripts/shell/fit_steering_vectors.sh runs both steps on a GPU):
    VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=src uv run --no-project --python 3.12 --with vllm==0.33.0 \
        --with-editable ./plugins/vllm_steer python scripts/experiments/capture_states.py \
        --model openai/gpt-oss-20b --pairs <dir with *_context_controlled.csv> --layers 7-19 --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ethos.steering.in_vllm import capture, make_llm
from ethos.steering.pairs import load_pairs, parse_layers


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--pairs", type=Path, required=True, help="folder of <value>_context_controlled.csv files")
    p.add_argument("--values", nargs="+", help="value slugs (default: every pair file in --pairs)")
    p.add_argument("--layers", required=True, help='e.g. "7-19"')
    p.add_argument("--dtype", default="auto")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)
    values = args.values or sorted(f.name.removesuffix("_context_controlled.csv")
                                   for f in args.pairs.glob("*_context_controlled.csv"))

    llm = make_llm(args.model, args.dtype)
    for value in values:
        pos, neg = load_pairs(args.pairs / f"{value}_context_controlled.csv")
        states = {f"pos_{l}": s for l, s in capture(llm, pos, layers).items()}
        states |= {f"neg_{l}": s for l, s in capture(llm, neg, layers).items()}
        np.savez(args.out / f"{value}_states.npz", **states)
        print(f"{value}: {len(pos)} pairs, layers {args.layers}", flush=True)
    config = llm.llm_engine.model_config
    (args.out / "states_meta.json").write_text(json.dumps(
        {"model": args.model, "dtype": str(config.dtype), "layers": layers,
         "num_hidden_layers": config.hf_config.num_hidden_layers, "values": values, "pairs": str(args.pairs)}, indent=1))


if __name__ == "__main__":
    main()
