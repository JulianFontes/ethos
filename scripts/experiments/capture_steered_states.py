"""Capture the judge's internal state for comments, unsteered and activation-steered, to see whether steering shifts it.

Runs in vLLM's environment with the steering plugin (plugins/vllm_steer); scripts/shell/capture_steered_states.sh
submits it. For each setting (unsteered, value@+alpha, not_value@-alpha) and each comment, the no-value judge prompt
(prompts/judge_minimal.txt + "Comment:\\n<text>", in the model's chat format) is sent alone, and the residual stream of
its last token is recorded after each --capture layer. Steering is added after every --layers layer (from the value's
vector file), so a captured layer's state includes the pushes of all steered layers up to it.

Writes <out>/states.npz: ids, and <setting>_<layer> arrays of shape (comments, hidden); plus meta.json.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ethos.steering.in_vllm import capture, make_llm
from ethos.steering.pairs import parse_layers

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--vectors", default="data/processed/steering_vectors/gpt-oss-20b")
    p.add_argument("--value", default="toxicity")
    p.add_argument("--layers", default="7-19", help="steered layers")
    p.add_argument("--alpha", type=float, default=68.6)
    p.add_argument("--capture", default="17", help="layers whose states are saved")
    p.add_argument("--order", default="data/processed/consensus_n1000_train/steer_toxicity_gpt_oss_20b_l7-19/order.csv")
    p.add_argument("--sample", default="data/processed/consensus_n1000_train/sample.csv")
    p.add_argument("--n", type=int, default=100, help="the first n comments of --order (band-balanced)")
    p.add_argument("--reasoning-effort", default="low", help="none = Qwen3 thinking off; low / medium / high = gpt-oss reasoning")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    steered, captured = parse_layers(args.layers), parse_layers(args.capture)

    ids = [int(r["id"]) for r in csv.DictReader(open(ROOT / args.order))][: args.n]
    text = {int(r["id"]): r["comment_text"] for r in csv.DictReader(open(ROOT / args.sample))}
    system = (ROOT / "prompts/judge_minimal.txt").read_text().strip()

    llm = make_llm(args.model)                       # ETHOS_STEER_LAYERS (set by the shell script): zero vectors
    tokenizer = llm.get_tokenizer()
    # same chat options as the runs (ethos.llm.vllm_client): "none" turns Qwen3's thinking off, else gpt-oss's effort
    template = ({"enable_thinking": False} if args.reasoning_effort == "none"
                else {"reasoning_effort": args.reasoning_effort})
    prompts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": f"Comment:\n{text[i]}"}],
        tokenize=False, add_generation_prompt=True, **template) for i in ids]

    import ethos_vllm_steer as plugin
    with np.load(ROOT / args.vectors / f"{args.value}.npz") as f:
        direction = {l: f[f"layer_{l}"] for l in steered}
    settings = {"unsteered": 0.0, f"{args.value}@{args.alpha:+g}": args.alpha,
                f"not_{args.value}@{-args.alpha:+g}": -args.alpha}
    states = {"ids": np.array(ids)}
    for name, alpha in settings.items():
        llm.collective_rpc(plugin.set_steering, args=({l: alpha * d for l, d in direction.items()},))
        for l, s in capture(llm, prompts, captured).items():
            states[f"{name}_{l}"] = s
        print(f"{name}: {len(ids)} comments, layers {captured}", flush=True)
    np.savez(args.out / "states.npz", **states)
    (args.out / "meta.json").write_text(json.dumps(
        {"model": args.model, "vectors": args.vectors, "value": args.value, "steered_layers": steered,
         "captured_layers": captured, "settings": settings, "n": len(ids), "order": args.order,
         "reasoning_effort": args.reasoning_effort,
         "prompt": "prompts/judge_minimal.txt + 'Comment:\\n<text>', chat template, last token of the prompt"}, indent=1))
    print(f"wrote {args.out}/states.npz", flush=True)


if __name__ == "__main__":
    main()
