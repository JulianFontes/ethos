"""Check the vLLM steering plugin (plugins/vllm_steer) against our HF steering, which matches VISPA's code.

Reference: data/experiments/05_activation_steering/vispa_equivalence/ours_tf4.57.6.npz (Qwen3-8B, float16, layers 10-25,
fairness / justice / tradition; directions and steered top-5 logits for 3 prompts at alpha 1 and 4).

  --run      (vLLM's environment, with the plugin) capture the pair sentences' states in vLLM, and steer the 3 prompts
             with the reference directions, the way runs do: zero steering on the layers at startup
             (ETHOS_STEER_LAYERS), switched with WorkerExtension.ethos_steer; writes <out>/vllm/
  --compare  (ethos environment) fit directions from the vLLM states and compare with the reference: cosine per layer,
             steered top token, and gaps between the top-5 logits (vLLM logprob gaps = logit gaps). Exits 1 on FAIL.

scripts/shell/check_vllm_steering.sh runs both on a gpu_test slice.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from ethos.steering.in_vllm import capture, make_llm
from ethos.steering.pairs import load_pairs

sys.path.insert(0, str(Path(__file__).resolve().parent))   # for compare_vispa_steering.PROMPTS in --compare
REF = Path("data/experiments/05_activation_steering/vispa_equivalence")
PROMPTS = [  # same as compare_vispa_steering.PROMPTS (checked in --compare)
    "Comment: You people are all idiots and should be banned from this site.\nIs this comment toxic? Answer:",
    "What should a manager do when an employee makes a costly mistake?",
    "Describe how you would spend a free weekend.",
]
VALUES, LAYERS, ALPHAS = ["fairness", "justice", "tradition"], list(range(10, 26)), [1.0, 4.0]

def run(out, model):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    out.mkdir(parents=True, exist_ok=True)
    ref = np.load(sorted(REF.glob("ours_tf4*.npz"))[-1])
    for value in VALUES:                                           # the reference directions as plugin vector files
        np.savez(out / f"{value}.npz", **{f"layer_{l}": ref[f"{value}/{l}"] for l in LAYERS})
    # the runs' code path: zero steering on layers 10-25 at startup, then WorkerExtension.ethos_steer switches it
    os.environ["ETHOS_STEER_LAYERS"] = ",".join(map(str, LAYERS))
    llm = make_llm(model, dtype="float16", worker_extension_cls="ethos_vllm_steer.WorkerExtension")

    for value in VALUES:                                           # unsteered states (zero steering)
        pos, neg = load_pairs(REF / "pairs" / f"{value}_context_controlled.csv")
        states = {f"pos_{l}": s for l, s in capture(llm, pos, LAYERS).items()}
        states |= {f"neg_{l}": s for l, s in capture(llm, neg, LAYERS).items()}
        np.savez(out / f"{value}_states.npz", **states)
        print(f"captured {value}", flush=True)

    tokenizer, steered = llm.get_tokenizer(), {}
    params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
    for value in VALUES:
        for alpha in ALPHAS:
            llm.collective_rpc("ethos_steer", args=(str((out / f"{value}.npz").resolve()), str(alpha)))
            for i, prompt in enumerate(PROMPTS):
                result = llm.generate([TokensPrompt(prompt_token_ids=tokenizer(prompt).input_ids)], params,
                                      use_tqdm=False)[0]
                top = result.outputs[0].logprobs[0]
                steered[f"{value}/{alpha}/{i}/ids"] = np.array(list(top))
                steered[f"{value}/{alpha}/{i}/logprobs"] = np.array([lp.logprob for lp in top.values()])
    np.savez(out / "steered.npz", **steered)
    print(f"wrote {out}")


def compare(out):
    from compare_vispa_steering import PROMPTS as HF_PROMPTS
    from ethos.steering.activation import directions_from_states
    assert PROMPTS == HF_PROMPTS, "prompts differ from compare_vispa_steering.py"

    ref = np.load(sorted(REF.glob("ours_tf4*.npz"))[-1])
    steered = np.load(out / "steered.npz")
    ok, results = True, {}
    for value in VALUES:
        with np.load(out / f"{value}_states.npz") as f:
            d = directions_from_states({l: f[f"pos_{l}"] for l in LAYERS}, {l: f[f"neg_{l}"] for l in LAYERS})
        cos = {l: float(d[l] @ ref[f"{value}/{l}"]) for l in LAYERS}          # both unit length
        same_top, max_gap = 0, 0.0
        for alpha in ALPHAS:
            for i in range(len(PROMPTS)):
                ids, logits = ref[f"steer/{value}/{alpha}/{i}/ids"], ref[f"steer/{value}/{alpha}/{i}/logits"]
                lp = dict(zip(steered[f"{value}/{alpha}/{i}/ids"], steered[f"{value}/{alpha}/{i}/logprobs"]))
                vllm_top = max(lp, key=lp.get)
                same_top += int(vllm_top == ids[0])
                for k in range(1, len(ids)):
                    if ids[k] in lp:
                        max_gap = max(max_gap, abs((lp[ids[0]] - lp[ids[k]]) - (logits[0] - logits[k])))
        n = len(ALPHAS) * len(PROMPTS)
        passed = bool(min(cos.values()) > 0.99 and same_top == n and max_gap < 0.5)
        ok &= passed
        results[value] = {"cosine": cos, "same_top_token": same_top, "max_logit_gap_diff": max_gap, "passed": passed}
        print(f"{value}: {'PASS' if passed else 'FAIL'}  direction cosine vs HF min {min(cos.values()):.4f}; "
              f"steered top token same {same_top}/{n}; max top-5 logit-gap difference {max_gap:.3f}")
    (out / "compare.json").write_text(json.dumps(results, indent=1))
    if not ok:
        raise SystemExit("vLLM steering check FAILED")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--compare", action="store_true")
    p.add_argument("--out", type=Path, default=REF / "vllm")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    args = p.parse_args()
    run(args.out, args.model) if args.run else compare(args.out)


if __name__ == "__main__":
    main()
