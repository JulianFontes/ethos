"""Check that our PCA steering (ethos.steering.activation) gives the same result as VISPA's projection_pca backend.

Two modes:
  --vispa-root DIR   (an environment with transformers<5, scikit-learn, and VISPA e038eff at DIR with RepE 5455d8a at
                     DIR/third_party/representation-engineering)
                     For each value: fit ours and VISPA's on the same model and pairs, and compare
                       1. directions per layer (cosine; 1.0 = same) against VISPA as-is (batches of 8) and VISPA with
                          batch_size=1 (no padding, like ours)
                       2. steered next-token logits and greedy generations at the same strength (alpha = VISPA's coeff)
                     and save our directions to <out>/ours_tf<version>.npz.
  (no --vispa-root)  (the ethos environment) fit ours again and compare to the saved directions: the same result
                     under a newer transformers.

Usage:
    python scripts/experiments/compare_vispa_steering.py --vispa-root <dir> --out data/experiments/05_activation_steering/vispa_equivalence
    python scripts/experiments/compare_vispa_steering.py --pairs <dir>/data/value --out data/experiments/05_activation_steering/vispa_equivalence
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ethos.steering.activation import fit_pca, steer  # noqa: E402
from ethos.steering.pairs import load_pairs  # noqa: E402

TOP_K = 5                                               # steered logits kept as the test reference
PROMPTS = [
    "Comment: You people are all idiots and should be banned from this site.\nIs this comment toxic? Answer:",
    "What should a manager do when an employee makes a costly mistake?",
    "Describe how you would spend a free weekend.",
]


def cosines(a, b):
    return {l: float(a[l] @ b[l] / np.linalg.norm(a[l]) / np.linalg.norm(b[l])) for l in a}

def vispa_directions(state, num_layers):
    """{physical layer: unit direction} from a fitted VISPA projection_pca state (activations = coeff * dir * sign)."""
    out = {}
    for neg, act in state["activations"].items():
        v = act.float().cpu().numpy().ravel()            # RepE keeps (n_components=1, hidden)
        out[neg + num_layers] = v / np.linalg.norm(v)
    return out

@torch.no_grad()
def last_logits(model, tokenizer, prompt):
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    return model(**ids).logits[0, -1].float().cpu()


@torch.no_grad()
def greedy(model, tokenizer, prompt, max_new_tokens):
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


def compare_with_vispa(args, model, tokenizer, layers):
    sys.path.insert(0, str(args.vispa_root))
    from steering.projection_pca import ProjectionPCASteering

    import repe.rep_control_reading_vec as rcv

    def getattr_through(self, name):                      # transformers 4.57 reads layer attributes (attention_type)
        try:                                              # through RepE's wrapper: fall through to the wrapped layer
            return torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            return getattr(torch.nn.Module.__getattr__(self, "block"), name)
    rcv.WrappedBlock.__getattr__ = getattr_through

    n = model.config.num_hidden_layers
    results, saved = {}, {}
    for value in args.values:
        pos, neg = load_pairs(args.vispa_root / "data/value" / f"{value}_context_controlled.csv")
        ours = fit_pca(model, tokenizer, pos, neg, layers)
        saved |= {f"{value}/{l}": d for l, d in ours.items()}
        as_is = vispa_directions(ProjectionPCASteering(coeff=1.0).fit(model, tokenizer, value, pos, neg, layers), n)
        r = {"cos_vs_vispa_batch1": None, "cos_vs_vispa_as_is": cosines(ours, as_is), "steering": []}
        for alpha in args.alphas:
            backend = ProjectionPCASteering(coeff=alpha, batch_size=1)
            state = backend.fit(model, tokenizer, value, pos, neg, layers)
            if r["cos_vs_vispa_batch1"] is None:
                r["cos_vs_vispa_batch1"] = cosines(ours, vispa_directions(state, n))
            wrapped = state["rep_control_pipeline"].wrapped_model
            for prompt in PROMPTS:
                wrapped.set_controller(list(state["activations"]), state["activations"])
                logits_v = last_logits(model, tokenizer, prompt)
                wrapped.reset()
                text_v = backend.generate(model, tokenizer, prompt, state, max_new_tokens=args.max_new_tokens)
                with steer(model, ours, alpha):
                    logits_o = last_logits(model, tokenizer, prompt)
                    text_o = greedy(model, tokenizer, prompt, args.max_new_tokens)
                top = logits_o.topk(TOP_K)                # reference for the regression test
                saved[f"steer/{value}/{alpha}/{PROMPTS.index(prompt)}/ids"] = top.indices.numpy()
                saved[f"steer/{value}/{alpha}/{PROMPTS.index(prompt)}/logits"] = top.values.numpy()
                r["steering"].append({
                    "alpha": alpha, "prompt": prompt, "same_text": text_v == text_o,
                    "same_top_token": int(logits_v.argmax()) == int(logits_o.argmax()),
                    "max_logit_diff": float((logits_v - logits_o).abs().max()),
                    "logit_range": float(logits_v.max() - logits_v.min()),
                    "finite": bool(torch.isfinite(logits_v).all() and torch.isfinite(logits_o).all()),
                    "vispa": text_v, "ours": text_o})
        results[value] = r
        b1, a8 = r["cos_vs_vispa_batch1"].values(), r["cos_vs_vispa_as_is"].values()
        s = r["steering"]
        print(f"{value}: cosine vs VISPA batch 1 min {min(b1):.6f}; vs VISPA as-is min {min(a8):.4f} "
              f"mean {np.mean(list(a8)):.4f}; same top token {sum(x['same_top_token'] for x in s)}/{len(s)}, "
              f"same greedy text {sum(x['same_text'] for x in s)}/{len(s)}, "
              f"max logit diff {max(x['max_logit_diff'] for x in s):.3g}")
    np.savez(args.out / f"ours_tf{transformers.__version__}.npz", **saved)
    return results


def compare_with_saved(args, model, tokenizer, layers):
    """Regression test: refit ours and steer; fail unless directions, top tokens and top logits match the reference."""
    saved = sorted(args.out.glob("ours_tf4*.npz"))
    assert saved, f"run with --vispa-root first (no {args.out}/ours_tf4*.npz)"
    old, ok, results = np.load(saved[-1]), True, {}
    for value in args.values:
        pos, neg = load_pairs(args.pairs / f"{value}_context_controlled.csv")
        ours = fit_pca(model, tokenizer, pos, neg, layers)
        c = cosines(ours, {l: old[f"{value}/{l}"] for l in layers})
        same_top, max_diff = 0, 0.0
        for alpha in args.alphas:
            for i, prompt in enumerate(PROMPTS):
                with steer(model, ours, alpha):
                    logits = last_logits(model, tokenizer, prompt)
                ids = old[f"steer/{value}/{alpha}/{i}/ids"]
                same_top += int(logits.argmax()) == int(ids[0])
                max_diff = max(max_diff, float(np.abs(logits[ids].numpy() - old[f"steer/{value}/{alpha}/{i}/logits"]).max()))
        n = len(args.alphas) * len(PROMPTS)
        passed = min(c.values()) > 0.9999 and same_top == n and max_diff < 0.25
        ok &= passed
        results[value] = {"cos_vs_" + saved[-1].stem: c, "same_top_token": same_top, "max_top_logit_diff": max_diff,
                          "passed": passed}
        print(f"{value}: {'PASS' if passed else 'FAIL'}  cosine vs {saved[-1].name} min {min(c.values()):.6f}; "
              f"steered top token same {same_top}/{n}; max top-{TOP_K} logit diff {max_diff:.3g}")
    if not ok:
        raise SystemExit("steering regression test FAILED")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vispa-root", type=Path, help="VISPA e038eff checkout (with third_party/representation-engineering)")
    p.add_argument("--pairs", type=Path, help="folder of VISPA pair files (without --vispa-root; default <out>/pairs)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", default="float16", help="VISPA loads float16")
    p.add_argument("--values", nargs="+", default=["fairness", "justice", "tradition"])
    p.add_argument("--layers", default="10-25", help="VISPA's default range")
    p.add_argument("--alphas", nargs="+", type=float, default=[1.0, 4.0])
    p.add_argument("--max-new-tokens", type=int, default=40)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.pairs = args.pairs or args.out / "pairs"
    start, end = map(int, args.layers.split("-"))
    layers = list(range(start, end + 1))

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"{args.model} ({args.dtype}), transformers {transformers.__version__}, layers {args.layers}")

    results = (compare_with_vispa if args.vispa_root else compare_with_saved)(args, model, tokenizer, layers)
    name = "vispa" if args.vispa_root else "saved"
    (args.out / f"compare_{name}_tf{transformers.__version__}.json").write_text(json.dumps(
        {"model": args.model, "dtype": args.dtype, "layers": layers, "results": results}, indent=1))


if __name__ == "__main__":
    main()
