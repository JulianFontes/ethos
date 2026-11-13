"""Check that switching steering in a running vLLM server (select_steering, no restart) gives the same answers as
starting a server with that steering, i.e. the in-place swap reaches the compiled model and its CUDA graphs.

  1. one swappable server (steering_server): for each setting below, switch, then ask the prompts
  2. fresh servers started with the fairness vector fixed (ETHOS_STEER) at alpha and at 0, and one without the
     plugin (reported for information: the rounding difference between plugin and no-plugin servers)
  3. compare next-token log-probabilities (temperature 0, one prompt at a time):
       swapped fairness = fresh fairness, swapped unsteered = fresh plugin server at alpha 0, fairness twice = same,
       and fairness != justice (the swap changes something)
Writes <out>/swap_check.json; exits 1 on FAIL. scripts/shell/check_steering_swap.sh runs it on a gpu_test slice.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path


from ethos.paths import ROOT
from ethos.steering.in_vllm import select_steering, serve, steered_layers, steering_server

PROMPTS = ["Comment: You people are all idiots and should be banned from this site.\nIs this comment toxic? Answer:",
           "Comment: I disagree with the mayor's new parking policy.\nIs this comment toxic? Answer:",
           "What should a manager do when an employee makes a costly mistake?"]


def ask(port, model):
    """[{token: logprob}] of the next token, per prompt (one request at a time, so batches are identical)."""
    out = []
    for prompt in PROMPTS:
        body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 5})
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body.encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out.append(json.loads(r.read())["choices"][0]["logprobs"]["top_logprobs"][0])
    return out


def diff(a, b):
    """(same top token for every prompt, largest |logprob difference| over tokens in both top-5s)."""
    same = all(max(x, key=x.get) == max(y, key=y.get) for x, y in zip(a, b))
    gap = max(abs(x[t] - y[t]) for x, y in zip(a, b) for t in x if t in y)
    return same, gap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vectors", default="data/processed/steering_vectors/gpt-oss-20b")
    p.add_argument("--layers", default="7-19")
    p.add_argument("--alpha", type=float, default=68.6)
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--scratch", default="/n/netscratch/protopapas_lab/Lab/$USER")
    p.add_argument("--out", type=Path, default=Path("data/experiments/civil-comments/05_activation_steering/swap_check"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "vllm.log"
    steering = {"vectors": args.vectors, "layers": args.layers, "values": ["fairness", "justice"],
                "alphas": [0, -args.alpha, args.alpha]}
    alpha = args.alpha
    fairness, justice = f"{steering['vectors']}/fairness.npz", f"{steering['vectors']}/justice.npz"
    fresh_env = {"VLLM_DISABLE_COMPILE_CACHE": "1"}                 # fresh references never reuse a compiled model

    env, uvx, serve_args = steering_server(steering)                # 1. swappable server
    _, port, server = serve(args.model, log, args.scratch, 8, env, uvx, serve_args)
    try:
        swapped = {}
        for name, vector, a in [("fairness", fairness, alpha), ("justice", justice, -alpha), ("unsteered", None, 0),
                                ("fairness_again", fairness, alpha)]:
            layers = select_steering(f"http://127.0.0.1:{port}", vector, a)
            swapped[name] = ask(port, args.model)
            print(f"swapped to {name} (alpha {a}, layers {layers})", flush=True)
    finally:
        server.terminate(), server.wait()

    layers = ",".join(map(str, steered_layers(steering)))         # 2. fresh servers
    _, uvx, _ = steering_server(steering)
    fixed = fresh_env | {"ETHOS_STEER": str(ROOT / fairness), "ETHOS_STEER_ALPHA": str(alpha), "ETHOS_STEER_LAYERS": layers}
    _, port, server = serve(args.model, log, args.scratch, 8, fixed, uvx)
    try:
        fresh_fairness = ask(port, args.model)
    finally:
        server.terminate(), server.wait()
    zero = fixed | {"ETHOS_STEER_ALPHA": "0"}                      # the plugin at alpha 0: the same model graph
    _, port, server = serve(args.model, log, args.scratch, 8, zero, uvx)
    try:
        fresh_unsteered = ask(port, args.model)
    finally:
        server.terminate(), server.wait()
    _, port, server = serve(args.model, log, args.scratch, 8, fresh_env)   # no plugin at all (for information:
    try:                                                                   # how far all plugin runs are from runs
        no_plugin = ask(port, args.model)                                  # without it, from rounding alone)
    finally:
        server.terminate(), server.wait()

    checks = {                                                     # 3. (same top token, max logprob gap), expected
        "swapped fairness = fresh fairness": (diff(swapped["fairness"], fresh_fairness), True),
        "swapped unsteered = fresh plugin at alpha 0": (diff(swapped["unsteered"], fresh_unsteered), True),
        "fairness, then again after other settings": (diff(swapped["fairness"], swapped["fairness_again"]), True),
        "fairness differs from justice": (diff(swapped["fairness"], swapped["justice"]), False),
        "fairness differs from unsteered": (diff(swapped["fairness"], fresh_unsteered), False),
        "info: unsteered vs no plugin (rounding only)": (diff(swapped["unsteered"], no_plugin), None),
    }
    ok, report = True, {}
    for name, ((same, gap), should_match) in checks.items():
        passed = None if should_match is None else (same and gap < 0.05) if should_match else gap > 0.05
        ok &= passed is not False
        report[name] = {"same_top_token": same, "max_logprob_gap": gap, "passed": passed}
        label = "INFO" if passed is None else "PASS" if passed else "FAIL"
        print(f"{label}  {name}: same top token {same}, max logprob gap {gap:.4f}")
    (args.out / "swap_check.json").write_text(json.dumps(
        {"model": args.model, "alpha": alpha, "layers": layers, "checks": report, "swapped": swapped,
         "fresh_fairness": fresh_fairness, "fresh_plugin_alpha0": fresh_unsteered, "no_plugin": no_plugin}, indent=1))
    if not ok:
        sys.exit("steering swap check FAILED")


if __name__ == "__main__":
    main()
