"""Activation steering in vLLM, through plugins/vllm_steer (checked against ethos.steering.activation).

Runs (a config's `steering:` section: vectors folder, values, alphas):
  steering_settings  the (condition tag, vector file, alpha) settings a run goes through, one vLLM server each
  steering_server    environment variables and uvx arguments that start a steered `vllm serve`
  save_run_figure    <data>/steering_layers.png and steering.json: which layers, vectors and strengths a run uses
Fitting vectors (in vLLM's environment, with the plugin installed):
  make_llm           an offline vLLM model set up for capturing
  capture            each text's last-token residual stream after the given layers

Imports only the standard library, numpy and ethos.paths at the top, so vLLM's environment can use it.
"""
import json

import numpy as np

from ethos.paths import ROOT

PLUGIN = ROOT / "plugins/vllm_steer"


def steering_settings(steering):
    """[(condition tag, vector file, alpha)] for a config's `steering:` section ({vectors, values, alphas}), or
    [(None, None, None)] without one. Alpha 0 runs once, unsteered and untagged, so its conditions keep their names."""
    if not steering:
        return [(None, None, None)]
    settings = [(None, None, None)] if 0 in steering["alphas"] else []
    return settings + [(f"{v}@{a:+g}", f"{steering['vectors']}/{v}.npz", a)
                       for v in steering["values"] for a in steering["alphas"] if a != 0]


def steering_server(vector, alpha):
    """(environment variables, uvx arguments) for a `vllm serve` that adds alpha x the vector file's directions after
    their layers; ({}, []) for an unsteered server."""
    if not vector:
        return {}, []
    env = {"ETHOS_STEER": str(ROOT / vector), "ETHOS_STEER_ALPHA": str(alpha),
           "VLLM_DISABLE_COMPILE_CACHE": "1"}           # never reuse a model compiled under another steering setting
    return env, ["--with", str(PLUGIN)]


def save_run_figure(model, steering, data):
    """Write <data>/steering_layers.png and steering.json for a run; returns the figure's path."""
    from ethos.plotting.steering import plot_steering_layers
    vectors = ROOT / steering["vectors"]
    meta = json.loads((vectors / "meta.json").read_text())
    assert meta["model"] == model, f"vectors in {steering['vectors']} are for {meta['model']}, not {model}"
    missing = [v for v in steering["values"] if not (vectors / f"{v}.npz").exists()]
    assert not missing, f"no vectors for {missing} in {steering['vectors']}"
    out = ROOT / data
    out.mkdir(parents=True, exist_ok=True)
    name = model.split("/")[-1]
    note = f"values: {', '.join(steering['values'])}; alpha: {', '.join(f'{a:g}' for a in steering['alphas'])}"
    plot_steering_layers([(name, meta["num_hidden_layers"], meta["layers"], note)], out / "steering_layers.png",
                         title=f"Steering in this run: {name}")
    (out / "steering.json").write_text(json.dumps(
        steering | {"layers": meta["layers"], "num_hidden_layers": meta["num_hidden_layers"]}, indent=1))
    return out / "steering_layers.png"


def make_llm(model, dtype="auto", **kwargs):
    """An offline vLLM model for capture: eager (the plugin records states in Python), no prefix caching (every
    token is computed). Set VLLM_ENABLE_V1_MULTIPROCESSING=0 so the plugin's RPCs run in this process."""
    from vllm import LLM
    return LLM(model=model, dtype=dtype, enforce_eager=True, enable_prefix_caching=False, max_model_len=1024,
               gpu_memory_utilization=0.90, **kwargs)


def capture(llm, texts, layers):
    """{layer: (len(texts), hidden) float32}: each text's last-token residual stream after `layer`, one text per
    forward pass (same tokens as ethos.steering.activation.last_token_states)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    import ethos_vllm_steer as plugin
    tokenizer = llm.get_tokenizer()
    out = {l: [] for l in layers}
    for text in texts:
        ids = tokenizer(text).input_ids
        llm.collective_rpc(plugin.capture_start, args=(layers,))
        llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1), use_tqdm=False)
        captured = llm.collective_rpc(plugin.capture_stop)[0]
        for l in layers:
            assert len(captured[l]) == 1 and captured[l][0][0] == len(ids), \
                f"layer {l}: expected one forward pass of {len(ids)} tokens, got {[n for n, _ in captured[l]]}"
            out[l].append(captured[l][0][1])
    return {l: np.stack(v) for l, v in out.items()}
