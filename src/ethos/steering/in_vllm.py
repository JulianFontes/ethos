"""Activation steering in vLLM, through plugins/vllm_steer (checked against ethos.steering.activation).

Runs (a config's `steering:` section: vectors folder, values, alphas or alpha_fractions, and optionally layers; or
`defaults: configs/<model>/steering.yaml` with the model's calibrated settings, plus what the run changes):
  resolve_steering   the section with its defaults file merged in (submit_vllm.py calls it on loading a config)
  steering_settings  the (condition tag, vector file, alpha) settings a run goes through, one vLLM server each
  steering_alphas    the strengths: `alphas`, or `alpha_fractions` x the steered layers' mean hidden-state norm
  steering_server    environment variables and uvx arguments that start a steered `vllm serve`
  steered_layers     the layers a run steers (its `layers:`, or all the vectors were fit on)
  save_run_figure    <data>/steering_layers.png and steering.json: which layers, vectors and strengths a run uses
Fitting vectors (in vLLM's environment, with the plugin installed):
  make_llm           an offline vLLM model set up for capturing
  capture            each text's last-token residual stream after the given layers

Imports only the standard library, numpy and ethos.paths at the top, so vLLM's environment can use it.
"""
import json

import numpy as np

from ethos.paths import ROOT
from ethos.steering.pairs import parse_layers

PLUGIN = ROOT / "plugins/vllm_steer"


def resolve_steering(steering):
    """A config's `steering:` section with its `defaults:` file (a model's calibrated settings, e.g.
    configs/gpt_oss_20b/steering.yaml) merged in; keys set in the run override the file's. None stays None."""
    if not steering or "defaults" not in steering:
        return steering
    import yaml
    defaults = yaml.safe_load((ROOT / steering["defaults"]).read_text()) or {}
    return defaults | {k: v for k, v in steering.items() if k != "defaults"} | {"defaults": steering["defaults"]}


def steering_settings(steering):
    """[(condition tag, vector file, alpha)] for a config's `steering:` section, or [(None, None, None)] without one.
    Alpha 0 runs once, unsteered and untagged, so its conditions keep their names."""
    if not steering:
        return [(None, None, None)]
    alphas = steering_alphas(steering)
    settings = [(None, None, None)] if 0 in alphas else []
    return settings + [(f"{v}@{a:+g}", f"{steering['vectors']}/{v}.npz", a)
                       for v in steering["values"] for a in alphas if a != 0]

def steering_alphas(steering):
    """The strengths a run uses: `alphas` as given, or `alpha_fractions` x the steered layers' mean hidden-state norm
    (meta.json `norms`, from fit_steering_vectors.py), to 3 significant figures. Fractions make strengths comparable
    across models, whose hidden states differ in size."""
    if "alphas" in steering:
        return list(steering["alphas"])
    return [float(f"{f * norm_scale(steering):.3g}") for f in steering["alpha_fractions"]]

def norm_scale(steering):
    """Mean hidden-state norm over the layers a run steers."""
    norms = json.loads((ROOT / steering["vectors"] / "meta.json").read_text()).get("norms")
    assert norms, f"no norms in {steering['vectors']}/meta.json: refit with scripts/shell/fit_steering_vectors.sh"
    layers = steered_layers(steering)
    return sum(norms[str(l)] for l in layers) / len(layers)


def steering_server(vector, alpha, layers=None):
    """(environment variables, uvx arguments) for a `vllm serve` that adds alpha x the vector file's directions after
    their layers (only `layers`, e.g. "11-16", if given); ({}, []) for an unsteered server."""
    if not vector:
        return {}, []
    env = {"ETHOS_STEER": str(ROOT / vector), "ETHOS_STEER_ALPHA": str(alpha),
           "VLLM_DISABLE_COMPILE_CACHE": "1"}           # never reuse a model compiled under another steering setting
    if layers:
        env["ETHOS_STEER_LAYERS"] = ",".join(map(str, parse_layers(str(layers))))
    return env, ["--with", str(PLUGIN)]


def steered_layers(steering):
    """The layers a run steers: its `layers:` (e.g. "11-16"), or every layer the vectors were fit on. Checks that
    they were fit."""
    meta = json.loads((ROOT / steering["vectors"] / "meta.json").read_text())
    layers = parse_layers(str(steering["layers"])) if steering.get("layers") else meta["layers"]
    unfit = sorted(set(layers) - set(meta["layers"]))
    assert not unfit, f"layers {unfit} are not in {steering['vectors']} (fit on {meta['layers']})"
    return layers


def save_run_figure(model, steering, data):
    """Write <data>/steering_layers.png and steering.json for a run; returns the figure's path."""
    from ethos.plotting.steering import plot_steering_layers
    vectors = ROOT / steering["vectors"]
    meta = json.loads((vectors / "meta.json").read_text())
    assert meta["model"] == model, f"vectors in {steering['vectors']} are for {meta['model']}, not {model}"
    missing = [v for v in steering["values"] if not (vectors / f"{v}.npz").exists()]
    assert not missing, f"no vectors for {missing} in {steering['vectors']}"
    layers = steered_layers(steering)
    out = ROOT / data
    out.mkdir(parents=True, exist_ok=True)
    name = model.split("/")[-1]
    alphas = steering_alphas(steering)
    note = f"values: {', '.join(steering['values'])}; alpha: {', '.join(f'{a:g}' for a in alphas)}"
    plot_steering_layers([(name, meta["num_hidden_layers"], layers, note)], out / "steering_layers.png",
                         title=f"Steering in this run: {name}")
    extra = {"layers": layers, "num_hidden_layers": meta["num_hidden_layers"], "alphas": alphas}
    if "alpha_fractions" in steering:
        extra["norm_scale"] = norm_scale(steering)
    (out / "steering.json").write_text(json.dumps(steering | extra, indent=1))
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
