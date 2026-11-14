"""Activation steering in vLLM, through plugins/vllm_steer (checked against ethos.steering.activation).

Runs (a config's `steering:` section: vectors folder, values, alphas or alpha_fractions, and optionally layers):
  steering_settings  the (condition tag, vector file, alpha) settings a run goes through
  steering_alphas    the strengths: `alphas`, or `alpha_fractions` x the steered layers' mean hidden-state norm
  steering_server    environment variables and arguments for one `vllm serve` that every setting of a run uses
  select_steering    switch that running server to a setting (in place, through its /collective_rpc endpoint)
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


def steering_settings(steering):
    """[(condition tag, vector file, alpha)] for a config's `steering:` section, or [(None, None, None)] without one.
    Alpha 0 runs once, unsteered and untagged, so its conditions keep their names. Tags name the pole the vector
    pushes toward (see pole_name), e.g. "toxicity@+68.6", "not_fairness@-68.6"."""
    if not steering:
        return [(None, None, None)]
    alphas = steering_alphas(steering)
    settings = [(None, None, None)] if 0 in alphas else []
    return settings + [(f"{pole_name(v, a, steering)}@{a:+g}", f"{steering['vectors']}/{v}.npz", a)
                       for v in steering["values"] for a in alphas if a != 0]

def pole_name(value, alpha, steering):
    """What a vector steers toward: the value itself for + (its VISPA "for" sentences), "not_<value>" for -.
    VISPA's toxicity / harmfulness values mean avoiding them, so "toxicity@+68.6" steers toward non-toxic posting."""
    return value if alpha > 0 else f"not_{value}"

# What each pole of a vector means, for plot labels: (+alpha, -alpha), from the VISPA pairs (data/VISPA-values: "for"
# sentences = +). VISPA's toxicity and harmfulness values are about avoiding them, so their + pole is the opposite of
# what the tag name suggests: toxicity "for" = kind posting, "against" = drama and negativity; harmfulness "for" =
# checking for and preventing harm, "against" = not bothering.
POLE_MEANINGS = {"toxicity": ("non-toxic posting", "toxic posting"),
                 "harmfulness": ("avoiding harm", "disregarding harm")}

def setting_label(setting):
    """A plot label for a steering setting: "toxicity@+0.3" -> "toward non-toxic posting (toxicity vector, α +0.3)";
    "not_toxicity@-0.3" -> "toward toxic posting (toxicity vector, α -0.3)"; values without a known meaning (e.g.
    random_0) -> "random_0 vector, α +0.3"; "unsteered" (or anything without "@") is returned as is."""
    if "@" not in setting:
        return setting
    pole, alpha = setting.rsplit("@", 1)
    value = pole.removeprefix("not_")
    vector = f"{value} vector, α {float(alpha):+g}"
    if value in POLE_MEANINGS:
        return f"toward {POLE_MEANINGS[value][float(alpha) < 0]} ({vector})"
    return vector

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


def steering_server(steering):
    """(environment variables, uvx arguments, `vllm serve` arguments) for one server that a run's every steering
    setting can use: its steered layers start at zero and select_steering switches them. ({}, [], []) without
    steering. Prefix caching is off, so no answer reuses prompt states computed under another setting."""
    if not steering:
        return {}, [], []
    env = {"ETHOS_STEER_LAYERS": ",".join(map(str, steered_layers(steering))),
           "VLLM_SERVER_DEV_MODE": "1"}                     # /collective_rpc (the server listens on 127.0.0.1 only)
    serve = ["--worker-extension-cls", "ethos_vllm_steer.WorkerExtension", "--no-enable-prefix-caching"]
    return env, ["--with-editable", str(PLUGIN)], serve   # editable: always the current plugin source


def select_steering(base_url, vector=None, alpha=0):
    """Switch a running steering server (steering_server) to alpha x the vector file; no vector = unsteered.
    base_url: the server root, e.g. http://127.0.0.1:8000. Returns the layers set."""
    import urllib.request
    body = json.dumps({"method": "ethos_steer", "args": [str(ROOT / vector) if vector else "", str(alpha or 0)]})
    request = urllib.request.Request(f"{base_url}/collective_rpc", data=body.encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())["results"][0]


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


def serve(model, log, scratch, max_num_seqs, env=None, uvx_args=(), serve_args=()):
    """Start `vllm serve model` (vLLM 0.33.0 in its own environment, via uvx) on 127.0.0.1 and a free port, and wait
    until it answers. Returns (environment for clients, with VLLM_BASE_URL; port; the server process). The server's
    output is appended to `log`. env / uvx_args / serve_args: e.g. from steering_server."""
    import os
    import subprocess
    import time
    import urllib.request
    port = free_port()   # nodes are shared (one job per GPU slice), so a fixed port can already be taken
    scratch = os.path.expandvars(scratch)
    env = dict(os.environ) | {
        "UV_CACHE_DIR": f"{scratch}/uv-cache", "HF_HOME": f"{scratch}/huggingface", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{ROOT}/src", "VLLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",       # its sampler compiles CUDA code; GPU nodes have no nvcc
        "HF_HUB_OFFLINE": "1"} | (env or {})      # models come from the shared cache (gated ones need no token at run time)
    with open(log, "a") as f:
        f.write(f"\n=== vllm serve {model} {' '.join(serve_args)} ===\n")
    server = subprocess.Popen(
        ["uvx", "--managed-python", "--python", "3.12", *uvx_args, "--from", "vllm==0.33.0", "vllm", "serve", model,
         "--host", "127.0.0.1", "--port", str(port), "--max-model-len", "4096", "--gpu-memory-utilization", "0.90",
         "--max-num-seqs", str(max_num_seqs), *serve_args],
        env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT)
    print(f"starting vLLM ({model}); the first run downloads it, which can take several minutes", flush=True)
    for _ in range(180):                                         # up to 30 min
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited, see {log}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            print("vLLM ready", flush=True)
            return env, port, server
        except OSError:
            time.sleep(10)
    server.terminate()
    raise RuntimeError(f"vLLM did not start within 30 min, see {log}")


def free_port():
    """A TCP port nothing on this node is listening on."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
