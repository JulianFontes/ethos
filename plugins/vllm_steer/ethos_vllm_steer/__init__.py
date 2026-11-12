"""vLLM plugin: PCA activation steering (the vLLM version of ethos.steering.activation.steer) and hidden-state capture.

Install it into vLLM's environment, e.g. `uvx --with-editable plugins/vllm_steer --from vllm==0.33.0 vllm serve ...` (editable, so
changes to this file are picked up; a plain --with install is cached); vLLM then
loads it in every process (entry point vllm.general_plugins). It patches the decoder layers of Qwen3 and gpt-oss and
does nothing unless asked:

Steering  ETHOS_STEER=<value.npz> (keys layer_<l>, written by ethos.steering.activation.save_directions) and ETHOS_STEER_ALPHA=<α>:
          after each decoder layer l in the file, α × direction_l is added to the residual stream at every position.
          ETHOS_STEER_LAYERS="11,12,13" (optional) steers only those of the file's layers.
Swapping  ETHOS_STEER_LAYERS without ETHOS_STEER gives those layers a zero steering vector, which a running server
          replaces through WorkerExtension.ethos_steer (no restart per steering setting; see the class).
          vLLM layers return (output, residual) and add the two at the start of the next layer (fused add + norm), so
          the stream after layer l is output + residual, and adding to `residual` equals ethos.steering.activation.steer's hook
          on the HF layer output. set_steering (through LLM.collective_rpc) replaces the vectors in place.
Capture   capture_start(layers) / capture_stop() through LLM.collective_rpc, with an offline LLM(enforce_eager=True):
          every forward pass records the residual stream (output + residual) of its last token after those layers.
          Send one prompt at a time so that the last token is that prompt's last token.
"""
import logging
import os

import numpy as np
import torch

log = logging.getLogger(__name__)
_LAYERS = []        # patched layers built in this process (the worker), for set_steering
_CAPTURE = None     # {layer: [(tokens in the forward pass, last-token state), ...]} while capturing


def _patch(cls):
    init, forward = cls.__init__, cls.forward

    def patched_init(self, *args, prefix="", **kwargs):
        init(self, *args, prefix=prefix, **kwargs)
        self.ethos_layer = int(prefix.rsplit(".", 1)[-1])          # "model.layers.12" -> 12
        _LAYERS.append(self)
        path, only = os.environ.get("ETHOS_STEER"), os.environ.get("ETHOS_STEER_LAYERS")
        steered = not only or self.ethos_layer in {int(l) for l in only.split(",")}
        if path and steered:
            with np.load(path) as f:
                key = f"layer_{self.ethos_layer}"
                if key in f.files:                                   # created on the model's device and dtype
                    vector = float(os.environ.get("ETHOS_STEER_ALPHA", "1")) * f[key]
                    self.register_buffer("ethos_steer", torch.tensor(vector, dtype=torch.get_default_dtype()),
                                         persistent=False)
        elif only and steered:                                       # swappable: zeros until WorkerExtension.ethos_steer
            size = self.input_layernorm.weight.shape[0]              # hidden size
            self.register_buffer("ethos_steer", torch.zeros(size, dtype=torch.get_default_dtype()), persistent=False)

    def patched_forward(self, *args, **kwargs):
        out, residual = forward(self, *args, **kwargs)
        if hasattr(self, "ethos_steer"):
            residual = residual + self.ethos_steer
        if _CAPTURE is not None and self.ethos_layer in _CAPTURE:
            _CAPTURE[self.ethos_layer].append((out.shape[0], (out[-1] + residual[-1]).float().cpu().numpy()))
        return out, residual

    cls.__init__, cls.forward = patched_init, patched_forward


def register():
    """Entry point: patch the decoder layers vLLM will build."""
    patched = []
    for module, name in [("qwen3", "Qwen3DecoderLayer"), ("gpt_oss", "TransformerBlock")]:
        try:
            mod = __import__(f"vllm.model_executor.models.{module}", fromlist=[name])
            cls = getattr(mod, name)
        except Exception as e:                                       # keep vLLM usable for other models
            log.warning("ethos_vllm_steer: could not patch %s.%s: %s", module, name, e)
            continue
        if not getattr(cls, "_ethos_patched", False):
            _patch(cls)
            cls._ethos_patched = True
        patched.append(f"{module}.{name}")
    if os.environ.get("ETHOS_STEER"):
        log.warning("ethos_vllm_steer: steering with %s, alpha %s, layers %s (patched %s)", os.environ["ETHOS_STEER"],
                    os.environ.get("ETHOS_STEER_ALPHA", "1"), os.environ.get("ETHOS_STEER_LAYERS", "all in the file"),
                    ", ".join(patched))


# --- called inside the worker through LLM.collective_rpc(fn, args=...) -------------------------------------------

def capture_start(worker, layers):
    global _CAPTURE
    _CAPTURE = {int(l): [] for l in layers}


def capture_stop(worker):
    """{layer: [(tokens in the forward pass, last-token state as float32 array), ...]}, in order."""
    global _CAPTURE
    captured, _CAPTURE = _CAPTURE, None
    return captured


class WorkerExtension:
    """Added to vLLM's GPU worker with `vllm serve ... --worker-extension-cls ethos_vllm_steer.WorkerExtension`, so a
    running server can switch steering through its dev endpoint (VLLM_SERVER_DEV_MODE=1):
      POST /collective_rpc {"method": "ethos_steer", "args": ["<value.npz>", "<alpha>"]}   ("", "0" = unsteered)"""

    def ethos_steer(self, path, alpha):
        """Set every steering buffer to alpha x the file's direction for its layer (zeros if path is empty or alpha
        is 0). In place, so compiled graphs and CUDA graphs keep reading the same memory. Returns the layers set."""
        layers = {m.ethos_layer: m for m in _LAYERS if hasattr(m, "ethos_steer")}
        assert layers, "no steering buffers: start the server with ETHOS_STEER_LAYERS"
        alpha = float(alpha)
        directions = {}
        if path and alpha:
            with np.load(path) as f:
                missing = [l for l in layers if f"layer_{l}" not in f.files]
                assert not missing, f"{path} has no direction for layers {missing}"
                directions = {l: f[f"layer_{l}"] for l in layers}
        with torch.no_grad():
            for l, m in layers.items():
                if l in directions:
                    m.ethos_steer.copy_(torch.as_tensor(alpha * directions[l]))
                else:
                    m.ethos_steer.zero_()
        torch.cuda.synchronize()
        log.warning("ethos_vllm_steer: steering set to %s, alpha %s, layers %s", path or "none", alpha, sorted(layers))
        return sorted(layers)


def set_steering(worker, vectors):
    """Replace the steering vectors in place: {layer: α × direction}. Only layers loaded through ETHOS_STEER have one."""
    layers = {m.ethos_layer: m for m in _LAYERS if hasattr(m, "ethos_steer")}
    missing = set(vectors) - set(layers)
    assert not missing, f"no steering buffer on layers {sorted(missing)}; start with ETHOS_STEER covering them"
    for l, v in vectors.items():
        layers[l].ethos_steer.copy_(torch.as_tensor(v))
    return sorted(vectors)
