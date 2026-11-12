"""Activation steering with HF transformers: a minimal re-implementation of VISPA's `projection_pca` backend.
(In vLLM: ethos.steering.in_vllm and plugins/vllm_steer, checked against this file.)

VISPA (e038eff, steering/projection_pca.py) uses RepE (andyzoujm/representation-engineering @ 5455d8a) with
direction_method="pca", n_difference=1, rep_token=-1, control_method="reading_vec". This file follows the same steps:

fit_pca  For each layer: the first PCA component of the hidden-state differences between paired sentences
         (for vs against a value), signed so that the "for" sentence projects higher in most pairs.
           1. shuffle each pair's order with random.Random(seed), as VISPA's _pairs_to_repe_format does
           2. take the last token's hidden state of every sentence after decoder block l (HF here; in vLLM:
              ethos.steering.in_vllm.capture)
           3. difference of each shuffled pair (first - second); sklearn PCA -> first component (unit length)
           4. sign: +1 if the "for" sentence projects higher in at least half of the pairs
         One deliberate difference: sentences go through one at a time, so position -1 is always the sentence's
         last token. RepE batches 8 with the tokenizer's padding side; with right padding (Qwen's default), -1 is a
         pad token for shorter sentences.
steer    Adds alpha * direction to the output of each decoder block l, at every position (RepE's linear_comb).
         plugins/vllm_steer does the same inside vLLM.
"""
import random
from contextlib import contextmanager

import numpy as np
import torch
from sklearn.decomposition import PCA


@torch.no_grad()
def last_token_states(model, tokenizer, texts, layers):
    """{layer: (len(texts), hidden) float32 array}: each text's last-token state after decoder block `layer`."""
    out = {l: [] for l in layers}
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").to(model.device)
        hidden = model(**ids, output_hidden_states=True).hidden_states    # (embeddings, block 0, block 1, ...)
        for l in layers:
            out[l].append(hidden[l + 1][0, -1].float().cpu().numpy())
    return {l: np.stack(v) for l, v in out.items()}


def directions_from_states(pos_states, neg_states, seed=0):
    """{layer: unit direction (float32)} from {layer: (pairs, hidden)} states of the for / against sentences (from HF
    here or from vLLM via ethos.steering.in_vllm.capture). Each pair's order is shuffled as VISPA does, so the differences point
    both ways and PCA finds the axis between for and against (with all differences pos - neg, centering would remove it)."""
    rng, pos_first = random.Random(seed), []
    for _ in range(len(next(iter(pos_states.values())))):
        pair = [True, False]                                     # [for, against], shuffled like VISPA's [pos, neg]
        rng.shuffle(pair)
        pos_first.append(pair[0])
    flip = np.where(pos_first, 1, -1)[:, None]

    directions = {}
    for l in pos_states:
        diffs = flip * (pos_states[l] - neg_states[l])           # first - second of each shuffled pair
        v = PCA(1).fit(diffs).components_[0]                     # first principal component (unit length)
        sign = 1 if np.mean((flip * diffs) @ v > 0) >= 0.5 else -1
        directions[l] = (sign * v).astype(np.float32)
    return directions


def fit_pca(model, tokenizer, pos, neg, layers, seed=0):
    """{layer: unit direction (float32 array)} pointing from `neg` toward `pos` sentences, with an HF model."""
    return directions_from_states(last_token_states(model, tokenizer, pos, layers),
                                  last_token_states(model, tokenizer, neg, layers), seed)


def save_directions(path, directions):
    """One .npz per value, keys layer_<l>: the format plugins/vllm_steer reads."""
    np.savez(path, **{f"layer_{l}": d for l, d in directions.items()})


def load_directions(path):
    with np.load(path) as f:
        return {int(k.removeprefix("layer_")): f[k] for k in f.files}

@contextmanager
def steer(model, directions, alpha):
    """Inside the block, add alpha * directions[l] to the output of decoder block l."""
    def add(vector):
        def hook(module, args, output):
            if isinstance(output, tuple):                    # older transformers return (hidden, ...)
                return (output[0] + vector,) + tuple(output[1:])
            return output + vector
        return hook

    layers = model.model.layers
    handles = [layers[l].register_forward_hook(
        add(torch.tensor(alpha * d, dtype=torch.float32).to(model.dtype).to(model.device)))
        for l, d in directions.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
