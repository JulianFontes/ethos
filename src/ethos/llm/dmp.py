"""Value profiles for the pluralistic (DMP) judge, after Dynamic Moral Profiling (Russo et al.).

Proof-of-concept stand-in: the base distribution G0 is uniform over the value taxonomy
(the paper uses value frequencies from human rationales). Sampling follows the paper:
draw p ~ Dirichlet(alpha * G0), keep the top-k values, renormalize their weights.
Each comment's profile is seeded by its id, so all its versions (and every model) get the same one.

Only G0 is a placeholder (marked TODO below); the Dirichlet sampling, top-k and renormalization
stay as they are when it is replaced.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # project root; not imported from ethos.judges to avoid a circular import

def load_values(path):
    return [v.strip() for v in (ROOT / path).read_text().splitlines() if v.strip()]


def sample_profile(comment_id, values, alpha=10, top_k=3, seed=0):
    """[(value, weight), ...] for one comment, highest weight first; weights sum to 1."""
    rng = np.random.default_rng([seed, int(comment_id)])
    # TODO: placeholder G0 (uniform). Replace with either ...
    #   - a human value distribution: share of human rationales mentioning each value
    #     (the paper's G0, e.g. its Table 1 frequencies), or
    #   - an LLM value weighting: per-comment value weights from Mistral (or family of models outside of evaluation)
    # Either must be a length-len(values) array in the same order as `values`, summing to 1.
    g0 = np.full(len(values), 1 / len(values))
    p = rng.dirichlet(alpha * g0)
    top = np.argsort(p)[::-1][:top_k]
    weights = p[top] / p[top].sum()
    return [(values[i], float(w)) for i, w in zip(top, weights)]


def format_profile(profile):
    """'Privacy 0.41; Loyalty 0.33; Tradition 0.26'"""
    return "; ".join(f"{v} {w:.2f}" for v, w in profile)
