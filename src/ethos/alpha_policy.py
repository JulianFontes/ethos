"""Steering-strength policy: for one value vector, learn which alpha to steer each comment with so that the LLM's
toxicity score lands closest to the human score.

Data: a steering run's scores.csv (LLM score per comment and alpha; ethos.data.load_steering_scores), the comment
embeddings, and optionally <run>/rationale_embeddings.parquet (scripts/experiments/embed_rationales.py).

Model: one linear layer and a softmax over the alphas, trained on the expected distance
sum_alpha p(alpha) * |LLM score at alpha - human score|. "bell" turns the layer's two outputs (a centre and a width) into
a bell curve over the ordered alphas; "softmax" gives every alpha its own score.

Inputs (features): "text" (comment embedding), "prior" (the LLM's score at alpha 0), "rationale" (its unsteered
rationales, embedded), joined with "+". "prior" and "rationale" come from the answers that also score alpha 0, so they
share their sampling noise: their errors are somewhat optimistic.

Plots: ethos.plotting.alpha_policy. Run: scripts/experiments/alpha_policy.py (scripts/shell/run_alpha_policy.sh).
"""
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm, trange

from ethos.data import load_embeddings, load_steering_scores
from ethos.paths import ROOT

# --- data ------------------------------------------------------------------------------------------------------

def load(scores, complete_only=False):
    """(X, D, S, H, options): X (comments, 768) embeddings; per comment and alpha, D = |LLM score - human score| and
    S = the LLM toxicity score; H (comments,) human toxicity scores. Comments are sorted by id."""
    rows, _ = load_steering_scores(scores, complete_only=complete_only)
    D = rows.pivot(index="id", columns="moral_value", values="distance")
    S = rows.pivot(index="id", columns="moral_value", values="llm_toxicity")
    H = rows.groupby("id").human_toxicity.first().loc[D.index]
    X = load_embeddings().loc[D.index]
    tensor = lambda df: torch.tensor(df.to_numpy(), dtype=torch.float32)
    return tensor(X), tensor(D), tensor(S), tensor(H), list(D.columns)

def rationale_embeddings(scores, ids):
    """The run's rationale_embeddings.parquet (scripts/experiments/embed_rationales.py) as a tensor in `ids` order, or
    None if the run has none."""
    path = ROOT / Path(scores).parent / "rationale_embeddings.parquet"
    if not path.exists():
        return None
    return torch.tensor(pd.read_parquet(path).loc[ids].to_numpy(), dtype=torch.float32)

def alpha_values(options):
    """The alpha of each option ("none" -> 0, "toxicity@+0.3" -> 0.3)."""
    return [0.0 if o == "none" else float(o.rsplit("@", 1)[1]) for o in options]

def features(kind, X, S, options, train_idx, R=None):
    """The policy's inputs, from the parts in `kind` joined by "+": "text" (the comment embedding), "rationale" (the
    LLM's unsteered rationales, embedded and averaged: R) and "prior" (the LLM's score at alpha 0). Every column is
    standardised with the training comments' mean and sd. When an embedding is combined with the prior score, each
    embedding block is also divided by sqrt(its width), so that the 768 columns weigh as much as the single prior-score
    column instead of drowning it (without the prior score there is nothing to balance, and the scaling only hurts)."""
    blocks = {"text": X, "rationale": R, "prior": S[:, [options.index("none")]]}
    names = kind.split("+")
    parts = []
    for name in names:
        A = blocks[name]
        assert A is not None, f"feature {name!r} needs its data (rationale: <run>/rationale_embeddings.parquet)"
        A = (A - A[train_idx].mean(dim=0)) / (A[train_idx].std(dim=0) + 1e-8)
        parts.append(A / A.shape[1] ** 0.5 if A.shape[1] > 1 and "prior" in names else A)
    return torch.cat(parts, dim=1)


# --- models and training ---------------------------------------------------------------------------------------

def expected_distance(policy, X, D):
    """Mean over comments of sum_alpha p(alpha | comment) * distance(alpha): the training loss."""
    return (torch.softmax(policy(X), dim=1) * D).sum(dim=1).mean()


class BellPolicy(torch.nn.Module):
    """Ordered policy: a linear layer gives each comment a centre mu (inside the range of alphas) and a width sigma; each
    alpha's score is -(alpha - mu)^2 / (2 sigma^2), so after the softmax the probabilities form a bell curve over the
    alphas around mu, and neighbouring alphas get similar probabilities."""
    def __init__(self, n_inputs, alphas):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, 2)
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))

    def centre_width(self, X):
        """(mu, sigma), each (comments, 1)."""
        out = self.linear(X)
        mu = self.alphas.abs().max() * torch.tanh(out[:, :1])
        sigma = torch.nn.functional.softplus(out[:, 1:]) + 0.02
        return mu, sigma

    def forward(self, X):
        mu, sigma = self.centre_width(X)
        return -(self.alphas - mu) ** 2 / (2 * sigma ** 2)


def train(X_train, D_train, X_val, D_val, epochs=300, lr=1e-2, weight_decay=1e-3, seed=0, progress=True,
          model="softmax", alphas=None):
    """(policy, history, picks): a policy trained with Adam on the expected distance: model "softmax" (one linear score
    per alpha) or "bell" (BellPolicy; needs alphas). history has one row per epoch: the train and validation loss
    (expected), and the validation error using the policy's top alpha per comment; picks[epoch - 1] are those top
    alphas (indices) for the validation comments."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)              # a model this small trains ~100x faster on one thread than on a shared node's 32
    policy = (BellPolicy(X_train.shape[1], alphas) if model == "bell"
              else torch.nn.Linear(X_train.shape[1], D_train.shape[1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
    history, picks = [], []
    for epoch in trange(1, epochs + 1, desc="training", unit="epoch", disable=not progress, leave=False):
        opt.zero_grad()
        loss = expected_distance(policy, X_train, D_train)
        loss.backward()
        opt.step()
        with torch.no_grad():
            pick = policy(X_val).argmax(dim=1)
            picks.append(pick)
            history.append({"epoch": epoch, "train": loss.item(), "validation": expected_distance(policy, X_val, D_val).item(),
                            "validation, top alpha": D_val.gather(1, pick[:, None]).mean().item()})
    return policy, pd.DataFrame(history), picks


# --- evaluation ------------------------------------------------------------------------------------------------

def cross_validate(X, D, S, options, methods, R=None, model="bell", folds=5, epochs=300, seed=0):
    """k-fold cross-validation, so every comment is scored by a policy that never saw it. Returns (results, best):
    results  {method: (history: one row per fold and epoch, oof: (epochs, comments) each comment's picked alpha index,
             from the fold that held it out)}
    best     the best constant alpha, chosen per fold on its training comments: {"choice": option per fold,
             "error": mean of the folds' validation errors, "scores": (comments,) each comment's score at its fold's
             choice}."""
    splits = list(KFold(folds, shuffle=True, random_state=seed).split(X))
    best = {"choice": [], "errors": [], "scores": torch.zeros(len(X))}
    for tr, va in splits:
        k = int(D[tr].mean(dim=0).argmin())
        best["choice"].append(options[k]); best["errors"].append(D[va, k].mean().item()); best["scores"][va] = S[va, k]
    best["error"] = sum(best["errors"]) / len(best["errors"])

    results = {}
    with tqdm(total=len(methods) * folds, desc="training", unit="fold") as bar:
        for method in methods:
            histories, oof = [], torch.zeros(epochs, len(X), dtype=torch.long)
            for f, (tr, va) in enumerate(splits, start=1):
                tr, va = torch.tensor(tr), torch.tensor(va)
                F = features(method, X, S, options, tr, R)
                _, history, picks = train(F[tr], D[tr], F[va], D[va], epochs=epochs, seed=seed, progress=False,
                                          model=model, alphas=alpha_values(options))
                histories.append(history.assign(fold=f))
                oof[:, va] = torch.stack(picks)
                bar.update()
            results[method] = (pd.concat(histories, ignore_index=True), oof)
    return results, best


def prior_curve(X, D, S, options, epochs=300, seed=0, points=201):
    """The bell prior-score policy fitted on all comments, as a curve: (grid of prior scores 0..1, mu, sigma) as numpy
    arrays: the alpha it aims for at each prior score, and how widely it spreads probability around it."""
    prior = S[:, options.index("none")]
    F = features("prior", X, S, options, torch.arange(len(X)))
    policy, _, _ = train(F, D, F, D, epochs=epochs, seed=seed, progress=False, model="bell", alphas=alpha_values(options))
    grid = torch.linspace(0, 1, points)
    with torch.no_grad():
        mu, sigma = policy.centre_width(((grid - prior.mean()) / (prior.std() + 1e-8))[:, None])
    return grid.numpy(), mu[:, 0].numpy(), sigma[:, 0].numpy()
