"""Linear softmax policy: from a comment's features, choose the steering strength (alpha) whose LLM toxicity score is
closest to the human score, for one fixed value vector.

Three policies, by input (--features, default all three): the comment's embedding ("text"), the prior score ("prior":
the LLM's toxicity score before any steering, i.e. at alpha 0; known without human labels, from one run), or both
("text+prior"); standardised on the training comments. Caveat for "prior": here it comes from the same answers that
score alpha 0, so it shares their sampling noise, which a fresh run would not; its validation error is somewhat
optimistic.

Each policy is one linear layer and a softmax over the alphas: --model softmax (default) gives every alpha its own
linear score; --model bell gives each comment a centre and a width, and shapes the probabilities as a bell curve over
the ordered alphas (BellPolicy), so neighbouring alphas share what they learn. It is trained on the
expected distance, sum over alphas of p(alpha) * |LLM score at alpha - human score|, so it directly minimises the gap
(no "best alpha" labels). Data: an activation steering run's scores.csv (ethos.data.load_steering_scores) and the
comment embeddings (all-mpnet-base-v2).

Usage:
    python -m ethos.alpha_policy \\
        --scores data/processed/civil-comments/boundary_n500_train/steer_toxicity_llama3_8b_l10-25_a0.15-0.3/scores.csv \\
        --out data/experiments/civil-comments/05_activation_steering/alpha_policy_llama3_8b_boundary_n500_5alphas \\
        [--features text prior text+prior]
Evaluated by k-fold cross-validation (--folds, default 5): each fold trains on the other comments and validates on its
own, so every comment is validated once. Writes <out>/history.csv (loss per method, fold and epoch) and <out>/loss.png:
each policy's validation error per epoch (mean over folds, +-1 sd; faint dashed: train) with a star at its lowest, and
the CDFs of all comments' toxicity scores, each comment scored at the alpha its own fold's policy picks at the last
epoch (humans, no steering and the best constant alpha for comparison); and <out>/toxic_share.png: toxic vs not toxic
shares for humans, the LLM unsteered and each policy; with --model bell and the prior-score policy, also
<out>/policy_curve.png: the strength that policy aims for at each prior score (fitted on all comments).
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm, trange

from ethos.data import load_embeddings, load_steering_scores
from ethos.paths import ROOT
from ethos.steering.in_vllm import POLE_MEANINGS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(scores):
    """(X, D, S, H, options): X (comments, 768) embeddings; per comment and alpha, D = |LLM score - human score| and
    S = the LLM toxicity score; H (comments,) human toxicity scores."""
    rows, _ = load_steering_scores(scores)
    D = rows.pivot(index="id", columns="moral_value", values="distance")
    S = rows.pivot(index="id", columns="moral_value", values="llm_toxicity")
    H = rows.groupby("id").human_toxicity.first().loc[D.index]
    X = load_embeddings().loc[D.index]
    tensor = lambda df: torch.tensor(df.to_numpy(), dtype=torch.float32)
    return tensor(X), tensor(D), tensor(S), tensor(H), list(D.columns)


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

    def forward(self, X):
        out = self.linear(X)
        mu = self.alphas.abs().max() * torch.tanh(out[:, :1])
        sigma = torch.nn.functional.softplus(out[:, 1:]) + 0.02
        return -(self.alphas - mu) ** 2 / (2 * sigma ** 2)

def alpha_values(options):
    """The alpha of each option ("none" -> 0, "toxicity@+0.3" -> 0.3)."""
    return [0.0 if o == "none" else float(o.rsplit("@", 1)[1]) for o in options]


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


def alpha_style(option):
    """(label, color, linestyle) for a constant alpha: grey solid 0, blue dashed +, orange dashed -."""
    alpha = 0.0 if option == "none" else float(option.rsplit("@", 1)[1])
    name = "α 0 (no steering)" if alpha == 0 else f"constant α {alpha:+g}"
    return name, "#555" if alpha == 0 else "#2e6fac" if alpha > 0 else "#e08a2e", "-" if alpha == 0 else "--"


def ecdf(ax, values, **kw):
    x = np.sort(np.asarray(values, dtype=float))
    ax.step(np.r_[0, x, 1], np.r_[0, np.arange(1, len(x) + 1) / len(x), 1], where="post", **kw)


def plot(results, S, H, constants, path, title):
    """Left: each policy's validation error using its top alpha per epoch (mean over folds, +-1 sd band; faint dashed:
    train), a star at its lowest, against no steering and the best constant alpha (each fold's best on its training
    comments; mean of their validation errors). Right: CDFs of all comments' toxicity scores, each comment scored at the
    alpha its own fold's policy picks at the last epoch, with humans, no steering and the best constant alpha.
    results: {method: (history: one row per fold and epoch, oof: (epochs, comments) picked alpha indices)}.
    constants: {"none" / "best": (validation error, scores of all comments, name, color, linestyle)}."""
    fig, (ax, cx) = plt.subplots(1, 2, figsize=(17, 6.5))
    lows = []
    for method, (history, oof) in results.items():
        name, color = METHODS[method]
        g = history.drop(columns="fold").groupby("epoch")
        mean, sd = g.mean(), g.std().fillna(0)
        v = mean["validation, top alpha"]
        ax.plot(mean.index, v, color=color, lw=2, label=f"{name} (final {v.iloc[-1]:.3f})")
        ax.fill_between(mean.index, v - sd["validation, top alpha"], v + sd["validation, top alpha"], color=color,
                        alpha=0.12, lw=0)
        ax.plot(mean.index, mean.train, color=color, lw=0.8, ls="--", alpha=0.45)
        top = v.idxmin()
        ax.plot([top], [v[top]], "*", ms=18, color=color, mec="white", mew=1, zorder=6)
        ax.annotate(f"{v[top]:.3f} (epoch {top})", (top, v[top]), xytext=(8, -14), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold")
        lows += (v - sd["validation, top alpha"]).tolist() + mean.train.tolist()
        ecdf(cx, S.gather(1, oof[-1][:, None]).squeeze(1), color=color, lw=2.2, label=f"policy: {name}")
    ax.plot([], [], color="#999", lw=0.8, ls="--", label="train (same colours)")
    for err, _, name, color, ls in constants.values():
        ax.axhline(err, color=color, lw=1.5, ls=ls, label=f"{name} ({err:.3f})")
    ax.set_ylim(np.floor(min(lows + [c[0] for c in constants.values()]) / 0.05) * 0.05 - 0.05, 0.5)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Mean |LLM − human| toxicity score (validation, top α)", fontweight="bold")
    ax.set_title(f"Loss ({results[next(iter(results))][0].fold.nunique()}-fold cross-validation, mean ± 1 sd)",
                 loc="left", fontsize=11, fontweight="bold")

    ecdf(cx, H, color="#7a6fb0", lw=2, label="Humans")
    for _, scores, name, color, ls in constants.values():
        ecdf(cx, scores, color=color, lw=1.5, ls=ls, label=name)
    cx.set_xlim(-0.02, 1.02); cx.set_ylim(0, 1.02)
    cx.set_xlabel("Toxicity score", fontweight="bold")
    cx.set_ylabel("Cumulative share of comments", fontweight="bold")
    cx.set_title("CDF of toxicity scores (every comment, from the fold that held it out)", loc="left", fontsize=11,
                 fontweight="bold")

    for a in (ax, cx):
        for t in a.get_xticklabels() + a.get_yticklabels():
            t.set_fontweight("bold")
        a.grid(color="#e6e6e6"); a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_toxic_share(results, S, H, none, path, title):
    """Stacked bars, toxic vs not toxic: humans (share of annotator ratings that were toxic), the LLM unsteered, and
    each policy (share of the LLM's answers that said TOXIC, every comment run at the alpha its own fold's policy picks
    at the last epoch)."""
    bars = [("Humans", H.mean().item(), "#7a6fb0"), ("Unsteered\n(α 0)", S[:, none].mean().item(), "#555")]
    bars += [(f"Policy:\n{METHODS[m][0]}", S.gather(1, oof[-1][:, None]).mean().item(), METHODS[m][1])
             for m, (_, oof) in results.items()]
    fig, ax = plt.subplots(figsize=(2.3 * len(bars) + 1, 6))
    x = np.arange(len(bars))
    for i, (name, toxic, color) in enumerate(bars):
        ax.bar(i, toxic * 100, color=color, width=0.6)
        ax.bar(i, (1 - toxic) * 100, bottom=toxic * 100, color=color, alpha=0.25, width=0.6)
        ax.text(i, toxic * 50, f"{toxic:.2%}\ntoxic", ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, toxic * 100 + (1 - toxic) * 50, f"{1 - toxic:.2%}\nnot toxic", ha="center", va="center", color="#333")
    ax.axhline(50, color="#999", lw=1, ls=":")
    ax.set_xticks(x, [b[0] for b in bars])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of judgements (%)", fontweight="bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.905, "Humans: share of annotator ratings that were toxic. Unsteered and policies: share of the LLM's "
             "answers that said TOXIC\n(policies: each comment run at the α its policy picked; fold that held it out, "
             "last epoch).", fontsize=9, color="#555", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policy_curve(X, D, S, options, path, title, epochs=300, seed=0):
    """The bell-curve prior-score policy as a curve: fitted on all comments, the strength it aims for (its centre mu) at
    every prior score, with mu +- sigma shaded (how widely it spreads probability over neighbouring alphas), over each
    comment's best alpha (ties -> the weakest push)."""
    alphas = alpha_values(options)
    prior = S[:, options.index("none")]
    F = features("prior", X, S, options, torch.arange(len(X)))
    policy, _, _ = train(F, D, F, D, epochs=epochs, seed=seed, progress=False, model="bell", alphas=alphas)
    grid = torch.linspace(0, 1, 201)
    with torch.no_grad():
        out = policy.linear(((grid - prior.mean()) / (prior.std() + 1e-8))[:, None])
        mu = (policy.alphas.abs().max() * torch.tanh(out[:, 0])).numpy()
        sigma = (torch.nn.functional.softplus(out[:, 1]) + 0.02).numpy()
    a = np.array(alphas)
    tied = np.isclose(D.numpy(), D.numpy().min(axis=1, keepdims=True))
    best = np.array([a[np.flatnonzero(t)][np.abs(a[np.flatnonzero(t)]).argmin()] for t in tied])
    rng = np.random.default_rng(seed)                                            # jitter: scores and alphas are on grids
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(prior.numpy() + rng.uniform(-0.02, 0.02, len(best)), best + rng.uniform(-0.012, 0.012, len(best)), s=14,
               color="#999", alpha=0.4, lw=0, label="each comment's best α")
    top = a.max() + 0.05
    ax.fill_between(grid, np.clip(mu - sigma, -top, top), np.clip(mu + sigma, -top, top), color="#2f9e78", alpha=0.18,
                    lw=0, label="policy's spread (μ ± σ)")
    ax.plot(grid, mu, color="#2f9e78", lw=2.5, label="policy: the α it aims for (μ)")
    ax.axhline(0, color="#bbb", lw=1, ls=":")
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-top, top)
    # which way each half steers: the value's poles (e.g. toxicity: + toward non-toxic posting, - toward toxic posting)
    value = next(o.rsplit("@", 1)[0].removeprefix("not_") for o in options if o != "none")
    plus, minus = POLE_MEANINGS.get(value, (value, f"away from {value}"))
    ax.axhspan(0, top, color="#2e6fac", alpha=0.04, zorder=0)
    ax.axhspan(-top, 0, color="#e08a2e", alpha=0.05, zorder=0)
    up, down = (("lower TOXIC share (toward non-toxic)", "raise TOXIC share (toward toxic)") if value == "toxicity"
                else (f"toward {plus}", f"toward {minus}"))
    ax.text(1.03, top - 0.015, up, ha="right", va="top", fontsize=10, color="#2e6fac", fontweight="bold")
    ax.text(1.03, -top + 0.015, down, ha="right", va="bottom", fontsize=10, color="#c06a1a", fontweight="bold")
    ax.set_xlabel("Prior score (the LLM's TOXIC share with no steering)", fontweight="bold")
    ax.set_ylabel("Steering strength α", fontweight="bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.grid(color="#e6e6e6"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


METHODS = {"text": ("comment embedding", "#d1495b"), "prior": ("prior score", "#2f9e78"),
           "text+prior": ("comment embedding + prior score", "#2e6fac")}


def features(kind, X, S, options, train_idx):
    """The policy's inputs: the embedding ("text"), the prior score ("prior": the LLM's score at alpha 0) or both
    ("text+prior"), each column standardised with the training comments' mean and sd."""
    prior = S[:, [options.index("none")]]
    F = torch.cat({"text": [X], "prior": [prior], "text+prior": [X, prior]}[kind], dim=1)
    mu, sd = F[train_idx].mean(dim=0), F[train_idx].std(dim=0) + 1e-8
    return (F - mu) / sd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", type=Path, required=True, help="an activation steering run's scores.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=["softmax", "bell"], default="softmax",
                   help="softmax: one linear score per alpha; bell: probabilities shaped as a bell curve over the ordered "
                        "alphas (a centre and a width per comment)")
    p.add_argument("--features", nargs="+", choices=list(METHODS), default=list(METHODS),
                   help="which policies to train: comment embedding, prior score, or both (default: all three)")
    args = p.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    X, D, S, H, options = load(args.scores)
    none = options.index("none")
    folds = list(KFold(args.folds, shuffle=True, random_state=args.seed).split(X))
    best_scores, best_errs, best_choice = torch.zeros(len(X)), [], []
    for tr, va in folds:                                                   # best constant alpha per fold, on its training
        k = int(D[tr].mean(dim=0).argmin())
        best_choice.append(options[k]); best_errs.append(D[va, k].mean().item()); best_scores[va] = S[va, k]

    results = {}
    with tqdm(total=len(args.features) * args.folds, desc="training", unit="fold") as bar:
        for method in args.features:
            histories, oof = [], torch.zeros(args.epochs, len(X), dtype=torch.long)   # each comment's pick, own fold
            for f, (tr, va) in enumerate(folds, start=1):
                tr, va = torch.tensor(tr), torch.tensor(va)
                F = features(method, X, S, options, tr)
                _, history, picks = train(F[tr], D[tr], F[va], D[va], epochs=args.epochs, seed=args.seed, progress=False,
                                          model=args.model, alphas=alpha_values(options))
                histories.append(history.assign(fold=f))
                oof[:, va] = torch.stack(picks)
                bar.update()
            results[method] = (pd.concat(histories, ignore_index=True), oof)
    pd.concat([h.assign(method=m) for m, (h, _) in results.items()], ignore_index=True).to_csv(out / "history.csv",
                                                                                                index=False)
    chosen = pd.Series(best_choice).value_counts()
    label = lambda o: "0 (no steering)" if o == "none" else o.rsplit("@", 1)[1]
    best_name = "best constant α " + (label(chosen.index[0]) if len(chosen) == 1 else
                                       " / ".join(f"{label(o)} ({n} folds)" for o, n in chosen.items()))
    constants = {"none": (D[:, none].mean().item(), S[:, none], "α 0 (no steering)", "#555", "-"),
                 "best": (float(np.mean(best_errs)), best_scores, best_name, "#e08a2e", "--")}
    plot(results, S, H, constants, out / "loss.png",
         f"Steering Strength Policy: {len(X)} comments, {args.folds}-fold cross-validation")

    plot_toxic_share(results, S, H, none, out / "toxic_share.png", f"Toxic vs Not Toxic: {len(X)} comments")
    curve = args.model == "bell" and "prior" in args.features
    if curve:
        plot_policy_curve(X, D, S, options, out / "policy_curve.png",
                          "Steering Strength Policy: What It Picks by Prior Score", epochs=args.epochs, seed=args.seed)
    print(f"validation error using the top alpha (mean over folds); {best_name}: {np.mean(best_errs):.4f}, "
          f"alpha 0: {D[:, none].mean().item():.4f}")
    for method, (history, _) in results.items():
        v = history.groupby("epoch")["validation, top alpha"].mean()
        print(f"  {METHODS[method][0]:32s} final {v.iloc[-1]:.4f}   lowest {v.min():.4f} at epoch {v.idxmin()}")
    print(f"wrote {args.out}/history.csv, loss.png, toxic_share.png" + (" and policy_curve.png" if curve else ""))


if __name__ == "__main__":
    main()
