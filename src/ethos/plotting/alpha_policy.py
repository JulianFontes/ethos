"""Plots for the steering-strength policy (ethos.alpha_policy), drawn from its cross-validation results.

results: {method: (history, oof)} from ethos.alpha_policy.cross_validate (labels and colours: style); oof[-1] are each comment's picks at the last
epoch, from the fold that held it out. S, D, H: the LLM scores, distances and human scores from ethos.alpha_policy.load.
"""
import matplotlib
import numpy as np

from ethos.steering.in_vllm import POLE_MEANINGS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = {"text": ("comment embedding", "#d1495b"), "prior": ("prior score", "#2f9e78"),
           "text+prior": ("comment embedding + prior score", "#2e6fac"),
           "rationale+prior": ("rationale embedding + prior score", "#c9a227"),
           "text+rationale+prior": ("comment + rationale embeddings + prior score", "#e377c2")}


def style(key):
    """(label, colour) of a policy in `results`: a features entry (METHODS), or "<value> only: <features>" for the
    single-value baseline (scripts/experiments/alpha_policy.py, `baseline:`), in a lighter shade of the same colour."""
    if " only: " not in key:
        return METHODS[key]
    value, method = key.split(" only: ")
    name, color = METHODS[method]
    light = matplotlib.colors.to_hex([c + (1 - c) * 0.5 for c in matplotlib.colors.to_rgb(color)])
    return f"{name} ({value} only)", light


def ecdf(ax, values, **kw):
    x = np.sort(np.asarray(values, dtype=float))
    ax.step(np.r_[0, x, 1], np.r_[0, np.arange(1, len(x) + 1) / len(x), 1], where="post", **kw)


def plot_loss(results, S, H, constants, path, title):
    """Left: each policy's validation error using its top alpha per epoch (mean over folds, +-1 sd band; faint dashed:
    train), a star at its lowest, against no steering and the best constant alpha (each fold's best on its training
    comments; mean of their validation errors). Right: CDFs of all comments' toxicity scores, each comment scored at the
    alpha its own fold's policy picks at the last epoch, with humans, no steering and the best constant alpha.
    constants: {"none" / "best": (validation error, scores of all comments, name, color, linestyle)}."""
    fig, (ax, cx) = plt.subplots(1, 2, figsize=(17, 6.5))
    lows = []
    for i, (method, (history, oof)) in enumerate(results.items()):
        name, color = style(method)
        g = history.drop(columns="fold").groupby("epoch")
        mean, sd = g.mean(), g.std().fillna(0)
        v = mean["validation, top alpha"]
        ax.plot(mean.index, v, color=color, lw=2, label=f"{name} (final {v.iloc[-1]:.3f})")
        ax.fill_between(mean.index, v - sd["validation, top alpha"], v + sd["validation, top alpha"], color=color,
                        alpha=0.12, lw=0)
        ax.plot(mean.index, mean.train, color=color, lw=0.8, ls="--", alpha=0.45)
        top = v.idxmin()
        ax.plot([top], [v[top]], "*", ms=18, color=color, mec="white", mew=1, zorder=6)
        ax.annotate(f"{v[top]:.3f} (epoch {top})", (top, v[top]), xytext=(8, -14 - 13 * i), textcoords="offset points",
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
        long = max(len(label) for label in a.get_legend_handles_labels()[1]) > 55     # long labels: one column
        a.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=1 if long else 2)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_toxic_share(results, S, H, none, path, title):
    """Stacked bars, toxic vs not toxic: humans (share of annotator ratings that were toxic), the LLM unsteered, and
    each policy (share of the LLM's answers that said TOXIC, every comment run at the alpha its own fold's policy picks
    at the last epoch)."""
    bars = [("Humans", H.mean().item(), "#7a6fb0"), ("Unsteered\n(α 0)", S[:, none].mean().item(), "#555")]
    bars += [(f"Policy:\n{style(m)[0]}", S.gather(1, oof[-1][:, None]).mean().item(), style(m)[1])
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


def plot_toxic_share_by_band(results, S, H, none, bands, path, title):
    """toxic_share.png split by consensus band, in one plot: per band, stacked toxic / not toxic bars for humans, the
    LLM unsteered and each policy (picks as in plot_toxic_share). bands: each comment's consensus bin (same order)."""
    bars = [("Humans", H.numpy(), "#7a6fb0"), ("Unsteered (α 0)", S[:, none].numpy(), "#555")]
    bars += [(f"Policy: {style(m)[0]}", S.gather(1, oof[-1][:, None]).squeeze(1).numpy(), style(m)[1])
             for m, (_, oof) in results.items()]
    bands = np.asarray(bands)
    labels = sorted(set(bands), key=lambda b: float(b.split("-")[0]))
    w, mid = 0.85 / len(bars), (len(bars) - 1) / 2
    fig, ax = plt.subplots(figsize=(3.8 * len(labels) + 1, 6.5))
    for j, band in enumerate(labels):
        for k, (name, values, color) in enumerate(bars):
            toxic, x = values[bands == band].mean(), j + (k - mid) * w
            ax.bar(x, toxic * 100, w * 0.92, color=color, label=name if j == 0 else None)
            ax.bar(x, (1 - toxic) * 100, w * 0.92, bottom=toxic * 100, color=color, alpha=0.25)
            ax.text(x, toxic * 50, f"{toxic:.1%}", ha="center", va="center", color="white", fontweight="bold",
                    fontsize=8, rotation=90)
            ax.text(x, toxic * 100 + (1 - toxic) * 50, f"{1 - toxic:.1%}", ha="center", va="center", color="#333",
                    fontsize=8, rotation=90)
    ax.axhline(50, color="#999", lw=1, ls=":")
    ax.set_xticks(range(len(labels)), [f"consensus {b.replace('-', '–')}\n({(bands == b).sum()} comments)"
                                       for b in labels])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of judgements (%): toxic (solid) / not toxic (light)", fontweight="bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=len(bars), loc="upper left", bbox_to_anchor=(0, -0.13))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.905, "Humans: share of annotator ratings that were toxic. Unsteered and policies: share of the "
             "LLM's answers that said TOXIC (policies: each comment run at the α its policy picked; fold that held it "
             "out, last epoch).", fontsize=9, color="#555", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_by_band(results, D, none, bands, path, title, n_boot=2000, seed=0):
    """Average |LLM - human| per consensus band, with 95% bootstrap intervals (resampling comments): the LLM unsteered
    and each policy (every comment at the alpha its own fold's policy picks, last epoch)."""
    lines = [("Unsteered (α 0)", D[:, none].numpy(), "#555")]
    lines += [(f"Policy: {style(m)[0]}", D.gather(1, oof[-1][:, None]).squeeze(1).numpy(), style(m)[1])
              for m, (_, oof) in results.items()]
    bands = np.asarray(bands)
    labels = sorted(set(bands), key=lambda b: float(b.split("-")[0]))
    rng = np.random.default_rng(seed)
    offsets = np.linspace(-0.26, 0.26, len(lines))
    fig, ax = plt.subplots(figsize=(13, 7))
    for (name, err, color), off in zip(lines, offsets):
        mean, lo, hi = [], [], []
        for band in labels:
            e = err[bands == band]
            boot = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(n_boot)]
            mean.append(e.mean()); lo.append(np.percentile(boot, 2.5)); hi.append(np.percentile(boot, 97.5))
        mean, lo, hi = map(np.array, (mean, lo, hi))
        ax.errorbar(np.arange(len(labels)) + off, mean, yerr=[mean - lo, hi - mean], color=color, lw=2, marker="o",
                    ms=7, mec="white", mew=1.5, capsize=4, capthick=2, label=name)
    ax.set_xticks(range(len(labels)), [f"{b.replace('-', '–')}\n(n = {(bands == b).sum()})" for b in labels])
    ax.set_ylim(0, None)
    ax.set_xlabel("Human annotator consensus", fontweight="bold")
    ax.set_ylabel("Average |LLM − human| toxicity score", fontweight="bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.925, f"Bars = 95% bootstrap interval ({n_boot:,} resamples of comments). 0 = the LLM matches the "
             "annotators' score exactly. Policies: held out (each comment from the fold that did not train on it).",
             fontsize=9, color="#555", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policy_curve(curve, D, S, options, path, title, seed=0):
    """The bell-curve prior-score policy as a curve (curve = (grid, mu, sigma) from ethos.alpha_policy.prior_curve):
    the strength it aims for (its centre mu) at every prior score, with mu +- sigma shaded (how widely it spreads
    probability over neighbouring alphas), over each comment's best alpha (ties -> the weakest push)."""
    grid, mu, sigma = curve
    a = np.array([0.0 if o == "none" else float(o.rsplit("@", 1)[1]) for o in options])
    prior = S[:, options.index("none")]
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
