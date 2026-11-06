"""Consensus figures: how much annotators agree, consensus = max(p, 1 - p) where p is the share voting toxic."""
import matplotlib.pyplot as plt
import numpy as np

from ethos.data import consensus
from ethos.plotting.sample_overview import STYLE, INK, INK2, GRID, SURFACE

BLUE = "#2a78d6"

def plot_consensus_distribution(p, path, scope, low=0.6):
    """Save a histogram of consensus max(p, 1 - p), with the low-consensus range shaded and the median marked."""
    c = consensus(np.asarray(p))
    med = np.median(c)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.axvspan(0.5, low, color=GRID, alpha=0.55, lw=0, zorder=0)
        ax.hist(c, bins=np.arange(0.5, 1.0001, 0.025), color=BLUE, edgecolor=SURFACE, lw=1.5)
        top = ax.get_ylim()[1]
        ax.text(0.503, top * 0.97, f"low consensus (≤ {low:g}): {(c <= low).mean():.0%} of comments",
                fontsize=8.5, color=INK2, va="top")
        ax.axvline(med, color=INK, lw=1, ls=(0, (3, 2)))
        ax.text(med + 0.004, top * 0.85, f"median {med:.2f}", fontsize=8.5, color=INK, va="top")
        ax.set_xlim(0.5, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("Consensus  max(p, 1 − p)   (0.5 = even split, 1 = unanimous)")
        ax.set_ylabel("Comments")
        ax.set_title("Distribution of annotator consensus", loc="left", color=INK, fontsize=13, fontweight="bold", pad=22)
        ax.text(0, 1.02, f"{scope} (n = {len(c):,}).", transform=ax.transAxes, fontsize=9.5, color=INK2)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_consensus_vs_toxicity(p, path, scope, low=0.6):
    """Save a scatter of consensus vs toxicity score p, with a histogram of p on top.

    p: toxicity scores (share of annotators voting toxic), one per comment.
    scope: subtitle text naming the comments, e.g. "Top-level train comments with ≥30 raters".
    low: consensus at or below this is shaded as low consensus.
    """
    p = np.asarray(p)
    lo, hi = 1 - low, low
    with plt.rc_context(STYLE):
        fig, (hx, ax) = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1, 3.2], hspace=0.06))
        hx.hist(p, bins=np.arange(0, 1.0001, 0.025), color=BLUE, edgecolor=SURFACE, lw=1)
        hx.axvspan(lo, hi, color=GRID, alpha=0.55, lw=0, zorder=0)
        hx.set_ylabel("Comments")
        hx.spines["bottom"].set_visible(False)
        ax.axvspan(lo, hi, color=GRID, alpha=0.55, lw=0)
        ax.text(lo + 0.005, 1.0, f"low consensus (consensus ≤ {low:g})", ha="left", fontsize=8.5, color=INK2)
        ax.scatter(p, consensus(p), s=14, color=BLUE, alpha=0.12, lw=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.48, 1.04)
        ax.set_xlabel("Human toxicity score p (share of raters voting toxic)")
        ax.set_ylabel("Consensus  max(p, 1 − p)")
        hx.set_title("Human consensus vs toxicity score", loc="left", color=INK, fontsize=13, fontweight="bold", pad=22)
        hx.text(0, 1.02, f"{scope} (n = {len(p):,}). Top: how many comments at each p. Bottom: each dot is a comment.",
                transform=hx.transAxes, fontsize=9.5, color=INK2)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.09)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_score_gap_by_consensus(df, judges, path, scope, edges=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), n_boot=2000, seed=0):
    """Save one panel per judge: (judge score - human score) against human consensus.

    df: one row per comment with columns `human`, `consensus` and one score column per judge in `judges`
    (a dict {column: panel title}). Dots are comments; the line is the mean gap per consensus bin with a
    95% bootstrap interval. Above 0 = the judge rates the comment more toxic than the annotators did.
    """
    rng = np.random.default_rng(seed)
    edges = np.asarray(edges, dtype=float)
    mids = (edges[:-1] + edges[1:]) / 2
    bins = np.clip(np.digitize(df.consensus, edges[1:-1]), 0, len(mids) - 1)
    gaps = {col: (df[col] - df.human).to_numpy() for col in judges}
    lo = min(g.min() for g in gaps.values())
    hi = max(g.max() for g in gaps.values())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(judges), figsize=(5.4 * len(judges), 5), sharey=True, squeeze=False)
        for ax, (col, name) in zip(axes[0], judges.items()):
            g = gaps[col]
            ax.axhline(0, color=INK2, lw=1)
            ax.scatter(df.consensus, g, s=22, color=BLUE, alpha=0.35, lw=0)
            means, cis = [], []
            for b in range(len(mids)):
                x = g[bins == b]
                boot = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
                means.append(x.mean())
                cis.append(np.percentile(boot, [2.5, 97.5]))
            cis = np.array(cis)
            ax.fill_between(mids, cis[:, 0], cis[:, 1], color=BLUE, alpha=0.18, lw=0)
            ax.plot(mids, means, color=BLUE, lw=2.2, marker="o", ms=6, markeredgecolor="white", markeredgewidth=1.5)
            ax.set_title(name, loc="left", fontsize=11.5, fontweight="bold", color=INK)
            ax.set_xlim(0.5, 1)
            ax.set_ylim(min(lo, -0.1) - 0.05, hi + 0.12)
            ax.set_xlabel("Human consensus  max(p, 1 − p)")
        axes[0, 0].set_ylabel("Judge score − human score")
        fig.suptitle("How far each judge's toxicity score is from the annotators', by annotator consensus",
                     x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
        fig.text(0.01, 0.905, f"{scope}. Dots are comments; line = mean per 0.1 consensus bin with 95% bootstrap "
                 "interval. Above 0 = judge rates it more toxic than the annotators.", fontsize=9, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


JUDGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # same categorical order as the other figures


def gap_by_bin(df, judges, edges=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), n_boot=2000, seed=0):
    """Mean signed and absolute (judge score - human score) per consensus bin, with 95% bootstrap intervals.

    Returns one row per bin and judge: bin, judge, n, then for the signed gap mean_gap, boot_mean (mean of the
    bootstrap means), ci_lo, ci_hi (2.5th / 97.5th percentiles of the bootstrap means), and the same for the
    absolute gap: mean_abs_gap, abs_boot_mean, abs_ci_lo, abs_ci_hi. Both use the same resamples of comments.
    A judge's comments without a score are left out; bins where it has none get no row. One generator is shared
    across bins and judges (in that order), so the judges passed and their order fix the draws.
    """
    rng = np.random.default_rng(seed)
    edges = np.asarray(edges, dtype=float)
    labels = [f"{lo:g}–{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    bins = np.clip(np.digitize(df.consensus, edges[1:-1]), 0, len(labels) - 1)
    rows = []
    for b, label in enumerate(labels):
        for col in judges:
            g = (df[col] - df.human).to_numpy()[bins == b]
            g = g[~np.isnan(g)]
            if len(g) == 0:
                continue
            idx = rng.integers(0, len(g), (n_boot, len(g)))
            boot, boot_abs = g[idx].mean(axis=1), np.abs(g)[idx].mean(axis=1)
            rows.append({"bin": label, "judge": col, "n": len(g),
                         "mean_gap": g.mean(), "boot_mean": boot.mean(),
                         "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                         "mean_abs_gap": np.abs(g).mean(), "abs_boot_mean": boot_abs.mean(),
                         "abs_ci_lo": np.percentile(boot_abs, 2.5), "abs_ci_hi": np.percentile(boot_abs, 97.5)})
    return rows


def plot_score_gap_by_bin(table, judges, path, scope, absolute=True):
    """Save a line plot of the bootstrap mean |judge - human| score per consensus bin, with 95% interval error bars.

    table: rows from gap_by_bin; judges: dict {judge column: legend name}. absolute=False plots the signed gap
    instead. Judges are nudged apart on x so their error bars don't overlap.
    """
    key = "abs_" if absolute else ""
    labels = list(dict.fromkeys(r["bin"] for r in table))
    x = np.arange(len(labels))
    offsets = np.linspace(-0.06, 0.06, len(judges)) if len(judges) > 1 else [0]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axhline(0, color=INK2, lw=1)
        for k, (col, name) in enumerate(judges.items()):
            rows = {r["bin"]: r for r in table if r["judge"] == col}   # only the bins this judge has results for
            have = np.array([b in rows for b in labels])
            get = lambda field: np.array([rows[b][field] for b in labels if b in rows])
            m = get(f"{key}boot_mean")
            err = np.clip([m - get(f"{key}ci_lo"), get(f"{key}ci_hi") - m], 0, None)   # rounding can give -1e-17
            color = JUDGE_COLORS[k % len(JUDGE_COLORS)]
            ax.errorbar(x[have] + offsets[k], m, yerr=err, color=color, lw=2, marker="o", ms=8, markeredgecolor=SURFACE,
                        markeredgewidth=2, elinewidth=1.5, capsize=4, capthick=1.5, label=name)
        n = [max(r["n"] for r in table if r["bin"] == b) for b in labels]
        ax.set_xticks(x, [f"{b}\n(n = {k})" for b, k in zip(labels, n)])
        ax.set_xlim(-0.4, len(labels) - 0.6)
        ax.set_xlabel("Human Annotator Consensus")
        if absolute:
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Average Absolute Difference in Toxicity Scores")
            title = "Model-Human Alignment Across Comments"
            note = "0 = judge matches the annotators' score exactly."
        else:
            ax.set_ylabel("Average Difference in Toxicity Scores")
            title = "Judges overrate toxicity most where annotators disagree"
            note = "Above 0 = judge rates comments more toxic than the annotators."
        ax.legend(loc="upper right", fontsize=9.5)
        ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=22)
        ax.text(0, 1.02, f"Bars = 95% bootstrap interval (2,000 resamples of comments). {note}",
                transform=ax.transAxes, fontsize=9, color=INK2)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
