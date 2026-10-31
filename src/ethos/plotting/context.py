"""Context-sensitivity figures: how judges' decisions shift from the original comment to each variation."""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ethos.plotting.alignment import DECISIONS
from ethos.plotting.sample_overview import STYLE, SURFACE, INK, INK2, MUTED, title

CONTEXT_COLORS = {"consequentialist": "#1baf7a", "emotional": "#eda100", "relational": "#e87ba4"}  # validated trio
VERSION_COLORS = {"original": MUTED, **CONTEXT_COLORS}                                            # gray = baseline
DIFF_COLORS = {1: "#008300", -1: "#c62828", 0: "#898781"}   # change vs. original: up, down, none


def plot_decision_counts(j, judges, versions, path):
    """Allow/remove split per version as horizontal stacked bars, one panel per judge. A dashed line marks the
    original's split; context rows note their change in remove count vs. the original."""
    n = j.id.nunique()
    fig, axes = plt.subplots(len(judges), 1, figsize=(9, 2.5 * len(judges)), sharex=True, squeeze=False)
    axes = axes[:, 0]   # always an array, even for a single judge
    y = np.arange(len(versions))
    colors = [VERSION_COLORS[v] for v in versions]
    for ax, judge in zip(np.atleast_1d(axes), judges):
        counts = (j[j.judge == judge].groupby(["version", "decision"]).size().unstack(fill_value=0)
                  .reindex(index=versions, columns=DECISIONS, fill_value=0))
        allow, remove = counts.allow.values, counts.remove.values
        ax.barh(y, allow, 0.62, **_decision_style(colors, "allow"))
        ax.barh(y, remove, 0.62, left=allow, **_decision_style(colors, "remove"))
        ax.axvline(allow[0], color=INK, lw=1, ls=(0, (3, 2)))
        for i, (a, r, diff) in enumerate(zip(allow, remove, remove - remove[0])):
            ax.text(-0.01 * n, i, f"{a}", ha="right", va="center", fontsize=8.5, color=INK2)
            count = ax.text(1.01 * n, i, f"{r}", ha="left", va="center", fontsize=8.5, color=INK2)
            if i:   # context rows: change in remove count vs. the original, right after the count
                ax.annotate(f" ({diff:+d})" if diff else " (±0)", xy=(1, 0), xycoords=count, va="bottom",
                            fontsize=8.5, color=DIFF_COLORS[np.sign(diff)])
        ax.set_yticks(y, versions)
        ax.invert_yaxis()
        ax.set_xlim(-0.08 * n, 1.12 * n)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(judge, loc="left", fontsize=11, color=INK2)
    axes[-1].set_xticks(np.linspace(0, n, 5))
    axes[-1].set_xlabel("Number of comments")
    fig.suptitle("Decisions on the original vs. each context", x=0.01, ha="left", fontsize=13, fontweight="bold")
    handles = [Patch(**_decision_style([INK2], d), label=d) for d in DECISIONS]
    handles.append(plt.Line2D([], [], color=INK, lw=1, ls=(0, (3, 2)), label="original split"))
    fig.legend(handles=handles, loc="upper right", ncols=3, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _decision_style(colors, decision):
    """Bar style: remove = solid version color, allow = light tint of it with a hatch in the full color."""
    if decision == "remove":
        return {"color": colors if len(colors) > 1 else colors[0]}
    tints = [mcolors.to_rgb(c) for c in colors]
    tints = [tuple(0.35 * a + 0.65 * b for a, b in zip(c, mcolors.to_rgb(SURFACE))) for c in tints]
    edges = colors if len(colors) > 1 else colors[0]
    return {"facecolor": tints if len(tints) > 1 else tints[0], "edgecolor": edges, "hatch": "////", "linewidth": 0}


def plot_score_shift(diffs, contexts, path):
    """Detoxify score change (variation - original) per context: box plus every comment as a dot."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rng = np.random.default_rng(0)
    data = [diffs[c].dropna().values for c in contexts]
    ax.boxplot(data, positions=range(len(contexts)), widths=0.5, showfliers=False, orientation="horizontal",
               medianprops={"color": INK, "lw": 2}, boxprops={"color": INK2}, whiskerprops={"color": INK2},
               capprops={"color": INK2})
    for i, (c, x) in enumerate(zip(contexts, data)):
        ax.scatter(x, i + rng.uniform(-0.18, 0.18, len(x)), s=28, color=CONTEXT_COLORS[c],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(contexts)), contexts)
    ax.invert_yaxis()
    ax.grid(True, axis="x"); ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Detoxify score change (less toxic ← | → more toxic)")
    title(ax, "Detoxify toxicity score shift when context is added",
          "Variation minus original, one dot per comment")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_context(j, diffs, judges, contexts, fig_dir):
    """Save decisions_by_version.png and detoxify_score_shift.png (if Detoxify scores exist)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        plot_decision_counts(j, judges, ["original", *contexts], fig_dir / "decisions_by_version.png")
        if diffs is not None:
            plot_score_shift(diffs, contexts, fig_dir / "detoxify_score_shift.png")
