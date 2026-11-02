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


def plot_decision_counts(j, judges, versions, path, suptitle="Decisions on the original vs. each context",
                         xlabel="Number of comments"):
    """Allow/remove split per version as horizontal stacked bars, one panel per judge. A dashed line marks the
    original's split; context rows note their change in remove count vs. the original."""
    n = j[j.version == "original"].groupby("judge").size().max()   # texts per version (pooled judges: more)
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
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13, fontweight="bold")
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


def pool_base_dmp(j):
    """Rows of the LLM judges, relabeled "base judges" / "DMP judges" (e.g. qwen3:8b-base + gpt-6-luna-base)."""
    llm = j[j.judge.str.endswith(("-base", "-dmp"))]
    return llm.assign(judge=np.where(llm.judge.str.endswith("-dmp"), "DMP judges", "base judges"))


def plot_net_shift(table, judges, contexts, path):
    """Net change in the share of comments removed (variation - original), one row per judge, one dot per context."""
    t = table
    y = np.arange(len(judges))
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(judges) + 2))
    for k, c in enumerate(contexts):
        d = t[t.context == c].set_index("judge").reindex(judges)
        offset = (k - (len(contexts) - 1) / 2) * 0.22
        ax.hlines(y + offset, 0, d.shift_pct, color=CONTEXT_COLORS[c], lw=1.5)
        ax.scatter(d.shift_pct, y + offset, s=40, color=CONTEXT_COLORS[c], edgecolor=SURFACE, lw=1, zorder=3, label=c)
    ax.axvline(0, color=INK, lw=1)
    lim = max(8, t.shift_pct.abs().max() * 1.2)
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if round(v, 6) else "0%")
    ax.set_xlabel("Change in comments removed\n"
                  r"$P(\mathrm{remove} \mid \mathrm{context}) - P(\mathrm{remove} \mid \mathrm{original})$")
    ax.set_yticks(y, judges)
    ax.invert_yaxis()
    ax.grid(True, axis="x"); ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncols=len(contexts), labelcolor=INK2)
    title(ax, "How much does added context shift each judge?",
          f"Share of {int(t.n.max())} comments removed · +: removes more with context · −: removes less with context")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_context(j, diffs, judges, contexts, fig_dir, table=None):
    """Save decisions_by_version.png, decisions_by_version_pooled.png (base vs DMP judges, models pooled),
    context_shift.png (from `table`, see eval_context.shift_table) and detoxify_score_shift.png (if Detoxify
    scores exist)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    versions = ["original", *contexts]
    with plt.rc_context(STYLE):
        plot_decision_counts(j, judges, versions, fig_dir / "decisions_by_version.png")
        pooled = pool_base_dmp(j)
        if not pooled.empty:
            plot_decision_counts(pooled, ["base judges", "DMP judges"], versions,
                                 fig_dir / "decisions_by_version_pooled.png",
                                 suptitle="Base vs DMP judges per context (models pooled)",
                                 xlabel="Number of decisions (each comment once per model)")
        if table is not None:
            plot_net_shift(table, judges, contexts, fig_dir / "context_shift.png")
        if diffs is not None:
            plot_score_shift(diffs, contexts, fig_dir / "detoxify_score_shift.png")
