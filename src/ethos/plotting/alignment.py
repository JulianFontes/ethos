"""Human-alignment figures for judgements.csv (original comments only)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ethos.plotting.sample_overview import STYLE, SURFACE, INK, INK2, title

DECISIONS = ["allow", "remove"]
COLORS = {"allow": "#2a78d6", "remove": "#eb6834"}   # validated pair (CVD + contrast)
JUDGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots, in config order


def _bars(ax, counts, judges, n, human_line=True):
    """Horizontal allow | remove bar per judge, counts at both ends; optional dashed line at the human split."""
    y = np.arange(len(judges))
    counts = counts.reindex(index=judges, columns=DECISIONS, fill_value=0)
    allow, remove = counts.allow.values, counts.remove.values
    ax.barh(y, allow, 0.62, color=COLORS["allow"])
    ax.barh(y, remove, 0.62, left=allow, color=COLORS["remove"])
    if human_line and "human" in judges:
        ax.axvline(allow[judges.index("human")], color=INK, lw=1, ls=(0, (3, 2)))
    for i, (a, r) in enumerate(zip(allow, remove)):
        ax.text(-0.01 * n, i, f"{a}", ha="right", va="center", fontsize=8.5, color=INK2)
        ax.text(1.01 * n, i, f"{r}", ha="left", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, judges)
    ax.invert_yaxis()
    ax.set_xlim(-0.08 * n, 1.08 * n)
    ax.grid(False)
    ax.spines["left"].set_visible(False)


def _legend_handles(human_label=None):
    handles = [Patch(color=COLORS[d], label=d) for d in DECISIONS]
    if human_label:
        handles.append(plt.Line2D([], [], color=INK, lw=1, ls=(0, (3, 2)), label=human_label))
    return handles


def plot_alignment(o, judges, bins, fig_dir, threshold=0.5):
    """Save decisions_overall.png, decisions_by_bin.png and removal_gap.png to fig_dir.

    `o` holds the original-version rows with columns judge, decision, toxicity_bin; `threshold` is the
    human judge's remove threshold (share of annotators), used in the subtitle.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    n = o.id.nunique()
    with plt.rc_context(STYLE):
        # Aggregate
        counts = o.groupby(["judge", "decision"]).size().unstack(fill_value=0)
        human_label = None
        if "human" in judges:
            h = counts.reindex(columns=DECISIONS, fill_value=0).loc["human"]
            human_label = f"Human allow/remove boundary ({h.allow} / {h.remove})"
        fig, ax = plt.subplots(figsize=(9, 0.6 * len(judges) + 2.2))
        _bars(ax, counts, judges, n)
        ax.set_xticks(np.linspace(0, n, 5))
        ax.set_xlabel("Number of comments")
        ax.legend(handles=_legend_handles(human_label), loc="upper center", bbox_to_anchor=(0.45, -0.22),
                  ncols=3, labelcolor=INK2, fontsize=9)
        title(ax, "Allow / remove decisions by judge",
              f"{n} original comments · Human: remove if ≥ {threshold:.0%} of annotators rated it toxic")
        fig.tight_layout()
        fig.savefig(fig_dir / "decisions_overall.png", dpi=200)
        plt.close(fig)

        # One panel per human-toxicity bin, stacked vertically
        per_bin = o.groupby("toxicity_bin").id.nunique().max()
        fig, axes = plt.subplots(len(bins), 1, figsize=(9, (0.45 * len(judges) + 0.6) * len(bins)), sharex=True,
                                 squeeze=False)
        axes = axes[:, 0]   # always an array, even for a single bin
        for ax, b in zip(np.atleast_1d(axes), bins):
            sub = o[o.toxicity_bin == b]
            _bars(ax, sub.groupby(["judge", "decision"]).size().unstack(fill_value=0), judges, per_bin, human_line=False)
            ax.set_title(f"Human toxicity {b}", loc="left", fontsize=11, color=INK2)
        axes[-1].set_xticks(np.linspace(0, per_bin, 6))
        axes[-1].set_xlabel("Number of comments")
        fig.suptitle("Allow / remove decisions by human-toxicity bin", x=0.01, ha="left",
                     fontsize=13, fontweight="bold")
        fig.legend(handles=_legend_handles(), loc="upper right", ncols=2, labelcolor=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(fig_dir / "decisions_by_bin.png", dpi=200)
        plt.close(fig)

        if "human" in judges:
            plot_removal_gap(o, [x for x in judges if x != "human"], bins, fig_dir / "removal_gap.png")


def plot_removal_gap(o, judges, bins, path):
    """Model minus human share of comments removed, per human-toxicity bin; one line per judge."""
    edges = sorted({float(e) for b in bins for e in b.split("-")})
    centers = [np.mean([float(e) for e in b.split("-")]) for b in bins]
    removed = o.assign(r=o.decision == "remove").groupby(["judge", "toxicity_bin"]).r.mean().unstack()
    removed = removed.reindex(columns=bins)
    counts = o[o.judge == "human"].groupby("toxicity_bin").size().reindex(bins, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(True, axis="both")   # x ticks sit on the bin edges, so the grid marks the bins
    ax.axhline(0, color=INK, lw=1.2, ls=":", zorder=2,
               label="0% = same as humans (above: model removes more; below: removes less)")
    # A DMP judge shares its base model's color, drawn dashed (e.g. qwen3:8b-base solid, qwen3:8b-dmp dashed).
    models = list(dict.fromkeys(j.removesuffix("-dmp").removesuffix("-base") for j in judges))
    for k, judge in enumerate(judges):
        last = k == len(judges) - 1 and len(judges) > 1   # smaller markers so an overlapping line shows through
        model = judge.removesuffix("-dmp").removesuffix("-base")
        ax.plot(centers, removed.loc[judge] - removed.loc["human"],
                color=JUDGE_COLORS[models.index(model) % len(JUDGE_COLORS)], ls="--" if judge.endswith("-dmp") else "-",
                lw=2, marker="o", ms=5 if last else 7, mec=SURFACE, mew=1.5, label=judge, zorder=3)
    for c, k in zip(centers, counts):
        ax.text(c, 1.1, f"n={k}", ha="center", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(-1.15, 1.18)
    ax.set_xticks(edges)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1], ["−100%", "−50%", "0%", "+50%", "+100%"])
    ax.set_xlabel("Human Toxicity Score")
    ax.set_ylabel("Difference in % of comments removed\n(model − human)")
    h, l = ax.get_legend_handles_labels()   # judges on one row, the 0% line on its own row below
    ax.add_artist(ax.legend(h[1:], l[1:], loc="upper center", bbox_to_anchor=(0.5, -0.2),
                            ncols=len(judges), labelcolor=INK2, fontsize=9, handlelength=3.5))
    ax.legend(h[:1], l[:1], loc="upper center", bbox_to_anchor=(0.5, -0.29), labelcolor=INK2, fontsize=9)
    title(ax, "Gap between model and human removals",
          f"{o.id.nunique()} Original Comments; n = comments per bin")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
