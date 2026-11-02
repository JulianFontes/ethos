"""Kaleido value figures: how many judge rationales mention each value."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ethos.plotting.alignment import JUDGE_COLORS
from ethos.plotting.sample_overview import STYLE, INK2, title

COLOR = "#2a78d6"


def judge_colors(judges):
    """Same colors as removal_gap.png: one per model, shared by its base and DMP judges."""
    models = list(dict.fromkeys(j.removesuffix("-dmp").removesuffix("-base") for j in judges))
    return {j: JUDGE_COLORS[models.index(j.removesuffix("-dmp").removesuffix("-base")) % len(JUDGE_COLORS)]
            for j in judges}


def _labels(ax, ys, values, total, fontsize):
    """Count at the end of each bar."""
    for y, v in zip(ys, values):
        ax.text(v + 0.01 * total, y, f"{v}", va="center", fontsize=fontsize, color=INK2)


def plot_overall(counts, n, path):
    """Top values across all judges."""
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(counts) + 1.5))
    ax.barh(counts.index, counts.values, color=COLOR)
    _labels(ax, range(len(counts)), counts.values, n, 8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, n * 1.08)
    ax.set_xlabel("Rationales Mentioning the Value")
    ax.grid(axis="y", visible=False)
    title(ax, "Most common Kaleido values", f"Top {len(counts)} values · {n} rationales, all judges and versions")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_by_judge(by_judge, n, colors, path):
    """The same values per judge: one row per model, base and DMP side by side; DMP hatched."""
    models = list(dict.fromkeys(j.removesuffix("-dmp").removesuffix("-base") for j in by_judge.columns))
    kinds = ["base", "dmp"]
    total = max(n[j] for j in by_judge.columns)
    fig, axes = plt.subplots(len(models), len(kinds), figsize=(8, 0.3 * len(by_judge) * len(models) + 1.5),
                             sharex=True, sharey=True, squeeze=False)
    for row, model in zip(axes, models):
        for ax, kind in zip(row, kinds):
            judge = f"{model}-{kind}"
            if judge not in by_judge:
                ax.axis("off")
                continue
            ax.barh(by_judge.index, by_judge[judge], color=colors[judge], hatch="///" if kind == "dmp" else None,
                    edgecolor="white", lw=0)
            _labels(ax, range(len(by_judge)), by_judge[judge], total, 7.5)
            ax.set_title(f"{judge} ({n[judge]} rationales)", fontsize=10, color=INK2)
            ax.set_xlim(0, total * 1.12)
            ax.grid(axis="y", visible=False)
    axes[0, 0].invert_yaxis()
    fig.supxlabel("Rationales Mentioning the Value", fontsize=10, color=INK2)
    fig.suptitle("Kaleido values by judge", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_base_vs_dmp(by_kind, n, path):
    """Paired bars per value: all base judges (solid) vs all DMP judges (lighter)."""
    y = np.arange(len(by_kind))
    total = max(n["base"], n["dmp"])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(by_kind) + 1.5))
    bar, gap = 0.34, 0.06   # bar height and space between the base and DMP bars
    for k, (kind, alpha) in enumerate([("base", 1), ("dmp", 0.45)]):
        offset = (k - 0.5) * (bar + gap)
        ax.barh(y + offset, by_kind[kind], bar, color=COLOR, alpha=alpha)
        _labels(ax, y + offset, by_kind[kind], total, 7.5)
    ax.set_yticks(y, by_kind.index)
    ax.invert_yaxis()
    ax.set_xlim(0, total * 1.08)
    ax.set_xlabel("Rationales Mentioning the Value")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=COLOR, label="Base judges"), Patch(color=COLOR, alpha=0.45, label="DMP judges")],
              loc="lower right", frameon=False)
    title(ax, "Kaleido values: base vs DMP", "Each group pools qwen3:8b and gpt-6-luna")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_values(counts, by_judge, by_kind, n, colors, fig_dir):
    """Save kaleido_values_overall.png, kaleido_values_by_judge.png and kaleido_values_base_vs_dmp.png.

    `n` is the number of rationales: "all", per judge, and per kind ("base", "dmp").
    """
    with plt.rc_context(STYLE):
        plot_overall(counts, n["all"], fig_dir / "kaleido_values_overall.png")
        plot_by_judge(by_judge, n, colors, fig_dir / "kaleido_values_by_judge.png")
        plot_base_vs_dmp(by_kind, n, fig_dir / "kaleido_values_base_vs_dmp.png")
