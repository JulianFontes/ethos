"""Steering figures.

  plot_steering_layers  which decoder layers the steering vector is added to
  plot_alpha_sweep      which steering strengths work: TOXIC rate, valid answers and rationale length per alpha
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from ethos.plotting.style import GRID, INK, INK2, STYLE

STEERED, OTHER = "#2a78d6", "#e4e3df"
DOT = "#b8b7b0"


def _cell(ax, x, y, color, h=0.7):
    ax.add_patch(FancyBboxPatch((x + 0.08, y), 0.84, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=color, linewidth=0))


def plot_steering_layers(rows, path, title="Where PCA steering is applied (decoder layers)"):
    """Save one row of decoder layers per model, steered layers in blue, input on the left.
    rows: [(model name, number of decoder layers, steered layers (iterable of ints), note), ...]"""
    width = max(n for _, n, _, _ in rows)
    with plt.rc_context(STYLE | {"axes.grid": False}):
        fig, ax = plt.subplots(figsize=(4 + 0.21 * width, 1.6 + 0.75 * len(rows)))
        for i, (name, n, steered, note) in enumerate(rows):
            y, steered = len(rows) - 1 - i, sorted(steered)
            for layer in range(n):
                _cell(ax, layer, y + 0.15, STEERED if layer in steered else OTHER)
            ticks = {0, n - 1, *([steered[0], steered[-1]] if steered else [])}
            for layer in ticks:
                ax.text(layer + 0.5, y + 0.03, str(layer), ha="center", va="top", fontsize=7.5, color=INK2)
            ax.text(-0.6, y + 0.5, name, ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
            ax.text(width + 0.6, y + 0.5, f"{_ranges(steered)} of 0–{n - 1}" + (f"\n{note}" if note else ""),
                    ha="left", va="center", fontsize=8.5, color=INK2)
        top = len(rows) + 0.15
        ax.annotate("", xy=(width, top), xytext=(0, top), arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
        ax.text(0, top + 0.15, "input (embeddings)", ha="left", va="bottom", fontsize=8, color=INK2)
        ax.text(width, top + 0.15, "output (next token)", ha="right", va="bottom", fontsize=8, color=INK2)
        _cell(ax, 0, -0.75, STEERED, h=0.4)
        ax.text(1.2, -0.55, "steered: α × direction added to this layer's output", va="center", fontsize=8.5,
                color=INK)
        _cell(ax, 20, -0.75, OTHER, h=0.4)
        ax.text(21.2, -0.55, "unchanged", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(-6.5, width + 9)
        ax.set_ylim(-0.95, len(rows) + 0.7)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _ranges(layers):
    """[7, 8, 9, 12] -> "layers 7–9, 12"."""
    if not layers:
        return "no layers"
    parts, start = [], layers[0]
    for a, b in zip(layers, layers[1:] + [None]):
        if b != a + 1:
            parts.append(f"{start}–{a}" if a != start else str(a))
            start = b
    return "layers " + ", ".join(parts)


def plot_alpha_sweep(summary, per_comment, path, title):
    """Save three panels sharing the x-axis (steering strength, unsteered in the middle):
      1. share of TOXIC answers: one grey dot per comment, the mean in blue, the unsteered mean as a dashed line
      2. valid answers: answers / (answers + failed attempts), e.g. invalid JSON from over-steered text
      3. median rationale length in words (over-steered text tends to collapse or ramble)
    summary: one row per strength, sorted, with columns label, toxic, valid, words, unsteered (bool).
    per_comment: columns label, toxic (one row per comment and strength)."""
    x = np.arange(len(summary))
    pos = dict(zip(summary.label, x))
    base = summary[summary.unsteered].iloc[0]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True, height_ratios=[2.2, 1, 1])
        ax = axes[0]
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(per_comment))
        ax.scatter(per_comment.label.map(pos) + jitter, per_comment.toxic, s=14, color=DOT, linewidth=0, zorder=2)
        ax.axhline(base.toxic, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(x[-1] + 0.35, base.toxic, "unsteered", va="center", fontsize=9, color=INK2)
        ax.plot(x, summary.toxic, color=STEERED, lw=2, marker="o", ms=7, zorder=3)
        ax.set_ylim(-0.04, 1.04)
        ax.set_ylabel("Share TOXIC")
        ax.text(0.0, 1.02, "grey: each comment (10 answers)   blue: mean over comments", transform=ax.transAxes,
                fontsize=9, color=INK2, va="bottom")
        panels = [(axes[1], summary.valid, "Valid answers", (0, 1.04)),
                  (axes[2], summary.words, "Rationale words\n(median)", (0, None))]
        for ax, y, label, lim in panels:
            ax.plot(x, y, color=STEERED, lw=2, marker="o", ms=6)
            ax.axhline(y[summary.unsteered].iloc[0], color=INK2, lw=1, ls=(0, (4, 3)))
            ax.set_ylim(*lim)
            ax.set_ylabel(label)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        for ax in axes:
            ax.axvline(pos[base.label], color=GRID, lw=6, zorder=0)
        axes[2].set_xticks(x, summary.label)
        axes[2].set_xlabel("Steering strength: fraction of the hidden-state norm (α)  —  negative = away from the value")
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
