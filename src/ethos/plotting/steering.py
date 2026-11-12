"""Steering figures: which decoder layers the steering vector is added to."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ethos.plotting.style import INK, INK2, STYLE

STEERED, OTHER = "#2a78d6", "#e4e3df"


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
