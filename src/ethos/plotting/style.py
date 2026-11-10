"""Shared matplotlib style and colors for every figure."""
SURFACE, INK, INK2, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

STYLE = {
    "font.family": ["Helvetica Neue", "Arial", "sans-serif"], "font.size": 11, "figure.dpi": 120,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
}
