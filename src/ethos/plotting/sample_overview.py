"""Overview figures for a sample built by scripts/extract_sample.py.

Ported from notebooks/02_sample_overview.ipynb.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

BIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SUBTYPES = ["insult", "obscene", "identity_attack", "threat", "sexual_explicit", "severe_toxicity"]

STYLE = {
    "font.family": ["Helvetica Neue", "Arial", "sans-serif"], "font.size": 11, "figure.dpi": 120,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
}


def title(ax, text, sub=None):
    ax.set_title(text, loc="left", color=INK, fontsize=13, fontweight="bold", pad=22 if sub else 12)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=9.5)


def plot_sample(s, labels, cfg, fig_dir):
    """Save the sample overview figures (PNGs) to fig_dir.

    `s` is the sample with a `toxicity_bin` column, `labels` the bin labels in order,
    and `cfg` the sample config (used for subtitles).
    """
    fig_dir.mkdir(exist_ok=True)
    # Fall back to tab10 if there are more bins than the hand-picked palette covers.
    palette = BIN_COLORS if len(labels) <= len(BIN_COLORS) else [matplotlib.colors.to_hex(c) for c in plt.cm.tab10.colors]
    colors = {b: palette[i % len(palette)] for i, b in enumerate(labels)}
    s = s.assign(toxicity_bin=s.toxicity_bin.astype(str))
    scope = "top-level test comments" if cfg.get("top_level_only", False) else "test comments"
    counts_per_bin = s.toxicity_bin.value_counts().reindex(labels)
    per_bin = f"{counts_per_bin.iloc[0]} per bin" if counts_per_bin.nunique() == 1 else f"{len(labels)} bins"

    with plt.rc_context(STYLE):
        # Human toxicity scores
        fig, ax = plt.subplots(figsize=(9, 4.5))
        edges = np.arange(0, 1.0001, 0.05)
        bottom = np.zeros(len(edges) - 1)
        for b in labels:
            h, _ = np.histogram(s.loc[s.toxicity_bin == b, "toxicity"], bins=edges)
            ax.bar(edges[:-1], h, width=0.05, align="edge", bottom=bottom, color=colors[b],
                   edgecolor=SURFACE, linewidth=2, label=f"{b} (n={(s.toxicity_bin == b).sum()})")
            bottom += h
        top = bottom.max()
        ax.vlines(0.5, 0, top * 1.08, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(0.505, top * 1.08, "0.5 remove threshold", color=INK2, fontsize=9, va="top")
        ax.set_ylim(0, top * 1.4); ax.set_xlim(0, 1); ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.set_xlabel("Human toxicity score (share of annotators who said toxic)")
        ax.set_ylabel("Comments")
        ax.legend(title="Toxicity bin", title_fontsize=9.5, fontsize=9.5, loc="upper left",
                  ncols=min(len(labels), 4), labelcolor=INK2)
        title(ax, "Human toxicity scores in the sample", f"{len(s)} {scope}, {per_bin}")
        fig.tight_layout()
        fig.savefig(fig_dir / "toxicity_scores.png", dpi=200)
        plt.close(fig)

        # Annotators per comment
        fig, ax = plt.subplots(figsize=(9, 4.5))
        counts = s.toxicity_annotator_count
        lo = min(cfg["min_annotators"], counts.min())
        log_edges = np.logspace(np.log10(max(lo, 1)), np.log10(counts.max() * 1.05), 25)
        ax.hist(counts, bins=log_edges, color=BIN_COLORS[0], edgecolor=SURFACE, linewidth=2)
        ax.set_xscale("log")
        ticks = [t for t in [10, 20, 30, 50, 100, 200, 500, 1000, 1500, 2000, 5000] if lo <= t <= counts.max() * 1.05]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(which="minor", length=0)
        med = counts.median()
        ax.axvline(med, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(med * 0.97, ax.get_ylim()[1] * 0.95, f"median {med:.0f}", color=INK2, fontsize=9, va="top", ha="right")
        ax.set_xlabel("Toxicity annotators per comment (log scale)")
        ax.set_ylabel("Comments")
        title(ax, "Annotators per comment",
              f"Minimum {counts.min()}, maximum {counts.max():,}; config lower bound is {cfg['min_annotators']}")
        fig.tight_layout()
        fig.savefig(fig_dir / "annotator_counts.png", dpi=200)
        plt.close(fig)

        # Comment length
        fig, ax = plt.subplots(figsize=(9, 4.5))
        lengths = s.comment_text.str.len()
        rng = np.random.default_rng(0)
        for i, b in enumerate(labels):
            y = lengths[s.toxicity_bin == b]
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=36, color=colors[b],
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.hlines(y.median(), i - 0.28, i + 0.28, color=INK, lw=2, zorder=4)
            ax.text(i + 0.31, y.median(), f"{y.median():.0f}", color=INK2, fontsize=9, va="center")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel("Toxicity bin")
        ax.set_ylabel("Comment length (characters)")
        title(ax, "Comment length by toxicity bin", "Each dot is a comment; black bars mark the median")
        fig.tight_layout()
        fig.savefig(fig_dir / "comment_length.png", dpi=200)
        plt.close(fig)

        # Toxicity subtypes
        means = s.groupby("toxicity_bin")[SUBTYPES].mean().reindex(labels)
        fig, ax = plt.subplots(figsize=(9, 4))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
        im = ax.imshow(means.T.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (j, i), v in np.ndenumerate(means.T.values):
            ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=9.5, color="#ffffff" if v > 0.45 else INK)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(SUBTYPES)), [t.replace("_", " ") for t in SUBTYPES])
        ax.set_xlabel("Toxicity bin")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelcolor=INK2)
        title(ax, "Toxicity subtypes by bin", "Mean share of annotators who flagged each subtype")
        fig.tight_layout()
        fig.savefig(fig_dir / "subtype_heatmap.png", dpi=200)
        plt.close(fig)
