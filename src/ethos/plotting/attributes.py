"""Attribute figures: how often annotators selected each toxicity attribute (insult, obscene, ...) per comment."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from ethos.plotting.sample_overview import STYLE, INK, INK2, SURFACE

# column in the Civil Comments data -> (panel title, color)
ATTRIBUTES = {"insult": ("Insult", "#2a78d6"), "obscene": ("Obscene", "#eb6834"),
              "identity_attack": ("Identity attack", "#1baf7a"), "threat": ("Threat", "#c98a00"),
              "sexual_explicit": ("Sexually explicit", "#e87ba4")}
# x-axis column -> (label, limits)
X_AXES = {"toxicity": ("Toxicity Score", (0, 1)), "consensus": ("Consensus", (0.5, 1))}


def plot_attributes_vs_toxicity(df, path, scope, x_cols=("toxicity",)):
    """Save one panel per attribute: each comment's score on the x-axis against the proportion of its annotators who
    selected that attribute. One dot per comment, nothing grouped or averaged.

    df: comments with the x columns and the attribute columns; scope: subtitle text naming the comments.
    x_cols: one row of panels per x-axis, from X_AXES ("toxicity", "consensus").
    """
    rows = len(x_cols)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows, len(ATTRIBUTES), figsize=(16, 3.7 + 3.6 * (rows - 1)), sharey=True,
                                 squeeze=False)
        for r, x in enumerate(x_cols):
            label, lim = X_AXES[x]
            for ax, (col, (name, color)) in zip(axes[r], ATTRIBUTES.items()):
                ax.scatter(df[x], df[col], s=3, color=color, alpha=0.08, lw=0, rasterized=True)
                if r == 0:
                    ax.set_title(name, loc="left", fontsize=11, fontweight="bold", color=INK)
                ax.set_xlim(*lim)
                ax.set_ylim(0, 1)
                ax.set_box_aspect(1)
                ax.grid(axis="both")
                ax.set_xlabel(label)
            axes[r, 0].set_ylabel("Proportion of Annotators\nWho Selected the Attribute")
        title = ("Each comment's toxicity score vs. how many annotators selected each attribute" if rows == 1 else
                 "How many annotators selected each attribute, by toxicity score (top) and consensus (bottom)")
        fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=13.5, fontweight="bold", color=INK)
        fig.text(0.01, 1 - 0.115 / rows, f"One dot per comment ({len(df):,} {scope}). Nothing grouped or averaged.",
                 fontsize=9.5, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.13 / rows), h_pad=2.5)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)


SIDES = {"Leaning toxic": "#eb6834", "Leaning clean": "#2a78d6"}   # toxicity >= 0.5 / < 0.5
GREY = "#7d7c76"


def attributes_by_bin(df):
    """Average proportion of annotators who selected each attribute, per bin.

    Returns (by_consensus, by_toxicity): by_consensus has one row per consensus bin (0.5-0.6, ..., 0.9-1.0) and side
    (leaning toxic / leaning clean); by_toxicity one row per toxicity bin (0.0-0.1, ..., 0.9-1.0). Bins are [lo, hi),
    the last one includes 1.0.
    """
    c_edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    t_edges = list(np.round(np.arange(0, 1, 0.1), 1)) + [1.0001]
    label = lambda edges: [f"{lo:.1f}–{min(hi, 1):.1f}" for lo, hi in zip(edges[:-1], edges[1:])]
    d = df.assign(cbin=pd.cut(df.consensus, c_edges, right=False, labels=label(c_edges)),
                  tbin=pd.cut(df.toxicity, t_edges, right=False, labels=label(t_edges)),
                  side=np.where(df.toxicity >= 0.5, "Leaning toxic", "Leaning clean"))
    by_consensus = d.groupby(["cbin", "side"], observed=True)[list(ATTRIBUTES)].mean()
    by_toxicity = d.groupby("tbin", observed=True)[list(ATTRIBUTES)].mean()
    return by_consensus, by_toxicity


def plot_attributes_by_bin(df, path, scope):
    """Save bar charts of the average proportion of annotators who selected each attribute: by consensus bin, split
    into comments leaning toxic / clean (top row), and by toxicity score bin (bottom row). One column per attribute."""
    by_consensus, by_toxicity = attributes_by_bin(df)
    c_labels = list(by_consensus.index.get_level_values(0).unique())
    t_labels = list(by_toxicity.index)
    two_lines = lambda labels: [b.replace("–", "–\n") for b in labels]
    pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, len(ATTRIBUTES), figsize=(14, 8), sharey=True)
        x, w = np.arange(len(c_labels)), 0.38
        for ax, (col, (name, _)) in zip(axes[0], ATTRIBUTES.items()):
            for k, (side, color) in enumerate(SIDES.items()):
                v = by_consensus.xs(side, level="side")[col].reindex(c_labels)
                pos = x - w / 2 + k * w
                ax.bar(pos, v, width=w, color=color, edgecolor=SURFACE, lw=1.5, label=side)
                for p, val in zip(pos, v):
                    ax.text(p, val + 0.015, f"{val:.0%}", ha="center", fontsize=7.5, color=INK2)
            ax.set_title(name, loc="left", fontsize=11, fontweight="bold", color=INK)
            ax.set_xticks(x, two_lines(c_labels), fontsize=8.5)
            ax.set_xlabel("Human Annotator Consensus", fontsize=9.5)
        xt = np.arange(len(t_labels))
        for ax, col in zip(axes[1], ATTRIBUTES):
            v = by_toxicity[col]
            ax.bar(xt, v, width=0.78, color=GREY, edgecolor=SURFACE, lw=1.5)
            for p, val in zip(xt, v):
                ax.text(p, val + 0.015, f"{val:.0%}", ha="center", fontsize=6.5, color=INK2)
            ax.set_xticks(xt, two_lines(t_labels), fontsize=7.5)
            ax.set_xlabel("Toxicity Score", fontsize=9.5)
        for ax in axes.flat:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(pct)
        for r in range(2):
            axes[r, 0].set_ylabel("Annotators Who Selected\nthe Attribute (average)")
        axes[0, 0].legend(loc="upper left", fontsize=9)
        fig.suptitle("How often annotators select each attribute, by consensus (top) and toxicity score (bottom)",
                     x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=INK)
        fig.text(0.01, 0.935, f"{len(df):,} {scope}. Top: leaning toxic = at least half said toxic. "
                 "Bottom: bins of 0.1 toxicity score.", fontsize=9.5, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.5)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


# (said toxic, selected the attribute) -> color: orange shades = said toxic, blue / grey = said not toxic
VOTE_COLORS = {(1, 1): "#eb6834", (1, 0): "#f5c3a6", (0, 1): "#2a78d6", (0, 0): "#d9d8d1"}
CONSENSUS_EDGES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
CONSENSUS_LABELS = ["0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9–1.0"]


def attribute_votes_by_consensus(votes, consensus):
    """Count individual annotator votes per consensus bin by (said toxic, selected the attribute), per attribute.

    votes: one row per annotator vote with `id`, `toxic` and the attribute columns (0/1);
    consensus: Series of comment consensus indexed by id. Returns a long table:
    attribute, bin, toxic, selected, votes.
    """
    bins = pd.cut(votes.id.map(consensus), CONSENSUS_EDGES, right=False, labels=CONSENSUS_LABELS)
    parts = []
    for col in ATTRIBUTES:
        n = votes.groupby([bins, votes.toxic, votes[col]], observed=True).size()
        n.index.names = ["bin", "toxic", "selected"]
        parts.append(n.rename("votes").reset_index().assign(attribute=col))
    return pd.concat(parts)[["attribute", "bin", "toxic", "selected", "votes"]]


def plot_attribute_votes_by_consensus(counts, path):
    """Save one panel per attribute: per consensus bin, a bar for votes that said toxic and one for votes that said
    not toxic, each split by whether the annotator selected the attribute, with the vote count on each segment.
    counts: the table from attribute_votes_by_consensus."""
    k = lambda v: f"{v / 1000:,.0f}k" if v >= 1000 else f"{v:,.0f}"
    x, w = np.arange(len(CONSENSUS_LABELS)), 0.38
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(ATTRIBUTES), figsize=(20, 5.2), sharey=True)
        for ax, (col, (name, _)) in zip(axes, ATTRIBUTES.items()):
            t = (counts[counts.attribute == col].pivot_table(index="bin", columns=["toxic", "selected"], values="votes",
                                                              aggfunc="sum", observed=False)
                 .reindex(CONSENSUS_LABELS).fillna(0))
            for j, tox in enumerate([1, 0]):
                pos = x - w / 2 - 0.02 + j * (w + 0.04)
                yes, no = t[(tox, 1)].to_numpy(), t[(tox, 0)].to_numpy()
                ax.bar(pos, yes, width=w, color=VOTE_COLORS[(tox, 1)], edgecolor=SURFACE, lw=1)
                ax.bar(pos, no, bottom=yes, width=w, color=VOTE_COLORS[(tox, 0)], edgecolor=SURFACE, lw=1)
                # count on every segment: inside when it is tall enough, otherwise just above it
                for p, a, b in zip(pos, yes, no):
                    for base, v, sel in ((0, a, 1), (a, b, 0)):
                        dark = (tox, sel) == (1, 1)
                        if v >= 45000:
                            ax.text(p, base + v / 2, k(v), ha="center", va="center", fontsize=7, rotation=90,
                                    color="white" if dark else INK)
                        elif sel == 1:   # small "attribute selected" segment: label just above it
                            ax.text(p, base + v + 4000, k(v), ha="center", va="bottom", fontsize=7, rotation=90,
                                    color=VOTE_COLORS[(tox, 1)], fontweight="bold")
            ax.set_title(name, loc="left", fontsize=11, fontweight="bold", color=INK)
            ax.set_xticks(x, [b.replace("–", "–\n") for b in CONSENSUS_LABELS], fontsize=8.5)
            ax.set_xlabel("Human Annotator Consensus", fontsize=9.5)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
        axes[0].set_ylabel("Annotator votes")
        fig.legend(handles=[Patch(color=VOTE_COLORS[(1, 1)], label="Toxic, attribute selected"),
                            Patch(color=VOTE_COLORS[(1, 0)], label="Toxic, attribute not selected"),
                            Patch(color=VOTE_COLORS[(0, 1)], label="Not toxic, attribute selected"),
                            Patch(color=VOTE_COLORS[(0, 0)], label="Not toxic, attribute not selected")],
                   loc="lower left", bbox_to_anchor=(0.01, -0.02), ncol=4, fontsize=9.5, frameon=False)
        fig.suptitle("Annotator Votes by Attribute and Consensus", x=0.01, ha="left", fontsize=13.5,
                     fontweight="bold", color=INK)
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
