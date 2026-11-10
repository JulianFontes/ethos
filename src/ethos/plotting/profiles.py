"""Figures for scripts/experiments/profile_signal.py runs.

Input: ratings.csv, one row per answer with `id`, `condition` ("none" = no profile, or "<name>|<weighting>") and
`toxic`. <name> is a profile file stem ("free_expression") or a single value ("Animal Welfare").

  plot_profile_shift_weighting   per-comment score shift vs. no profile, by profile and by weighting
  plot_profile_distance          per-comment distance from the human score, no profile vs. each profile
  plot_toxic_change              change in TOXIC answers vs. no profile (count or percentage points)
  plot_distance_by_consensus     average distance from the human score per consensus bin, one dot per condition
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ethos.plotting.style import INK, INK2, STYLE, SURFACE
from ethos.scores import toxicity_score

BLUE, GREY, LINE, GREEN, RED, ORANGE = "#2a78d6", "#7d7c76", "#e1e0d9", "#1baf7a", "#d03b3b", "#eb6834"
TABLEAU10 = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b", "#b279a2", "#ff9da6", "#9d755d",
             "#bab0ac"]
WEIGHTINGS = [("as-is", "As written", "#2a78d6"), ("equal", "Equal weights", "#eda100"),
              ("reversed", "Reversed", "#e87ba4")]
DOTTED = (0, (1.5, 1.5))


# --- helpers ---------------------------------------------------------------------------------------------------

def comment_scores(ratings):
    """Toxicity score per comment (rows) and condition (columns)."""
    return toxicity_score(ratings, by=["id", "condition"]).unstack("condition")


def _as_written(scores):
    """Only no profile and the conditions with their weights as written."""
    return scores[["none"] + [c for c in scores if c.endswith("|as-is")]]


def _distance(ratings, human):
    """Per comment |score - human score| (0-1), for no profile and each condition as written."""
    m = _as_written(comment_scores(ratings))
    return m.sub(human.loc[m.index], axis=0).abs()


def _name(condition):
    """Display name: "free_expression|as-is" -> "Free expression"; "Animal Welfare|as-is" -> "Animal Welfare"."""
    key = condition.split("|")[0]
    return key.replace("_", " ").capitalize() if key.islower() else key


def _is_profiles(conditions):
    """True for runs of the profile files (lowercase stems), False for single values."""
    return all(c.split("|")[0].islower() for c in conditions if c != "none")


def _ci(values, idx):
    """95% bootstrap interval of the mean, given resample indices idx (n_boot x n)."""
    return np.percentile(values[idx].mean(axis=1), [2.5, 97.5])


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _signed(v, digits=1):
    return f"{v:+.{digits}f}" if abs(v) > 1e-9 else "0"


# --- figures ---------------------------------------------------------------------------------------------------

def _shift_axis(ax, n_rows):
    ax.axvline(0, color=INK, lw=1.2)
    ax.text(0.03, -0.85, "no profile", fontsize=9, color=INK, va="center")
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks(np.arange(-1, 1.01, 0.5))
    ax.set_ylim(-1.05, n_rows - 0.4)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _signed(v)))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for yi in range(n_rows - 1):
        ax.axhline(yi + 0.5, color=LINE, lw=0.8)
    ax.set_xlabel("Toxicity Score Shift vs. No Profile\n(profile score − no-profile score)")


def _dots_with_ci(ax, shift, cols, y, idx, rng, color, spread, dot_size, alpha, **marker):
    """Per-comment dots (jittered by `spread`) and the mean with its 95% interval, one row per column."""
    for yi, c in zip(y, cols):
        ax.scatter(shift[c], yi + rng.uniform(-spread, spread, len(shift)), s=dot_size, color=color, alpha=alpha,
                   lw=0)
    mu = np.array([shift[c].mean() for c in cols])
    ci = np.array([_ci(shift[c].to_numpy(), idx) for c in cols])
    ax.errorbar(mu, y, xerr=[mu - ci[:, 0], ci[:, 1] - mu], color=color, lw=0, marker="o",
                markeredgecolor=SURFACE, **marker)


def plot_profile_shift_weighting(ratings, path, scope, seed=0):
    """(A) each comment's score shift vs. no profile per profile, weights as written; (B) the same for each weighting
    (as written / equal / reversed). Dots = comments, points = mean with 95% bootstrap interval over comments."""
    m = comment_scores(ratings)
    shift = m.drop(columns="none").sub(m["none"], axis=0)
    profiles = shift[[c for c in shift if c.endswith("|as-is")]].mean().sort_values().index.str.split("|").str[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shift), (2000, len(shift)))
    y = np.arange(len(profiles))
    with plt.rc_context(STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
        _dots_with_ci(a, shift, [f"{p}|as-is" for p in profiles], y, idx, rng, BLUE, 0.18, 16, 0.25,
                      ms=8, markeredgewidth=2, capsize=4, elinewidth=1.5, capthick=1.5)
        _shift_axis(a, len(profiles))
        a.set_yticks(y, [_name(p) for p in profiles])
        a.set_title("A. By profile (weights as written)", loc="left", fontsize=11.5, fontweight="bold", color=INK)
        for k, (w, label, color) in enumerate(WEIGHTINGS):
            cols = [f"{p}|{w}" for p in profiles]
            if all(c in shift for c in cols):
                _dots_with_ci(b, shift, cols, y + (1 - k) * 0.27, idx, rng, color, 0.06, 12, 0.3, label=label,
                              ms=7, markeredgewidth=1.5, capsize=3, elinewidth=1.4, capthick=1.4)
        _shift_axis(b, len(profiles))
        b.legend(loc="upper left", fontsize=9)
        b.set_title("B. By weighting (same values per profile)", loc="left", fontsize=11.5, fontweight="bold",
                    color=INK)
        fig.suptitle("Toxicity Score Relative to No Profile, by Moral Profile and Weighting", x=0.01, ha="left",
                     fontsize=13.5, fontweight="bold", color=INK)
        fig.text(0.01, 0.925, f"{scope}. Dots = comments; points = mean shift with 95% bootstrap interval. "
                 "Right of 0 = more toxic than no profile.", fontsize=9, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        _save(fig, path)


def plot_profile_distance(ratings, human, path, scope, seed=0):
    """Each comment's distance from the human score, for no profile and each profile as written.
    human: Series of human scores indexed by comment id. Dots = comments, points = mean with 95% bootstrap interval."""
    err = _distance(ratings, human)
    order = ["none"] + list(err.drop(columns="none").mean().sort_values().index)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), (2000, len(err)))
    y = np.arange(len(order))[::-1]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 5))
        ax.axvline(err["none"].mean(), color=GREY, lw=1.2, ls="--", zorder=1)
        for yi, c in zip(y, order):
            color = GREY if c == "none" else BLUE
            e = err[c].to_numpy()
            ci = _ci(e, idx)
            ax.scatter(e, yi + rng.uniform(-0.18, 0.18, len(e)), s=16, color=color, alpha=0.3, lw=0)
            ax.errorbar(e.mean(), yi, xerr=[[e.mean() - ci[0]], [ci[1] - e.mean()]], color=color, lw=0, marker="o",
                        ms=8, markeredgecolor=SURFACE, markeredgewidth=2, capsize=4, elinewidth=1.5, capthick=1.5)
            ax.text(1.04, yi, f"{e.mean():.2f}", va="center", fontsize=9, color=INK2)
        ax.axhline(len(order) - 1.5, color=LINE, lw=1)
        ax.set_yticks(y, ["No profile (base)" if c == "none" else _name(c) for c in order])
        ax.set_xlim(-0.03, 1.0)
        ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Absolute Difference from Human Score\n(|model score − human score|)")
        ax.set_title("Distance from Human Score: Base and Moral Profiles", loc="left", fontsize=11.5,
                     fontweight="bold", color=INK, pad=30)
        ax.text(0, 1.02, f"{scope} (weights as written). Dots = comments;\npoints = mean with 95% bootstrap "
                "interval (value at right). 0 = matches the annotators.", transform=ax.transAxes, fontsize=8.5,
                color=INK2)
        fig.tight_layout()
        _save(fig, path)


def plot_toxic_change(ratings, path, scope, percent=False):
    """Bar per condition (weights as written) of the change in TOXIC answers vs. no profile, with a no-profile row on
    top. percent=False: difference in counts; percent=True: difference in % of TOXIC answers (percentage points)."""
    r = ratings[(ratings.condition == "none") | ratings.condition.str.endswith("|as-is")]
    g = r.assign(toxic=r.toxic.astype(bool)).groupby("condition").toxic
    counts, n = g.sum(), int(g.size().max())
    base = counts["none"]
    diff = (counts.drop("none") - base).sort_values()
    val = diff / n * 100 if percent else diff
    profiles = _is_profiles(diff.index)
    unit = "profile" if profiles else "value"
    pad = 0.4 if percent else 5
    y = np.arange(len(diff))
    top = len(diff) + 0.4
    height = max(3.8, 0.21 * len(diff) + 1.9)
    small = len(diff) > 15
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, height))
        ax.barh(y, val, color=np.where(val > 0, ORANGE, GREEN), height=0.7)
        for yi, v in zip(y, val):
            ax.text(v + (pad if v > 0 else -pad), yi, f"{v:+.1f}" if percent else f"{v:+d}",
                    ha="left" if v > 0 else "right", va="center", fontsize=7.5 if small else 8.5, color=INK2)
        ax.axvline(0, color=INK, lw=1.2)
        ax.plot(0, top, marker="D", ms=8, color=GREY, zorder=3, clip_on=False)
        ax.text(pad * 2.4, top, f"0  ({base / n * 100:.1f}% TOXIC)" if percent else
                f"0  ({base} of {n} answers TOXIC)", va="center", fontsize=8 if small else 8.5, color=INK2)
        ax.axhline(len(diff) - 0.3, color=LINE, lw=1)
        ax.set_yticks(list(y) + [top], [_name(c) for c in diff.index] + ["No profile (base)"],
                      fontsize=8.5 if small else 9.5)
        ax.get_yticklabels()[-1].set_fontweight("bold")
        ax.set_ylim(-0.8, top + 0.8)
        span = max(abs(val.min()), abs(val.max())) * 1.2
        ax.set_xlim(min(-span * 0.3, val.min() * 1.2), span)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _signed(v, 0)))
        ax.set_xlabel(f"Change in % of TOXIC Answers vs. No Profile (percentage points)\n"
                      f"({unit} % TOXIC − no-profile % TOXIC)" if percent else
                      f"Difference in TOXIC Answers vs. No Profile\n({unit} count − no-profile count)")
        what = "Moral Profile" if profiles else "Single Moral Value"
        fig.suptitle(f"Change in {'% of ' if percent else ''}TOXIC Answers vs. No Profile, by {what}", x=0.02,
                     y=1 - 0.12 / height, va="top", ha="left", fontsize=12.5, fontweight="bold", color=INK)
        fig.text(0.02, 1 - 0.45 / height, f"{scope} ({n} answers per {unit}). Right of 0 = more TOXIC.", va="top",
                 fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / height))
        _save(fig, path)


def plot_distance_by_consensus(ratings, human, consensus_bin, path, model="gpt-6-luna"):
    """Average distance from the human score (0-1) per consensus bin: one dot per condition (weights as written;
    Tableau 10 colors, alphabetical, so they repeat past 10), no profile as a dotted line, and the closest (green ring)
    and furthest (red ring) condition in each bin labelled. human / consensus_bin: Series indexed by comment id."""
    err = _distance(ratings, human)
    bins = consensus_bin.loc[err.index]
    levels = sorted(bins.unique())
    g = err.groupby(bins).mean()
    conds = sorted(c for c in err if c != "none")
    names = [_name(c) for c in conds]
    colors = [TABLEAU10[k % len(TABLEAU10)] for k in range(len(conds))]
    n_answers = int(ratings.groupby(["id", "condition"]).size().max())
    sizes = ratings.loc[ratings.condition != "none", "profile"].str.count(";").add(1).unique()
    n_values = str(sizes[0]) if len(sizes) == 1 else f"{sizes.min()}-{sizes.max()}"
    given = "one moral value" if n_values == "1" else f"{n_values} moral values"
    subtitle = (f"Each value-conditioned LLM ({model} given {given}) rated all {ratings.id.nunique()} comments "
                f"{n_answers} times.")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6.6))
        for xi, b in enumerate(levels):
            v, base = g.loc[b, conds].to_numpy(), g.loc[b, "none"]
            ax.scatter(np.full(len(v), xi), v, s=26, c=colors, lw=0.3, edgecolor="white", zorder=2)
            ax.hlines(base, xi - 0.3, xi + 0.3, color=INK, lw=2.5, ls=DOTTED, zorder=3)
            for i, ring, side in ((v.argmin(), GREEN, -1), (v.argmax(), RED, 1)):
                ax.scatter(xi, v[i], s=60, facecolor="none", edgecolor=ring, lw=1.8, zorder=4)
                # label above the furthest / below the closest; if that would land on the dotted line, put it past
                near_line = v[i] > base - 0.04 if side > 0 else v[i] < base + 0.04
                y = (max if side > 0 else min)(v[i], base) if near_line else v[i]
                ax.annotate(names[i], (xi, y), xytext=(0, 9 * side), textcoords="offset points", ha="center",
                            va="bottom" if side > 0 else "top", fontsize=8, color=INK)
        n = bins.value_counts()
        ax.set_xticks(range(len(levels)), [f"{b.replace('-', '–')}\n(n = {n[b]})" for b in levels])
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Human Annotator Consensus")
        ax.set_ylabel("Average Distance from Human Score\n(|model score − human score|)")
        handles = [Line2D([], [], color=INK, lw=2.5, ls=DOTTED, label=f"{model}, no moral value in the prompt (base)"),
                   Line2D([], [], marker="o", ls="", color=TABLEAU10[0], ms=6,
                          label=f"Value-conditioned LLM, {given}\n({len(conds)} in total, one color each)"),
                   Line2D([], [], marker="o", ls="", mfc="none", mec=GREEN, mew=1.8, ms=7,
                          label="Closest in the band"),
                   Line2D([], [], marker="o", ls="", mfc="none", mec=RED, mew=1.8, ms=7,
                          label="Furthest in the band")]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=4, fontsize=8,
                  frameon=False, columnspacing=1.0, handletextpad=0.4, handlelength=1.8)
        ax.set_title("Human-LLM Alignment by Annotator Consensus", loc="left", color=INK, fontsize=13,
                     fontweight="bold", pad=52)
        ax.text(0, 1.115, subtitle, transform=ax.transAxes, fontsize=9, color=INK2)
        fig.tight_layout()
        _save(fig, path)
