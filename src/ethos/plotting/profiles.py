"""Moral-profile figures for scripts/experiments/profile_signal.py results (ratings.csv: one row per answer with
`id`, `condition` ("none" or "<profile>|<weighting>") and `toxic`)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ethos.plotting.sample_overview import STYLE, INK, INK2, SURFACE

BLUE, GREY, LINE = "#2a78d6", "#7d7c76", "#e1e0d9"
WEIGHTINGS = [("as-is", "As written", "#2a78d6"), ("equal", "Equal weights", "#eda100"),
              ("reversed", "Reversed", "#e87ba4")]


def comment_scores(ratings):
    """Share of TOXIC answers per comment (rows) and condition (columns)."""
    r = ratings.assign(toxic=ratings.toxic.astype(bool))
    return r.pivot_table(index="id", columns="condition", values="toxic", aggfunc="mean")


def _ci(values, idx):
    return np.percentile(values[idx].mean(axis=1), [2.5, 97.5])


def _name(profile):
    return profile.replace("_", " ").capitalize()


def _shift_axis(ax, n_rows):
    ax.axvline(0, color=INK, lw=1.2)
    ax.text(0.03, -0.85, "no profile", fontsize=9, color=INK, va="center")
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks(np.arange(-1, 1.01, 0.5))
    ax.set_ylim(-1.05, n_rows - 0.4)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}" if abs(v) > 1e-9 else "0"))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for yi in range(n_rows - 1):
        ax.axhline(yi + 0.5, color=LINE, lw=0.8)
    ax.set_xlabel("Toxicity Score Shift vs. No Profile\n(profile score − no-profile score)")


def plot_profile_shift_weighting(ratings, path, scope, seed=0):
    """Two panels of each comment's toxicity score shift vs. no profile: (A) per profile as written,
    (B) per profile for each weighting (as written / equal / reversed). Dots = comments, points = mean with 95%
    bootstrap interval over comments."""
    m = comment_scores(ratings)
    shift = m.drop(columns="none").sub(m["none"], axis=0)
    profiles = shift[[c for c in shift if c.endswith("|as-is")]].mean().sort_values().index.str.split("|").str[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shift), (2000, len(shift)))
    y = np.arange(len(profiles))
    with plt.rc_context(STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
        cols = [f"{p}|as-is" for p in profiles]
        for yi, c in zip(y, cols):
            a.scatter(shift[c], yi + rng.uniform(-0.18, 0.18, len(shift)), s=16, color=BLUE, alpha=0.25, lw=0)
        mu = np.array([shift[c].mean() for c in cols])
        ci = np.array([_ci(shift[c].to_numpy(), idx) for c in cols])
        a.errorbar(mu, y, xerr=[mu - ci[:, 0], ci[:, 1] - mu], color=BLUE, lw=0, marker="o", ms=8,
                   markeredgecolor=SURFACE, markeredgewidth=2, capsize=4, elinewidth=1.5, capthick=1.5)
        _shift_axis(a, len(profiles))
        a.set_yticks(y, [_name(p) for p in profiles])
        a.set_title("A. By profile (weights as written)", loc="left", fontsize=11.5, fontweight="bold", color=INK)
        for k, (w, label, color) in enumerate(WEIGHTINGS):
            cols = [f"{p}|{w}" for p in profiles]
            if not all(c in shift for c in cols):
                continue
            off = (1 - k) * 0.27
            for yi, c in zip(y, cols):
                b.scatter(shift[c], yi + off + rng.uniform(-0.06, 0.06, len(shift)), s=12, color=color, alpha=0.3,
                          lw=0)
            mu = np.array([shift[c].mean() for c in cols])
            ci = np.array([_ci(shift[c].to_numpy(), idx) for c in cols])
            b.errorbar(mu, y + off, xerr=[mu - ci[:, 0], ci[:, 1] - mu], color=color, lw=0, marker="o", ms=7,
                       markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=3, elinewidth=1.4, capthick=1.4,
                       label=label)
        _shift_axis(b, len(profiles))
        b.legend(loc="upper left", fontsize=9)
        b.set_title("B. By weighting (same values per profile)", loc="left", fontsize=11.5, fontweight="bold",
                    color=INK)
        fig.suptitle("Toxicity Score Relative to No Profile, by Moral Profile and Weighting", x=0.01, ha="left",
                     fontsize=13.5, fontweight="bold", color=INK)
        fig.text(0.01, 0.925, f"{scope}. Dots = comments; points = mean shift with 95% bootstrap interval. "
                 "Right of 0 = more toxic than no profile.", fontsize=9, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_profile_distance(ratings, human, path, scope, seed=0):
    """Each comment's absolute difference from the human toxicity score, for no profile and each profile as written.
    human: Series of human scores indexed by comment id. Dots = comments, points = mean with 95% bootstrap interval."""
    m = comment_scores(ratings)
    m = m[["none"] + [c for c in m if c.endswith("|as-is")]]
    err = m.sub(human.loc[m.index], axis=0).abs()
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
        ax.set_yticks(y, ["No profile (base)" if c == "none" else _name(c.split("|")[0]) for c in order])
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
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
