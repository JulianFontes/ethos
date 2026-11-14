"""Plot the distribution (CDF) of per-comment toxicity scores: humans vs the model unsteered and steered both ways.

Reads <data>/ratings.csv (a steering run of profile_signal.py) and the sample's human scores, and writes
<data>/figures/toxicity_cdf.png. A comment's model score is the share of its answers that said TOXIC (per setting); its
human score is the share of annotators who rated it toxic.

Usage:
    python scripts/plots/plot_toxicity_cdf.py --data data/processed/civil-comments/boundary_n100_train/steer_toxicity_gpt_oss_20b_l7-19 \\
        --sample data/processed/civil-comments/boundary_n100_train/sample.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from ethos.steering.in_vllm import POLE_MEANINGS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def setting(condition):
    """'none' -> 'unsteered'; 'none|steer=toxicity@+68.6' -> 'toxicity@+68.6'."""
    return condition.split("|steer=", 1)[1] if "|steer=" in condition else "unsteered"


def ecdf(ax, values, **kw):
    x = np.sort(np.asarray(values, dtype=float))
    ax.step(np.r_[0, x, 1], np.r_[0, np.arange(1, len(x) + 1) / len(x), 1], where="post", **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="a steering run's data folder, with ratings.csv")
    p.add_argument("--sample", type=Path, required=True, help="the run's sample.csv (human toxicity scores)")
    args = p.parse_args()
    data = ROOT / args.data
    r = pd.read_csv(data / "ratings.csv")
    r["setting"] = r.condition.map(setting)
    scores = r.groupby(["id", "setting"]).toxic.mean().unstack()
    human = pd.read_csv(ROOT / args.sample).set_index("id").toxicity.reindex(scores.index)

    def alpha(s):
        return 0.0 if s == "unsteered" else float(s.rsplit("@", 1)[1])

    def vector(s):
        return s.rsplit("@", 1)[0].removeprefix("not_")
    order = sorted(scores.columns, key=lambda s: (vector(s) if alpha(s) else "", alpha(s)))
    vectors = sorted({vector(s) for s in order if alpha(s)})          # several vectors (e.g. random_0..2): shades
    blues, reds = ["#2e6fac", "#7fb3e0", "#163a5f"], ["#d1495b", "#f2a0a8", "#7a1c28"]
    color = {s: "#555555" if alpha(s) == 0 else (blues if alpha(s) > 0 else reds)[vectors.index(vector(s)) % 3]
             for s in order}

    steering = json.loads((data / "steering.json").read_text()) if (data / "steering.json").exists() else {}
    model = r.attrs.get("model") or Path(steering.get("vectors", "model")).name

    def label(s):                                     # model (alpha[, vector when a run steers with several], average)
        a = "α 0" if alpha(s) == 0 else f"α {alpha(s):+g}"
        a = f"{vector(s)}, {a}" if alpha(s) and len(vectors) > 1 else a
        return f"{model}{' Base' if alpha(s) == 0 else ''} ({a}, avg {scores[s].mean():.2f})"

    fig, ax = plt.subplots(figsize=(11, 8))
    ecdf(ax, human, color="#7a6fb0", lw=2, label=f"Humans (avg {human.mean():.2f})")
    # the target: the human curve at score 0.5 (the TOXIC / NOT TOXIC boundary), i.e. the share of comments at most half
    # of annotators found toxic; marked with a star and explained in the subtitle
    m = 0.5; y = float((human <= m).mean())
    ax.plot([-0.02, m], [y, y], ls=":", color="#7a6fb0", lw=1.2, zorder=4)              # dotted line to the y-axis
    ax.text(-0.01, y + 0.008, f"{y:.0%}", color="#7a6fb0", fontsize=9, fontweight="bold", va="bottom")
    ax.plot([m], [y], "*", ms=24, color="#c7b3f0", mec="#7a6fb0", mew=1.2, zorder=5)   # lavender, purple outline
    for s in order:
        ecdf(ax, scores[s], color=color[s], lw=2, ls=":" if alpha(s) == 0 else "-", label=label(s))   # unsteered dotted
    ax.axvline(0.5, color="#ccc", lw=0.8, ls=":")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
    ax.grid(color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Toxicity score", fontweight="bold")
    ax.set_ylabel("Cumulative share of comments", fontweight="bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    # legend under the plot, two columns
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.09), ncol=2,
              columnspacing=1.5, handlelength=2.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Toxicity Scores: Humans vs {model}", x=0.02, ha="left", fontsize=13, fontweight="bold")
    n_answers = int(r.groupby(["id", "setting"]).size().max())
    def meaning(v):                     # VISPA "for" sentences = +: the value itself, except where POLE_MEANINGS says
        if v not in POLE_MEANINGS:
            return f"{v}: α > 0 toward {v}, α < 0 away from it"
        return f"{v}: α > 0 toward {POLE_MEANINGS[v][0]}, α < 0 toward {POLE_MEANINGS[v][1]}"
    meaning = "; ".join(meaning(v) for v in vectors if not v.startswith("random_"))
    fig.text(0.02, 0.935, f"Score = share of annotators (humans) or of the model's {n_answers} answers that said toxic. "
             f"{len(scores)} comments ({args.sample.parent.name}).\nSteering: {meaning or ', '.join(vectors) + ' vector'}.\n"
             f"★ {y:.0%} of comments have a human score of {m:.2f} or lower.",
             fontsize=9, color="#555", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    (data / "figures").mkdir(exist_ok=True)
    fig.savefig(data / "figures/toxicity_cdf.png", dpi=150)
    print(f"wrote {(data / 'figures/toxicity_cdf.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
