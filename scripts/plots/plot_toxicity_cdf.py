"""Plot the distribution (CDF) of per-comment toxicity scores: humans vs the model unsteered and steered both ways.

Reads <data>/ratings.csv (a steering run of profile_signal.py) and the sample's human scores, and writes
<data>/figures/toxicity_cdf.png. A comment's model score is the share of its answers that said TOXIC (per setting); its
human score is the share of annotators who rated it toxic.

Usage:
    python scripts/plots/plot_toxicity_cdf.py --data data/processed/boundary_n100_train/steer_toxicity_gpt_oss_20b_l7-19 \\
        --sample data/processed/boundary_n100_train/sample.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

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
    order = sorted(scores.columns, key=alpha)
    color = {s: "#555555" if alpha(s) == 0 else "#2e6fac" if alpha(s) > 0 else "#d1495b" for s in order}

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ecdf(ax, human, color="#7a6fb0", lw=2.5, label=f"humans (average {human.mean():.2f})")
    for s in order:
        ecdf(ax, scores[s], color=color[s], lw=2, ls="-" if alpha(s) == 0 else "--",
             label=f"model, {s} (average {scores[s].mean():.2f})")
    ax.axvline(0.5, color="#ccc", lw=0.8, ls=":")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("toxicity score (humans: share of annotators; model: share of its answers that said TOXIC)")
    ax.set_ylabel("share of comments with this score or lower")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    steering = json.loads((data / "steering.json").read_text()) if (data / "steering.json").exists() else {}
    model = r.attrs.get("model") or Path(steering.get("vectors", "model")).name
    fig.suptitle(f"Toxicity scores per comment: humans vs {model}, unsteered and steered", x=0.02, ha="left",
                 fontsize=13, fontweight="bold")
    n_answers = int(r.groupby(["id", "setting"]).size().max())
    fig.text(0.02, 0.89, f"{len(scores)} comments ({args.sample.parent.name}); {n_answers} answers per comment and setting. "
             f"Steering: {', '.join(order[:1] + order[2:]) if len(order) == 3 else ', '.join(order)}.", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    (data / "figures").mkdir(exist_ok=True)
    fig.savefig(data / "figures/toxicity_cdf.png", dpi=150)
    print(f"wrote {(data / 'figures/toxicity_cdf.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
