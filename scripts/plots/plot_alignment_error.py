"""Plot the model's average distance from the humans at each steering strength (alpha); lower is better.

Reads <data>/ratings.csv (a steering run of profile_signal.py, e.g. an alpha sweep) and the sample's human scores, and
writes <data>/figures/alignment_error.png: for each alpha (unsteered = 0), the mean over comments of
|model toxicity score - human toxicity score|. A comment's model score is the share of its answers that said TOXIC;
its human score is the share of annotators.

Usage:
    python scripts/plots/plot_alignment_error.py \\
        --data data/processed/civil-comments/boundary_n100_train/.archive/alpha_sweep_toxicity_llama3_8b_l10-25 \\
        --sample data/processed/civil-comments/boundary_n100_train/sample.csv
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


def alpha_of(condition):
    """'none' -> 0.0; 'none|steer=toxicity@+0.3' -> 0.3."""
    return float(condition.rsplit("@", 1)[1]) if "|steer=" in condition else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="a steering run's data folder, with ratings.csv")
    p.add_argument("--sample", type=Path, required=True, help="the run's sample.csv (human toxicity scores)")
    p.add_argument("--passed", nargs=2, type=float, metavar=("LOW", "HIGH"),
                   help="the alpha range whose rationales stayed coherent (e.g. -0.3 0.3, from reading the sweep's "
                        "rationales): shaded, and the minimum is taken within it")
    args = p.parse_args()
    data = ROOT / args.data
    r = pd.read_csv(data / "ratings.csv")
    r["alpha"] = r.condition.map(alpha_of)
    scores = r.groupby(["id", "alpha"]).toxic.mean().unstack().dropna()     # comments with answers at every alpha
    human = pd.read_csv(ROOT / args.sample).set_index("id").toxicity.reindex(scores.index)
    error = (scores.sub(human, axis=0)).abs().mean().rename("mean |model - human|")   # one value per alpha

    steering = json.loads((data / "steering.json").read_text()) if (data / "steering.json").exists() else {}
    model = Path(steering.get("vectors", "model")).name
    value = ", ".join(steering.get("values", []))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(error.index, error, "o-", color="#d1495b", lw=2, ms=7)
    if args.passed:                                                    # strengths that kept the model coherent
        low, high = args.passed
        ax.axvspan(low - 0.03, high + 0.03, color="#98e8c8", alpha=0.25, zorder=0)   # mint
        ax.text(low - 0.02, 0.03, "passed:\ncoherent\nrationales", ha="left", va="bottom",
                color="#2f9e78", fontsize=9, fontweight="bold")
        failed = error[(error.index < low) | (error.index > high)]
        ax.plot(failed.index, failed, "o", ms=7, mfc="white", mec="#d1495b", mew=1.5, zorder=3)   # hollow = failed
        error_ok = error[(error.index >= low) & (error.index <= high)]
    else:
        error_ok = error
    best = error_ok.idxmin()                                           # the alpha closest to humans (that passed)
    ax.axvline(best, color="#d1495b", lw=1.2, ls="--", alpha=0.7)
    ax.text(best, 0.97, f" minimum: α {best:+g} ({error[best]:.2f})" if best else f" minimum: α 0 ({error[best]:.2f})",
            color="#d1495b", fontsize=9, fontweight="bold", va="top", ha="left")
    ax.axvline(0, color="#999", lw=1, ls=":")
    ax.set_ylim(0, 1)
    ax.set_xticks(error.index, [f"{a:+g}" if a else "0" for a in error.index])
    ax.grid(color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Steering strength (α)", fontweight="bold")
    ax.set_ylabel("Mean |model − human| toxicity score", fontweight="bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Distance from Humans by Steering Strength: {model}", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.935, f"{len(scores)} comments ({args.sample.parent.name}); {value} vector, α 0 = unsteered. "
             "Lower is closer to humans." + (" Hollow points: strengths that broke the model." if args.passed else ""),
             fontsize=9, color="#555", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    (data / "figures").mkdir(exist_ok=True)
    fig.savefig(data / "figures/alignment_error.png", dpi=150)
    print(error.round(3).to_string())
    print(f"wrote {(data / 'figures/alignment_error.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
