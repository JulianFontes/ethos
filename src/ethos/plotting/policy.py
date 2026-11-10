"""Figures for the value policy (scripts/experiments/value_policy.py)."""
import matplotlib.pyplot as plt
import numpy as np

from ethos.plotting.style import INK, INK2, STYLE

TRAIN, VAL = "#4c78a8", "#f58518"
BASELINES = [("no_value", "No value", "#7d7c76", (0, (1.5, 1.5))), ("fixed_value", "Best fixed value", "#54a24b", "--"),
             ("oracle", "Oracle (ceiling)", "#b279a2", ":")]


def _band(ax, results, col, color, label):
    g = results.groupby("depth")[col]
    m, s = g.mean(), g.std()
    ax.plot(m.index, m, color=color, lw=2, marker="o", ms=5, label=label)
    ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.15, lw=0)


def plot_depth_curves(results, baselines, path, title):
    """(A) train vs validation MSE of the predicted distance by tree depth; (B) the policy's real distance from the
    human score on train and validation comments by depth, against the baselines (validation means).
    Lines = mean over CV rounds, bands = +-1 std."""
    with plt.rc_context(STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
        _band(a, results, "train_mse", TRAIN, "Train")
        _band(a, results, "val_mse", VAL, "Validation")
        a.set_title("A. Loss: error predicting the distance", loc="left", fontsize=11, fontweight="bold", color=INK)
        a.set_ylabel("Mean squared error")
        _band(b, results, "train_policy", TRAIN, "Policy, train comments")
        _band(b, results, "val_policy", VAL, "Policy, validation comments")
        val = baselines[baselines.split == "val"]
        for key, label, color, ls in BASELINES:
            b.axhline(val[key].mean(), color=color, lw=1.6, ls=ls, label=f"{label} ({val[key].mean():.3f})")
        b.set_title("B. Policy: distance from human score of the chosen value", loc="left", fontsize=11,
                    fontweight="bold", color=INK)
        b.set_ylabel("Mean |LLM score − human score|")
        b.set_ylim(0, None)
        for ax in (a, b):
            ax.set_xlabel("Tree depth")
            ax.set_xticks(np.unique(results.depth))
            ax.grid(axis="x", visible=False)
            ax.legend(fontsize=8.5, frameon=False)
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
        fig.text(0.01, 0.91, f"Lines = mean over {results.groupby(['repeat', 'fold']).ngroups} cross-validation rounds "
                 "(split by comment); bands = ±1 std. Lower is better.", fontsize=9, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
