"""Plot how far activation steering moves the model's internal state, layer by layer.

Reads <data>/hidden_states/states.npz and meta.json (written by scripts/experiments/capture_steered_states.py; the
capture script calls this plot when it finishes) and writes <data>/figures/shift_by_layer.png: for each steered setting,
the comments' average shift from unsteered along each layer's steering vector (layers without a vector use the nearest
steered layer's), with the steered layers shaded.

Usage:
    python scripts/plots/plot_steered_states.py --data data/processed/civil-comments/boundary_n100_train/steer_toxicity_gpt_oss_20b_l7-19
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ethos.steering.in_vllm import setting_label

ROOT = Path(__file__).resolve().parents[2]


def shifts(data):
    """(meta, {setting: [average shift per captured layer]}) for the steered settings."""
    meta = json.loads((data / "hidden_states/meta.json").read_text())
    z = np.load(data / "hidden_states/states.npz")
    vectors = np.load(ROOT / meta["vectors"] / f"{meta['value']}.npz")
    steered, captured = meta["steered_layers"], meta["captured_layers"]
    base = next(s for s, a in meta["settings"].items() if a == 0)

    def direction(layer):
        v = vectors[f"layer_{min(max(layer, steered[0]), steered[-1])}"].astype(float)
        return v / np.linalg.norm(v)

    out = {s: [((z[f"{s}_{l}"].astype(float) - z[f"{base}_{l}"].astype(float)) @ direction(l)).mean() for l in captured]
           for s, a in meta["settings"].items() if a != 0}
    return meta, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="a run's data folder, with hidden_states/")
    args = p.parse_args()
    data = ROOT / args.data
    meta, out = shifts(data)
    steered, captured = meta["steered_layers"], meta["captured_layers"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvspan(steered[0] - 0.5, steered[-1] + 0.5, color="#f1eefa", zorder=0)
    for s, values in out.items():
        ax.plot(captured, values, "o-", ms=4, lw=2, color="#2e6fac" if meta["settings"][s] > 0 else "#d1495b", label=setting_label(s))
    ax.axhline(0, color="#bbb", lw=0.8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [Patch(color="#e4ddf5")], labels + [f"steered layers ({steered[0]}-{steered[-1]})"],
              frameon=False, fontsize=9, loc="lower left")
    ax.set_xlabel("layer")
    ax.set_ylabel(f"average shift vs unsteered\n(along each layer's {meta['value']} vector)")
    ax.spines[["top", "right"]].set_visible(False)
    model = meta["model"].split("/")[-1]
    alpha = max(meta["settings"].values())
    fig.suptitle(f"How far steering moves {model}'s internal state, layer by layer", x=0.02, ha="left", fontsize=13,
                 fontweight="bold")
    fig.text(0.02, 0.885, f"{meta['n']} comments ({Path(meta.get('sample') or meta['order']).parent.name}); last token of the "
             f"judge prompt. {meta['value']} vector added at the steered layers, alpha +/-{alpha:g}.", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    (data / "figures").mkdir(exist_ok=True)
    fig.savefig(data / "figures/shift_by_layer.png", dpi=150)
    print(f"wrote {(data / 'figures/shift_by_layer.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
