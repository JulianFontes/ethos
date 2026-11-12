"""Overview figure: which decoder layers PCA steering is applied to, per model (the paper's setting and ours).
Each fitted vector set and each steered run also saves its own figure (fit_steering_vectors.py, submit_vllm.py).

Usage:
    python scripts/plots/plot_steering_layers.py --out data/experiments/05_activation_steering/steering_layers.png
"""
import argparse
from pathlib import Path

from ethos.plotting.steering import plot_steering_layers

ROWS = [
    ("LLaMA3-8B", 32, range(10, 26), "VISPA paper (App. D)"),
    ("Qwen3-8B", 36, range(10, 26), "our VISPA check"),
    ("gpt-oss-20b", 24, range(7, 20), "proposed: same share of depth"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/experiments/05_activation_steering/steering_layers.png"))
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_steering_layers(ROWS, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
