"""Context sensitivity: how each judge's allow/remove decision shifts from the original comment to its
consequentialist, emotional and relational variations.

Writes to data/judgements/<sample>/eval/:
  context_shift.csv         flips each way (allow -> remove, remove -> allow) and the net change in remove count
  decisions_by_version.png  allow/remove split per version, with each context's change vs. the original
  detoxify_score_shift.png  raw Detoxify score change (variation - original) per context
"""
import argparse
from pathlib import Path

import pandas as pd

from ethos.judges.core import VERSIONS, load_config, output_path
from ethos.plotting.context import plot_context

ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = [v for v in VERSIONS if v != "original"]


def shift_table(j, judges):
    """One row per judge and context: flips each way and the net change in remove count."""
    rows = []
    for judge in judges:
        d = j[j.judge == judge].pivot(index="id", columns="version", values="decision")
        for c in CONTEXTS:
            pair = d[["original", c]].dropna()                       # comments judged in both versions
            delta = (pair[c] == "remove").astype(int) - (pair.original == "remove").astype(int)
            to_remove, to_allow = int((delta == 1).sum()), int((delta == -1).sum())
            rows.append({"judge": judge, "context": c, "n": len(delta), "to_remove": to_remove,
                         "to_allow": to_allow, "net_change": to_remove - to_allow})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=ROOT / "configs" / "judges.yaml", type=Path)
    parser.add_argument("--sample-dir", help="overrides sample_dir in the config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample_dir = ROOT / (args.sample_dir or cfg["sample_dir"])
    path = output_path(sample_dir)
    out_dir = path.parent / "eval"

    j = pd.read_csv(path)
    judges = [x for x in dict.fromkeys(j.judge) if x != "human"]   # humans only labeled the originals
    table = shift_table(j, judges)

    diffs = None
    if "detoxify" in judges:
        s = j[j.judge == "detoxify"].pivot(index="id", columns="version", values="score")
        diffs = s[CONTEXTS].sub(s.original, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "context_shift.csv", index=False)
    plot_context(j, diffs, judges, CONTEXTS, out_dir)

    print(table.to_string(index=False, float_format="%.1f"))
    print(f"wrote context_shift.csv and figures to {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
