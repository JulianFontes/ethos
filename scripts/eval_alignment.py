"""Human-judgement alignment: how each judge's allow/remove decisions on the original comments
compare with the human annotations, overall and per human-toxicity bin.

Writes to data/judgements/<sample>/eval/:
  alignment.csv          remove rate and agreement with `human`, per judge (bin = "all") and per bin
  decisions_overall.png  allow/remove counts per judge
  decisions_by_bin.png   the same, one panel per toxicity bin
  removal_gap.png        model minus human share of comments removed, per toxicity bin
"""
import argparse
from pathlib import Path

import pandas as pd

from ethos.judges.core import load_config, output_path
from ethos.plotting.alignment import plot_alignment

ROOT = Path(__file__).resolve().parents[1]


def alignment_table(o):
    """Remove rate and agreement with the human decision, per judge and bin."""
    human = o[o.judge == "human"].set_index("id").decision
    o = o.assign(agree=o.decision.values == human.reindex(o.id).values)
    rows = []
    for b, sub in [("all", o), *o.groupby("toxicity_bin")]:
        for judge, g in sub.groupby("judge", sort=False):
            rows.append({"bin": b, "judge": judge, "n": len(g),
                         "remove_rate": (g.decision == "remove").mean(), "agreement": g.agree.mean()})
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
    sample = pd.read_csv(sample_dir / "sample.csv", usecols=["id", "toxicity_bin"])
    o = j[j.version == "original"].merge(sample, on="id")
    judges = list(dict.fromkeys(j.judge))                                   # config order
    bins = sorted(o.toxicity_bin.unique(), key=lambda b: float(b.split("-")[0]))

    out_dir.mkdir(parents=True, exist_ok=True)
    table = alignment_table(o)
    table.to_csv(out_dir / "alignment.csv", index=False)
    human = next((x for x in cfg["judges"] if x["type"] == "human"), {})
    plot_alignment(o, judges, bins, out_dir, threshold=human.get("threshold", 0.5))

    print(table[table.bin == "all"].to_string(index=False, float_format="%.2f"))
    print(f"wrote alignment.csv and figures to {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
