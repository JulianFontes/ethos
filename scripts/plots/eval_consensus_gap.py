"""Score gap between each judge and the human annotators, by human consensus (bootstrap plot).

Reads data/judgements/<sample>/judgements.csv (original comments) and the sample's sample.csv, and writes to
data/experiments/judge_alignment/<sample>/<name>/ (name: --name, or "main"):
  score_gap_by_bin.csv   mean signed and absolute gap per consensus bin and judge, with 95% bootstrap intervals
  score_gap_by_bin.png   bootstrap mean |judge - human| per consensus bin and judge, with 95% interval (line plot)
  sources.yaml           which data it came from

--extra adds judges with their own bootstrap draws (seed 1), so the --judges numbers stay identical to the main plot's.

Usage:
    python scripts/plots/eval_consensus_gap.py                              # detoxify + gpt-6-luna-base
    python scripts/plots/eval_consensus_gap.py --judges gpt-6-luna-base gpt-6-luna-dmp-random --name gpt_base_vs_random_dmp
"""
import argparse
from pathlib import Path

import pandas as pd

from ethos.judges.core import output_path
from ethos.paths import EXPERIMENTS, ROOT
from ethos.plotting.consensus import gap_by_bin, plot_score_gap_by_bin
from ethos.sources import write_sources

TITLES = {"detoxify": "Detoxify", "gpt-6-luna-base": "gpt-6-luna, minimal prompt (10 ratings)", "gpt-6-luna-dmp-random": "gpt-6-luna + random DMP profiles (10 ratings)",
          "gpt-6-luna-dmp-expression": "gpt-6-luna + expression profile (10 ratings)",
          "gpt-6-luna-dmp-protective": "gpt-6-luna + protective profile (10 ratings)", "qwen3:8b-annotator": "qwen3:8b, annotator instructions (10 ratings)",
          "qwen3:8b-annotator-dmp": "qwen3:8b + moral profiles (10 profiles)"}

# fixed colors so a judge looks the same in every version of the plot (others follow JUDGE_COLORS order)
COLORS = {"detoxify": "#2a78d6", "gpt-6-luna-base": "#eb6834", "gpt-6-luna-dmp-random": "#1baf7a"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-dir", default="data/processed/consensus_n125_test")
    parser.add_argument("--judges", nargs="+", default=["detoxify", "gpt-6-luna-base"])
    parser.add_argument("--extra", nargs="+", default=[], help="judges to add, with their own bootstrap draws")
    parser.add_argument("--name", default=None,
                        help="output folder name under judge_alignment/<sample>/ (default: main, or extra with --extra)")
    args = parser.parse_args()

    path = output_path(args.sample_dir)
    j = pd.read_csv(path)
    j = j[(j.version == "original") & j.score.notna()]
    scores = j.pivot_table(index="id", columns="judge", values="score")
    sample = pd.read_csv(ROOT / args.sample_dir / "sample.csv", usecols=["id", "consensus"])
    df = sample.set_index("id").join(scores, how="inner")

    table = gap_by_bin(df, args.judges)
    if args.extra:
        table += gap_by_bin(df, args.extra, seed=1)
    out = EXPERIMENTS / "judge_alignment" / Path(args.sample_dir).name / (args.name or ("extra" if args.extra else "main"))
    out.mkdir(parents=True, exist_ok=True)
    stem = "score_gap_by_bin"
    pd.DataFrame(table).to_csv(out / f"{stem}.csv", index=False)
    write_sources(out, [path, ROOT / args.sample_dir / "sample.csv"])
    plot_score_gap_by_bin(table, {c: TITLES.get(c, c) for c in args.judges + args.extra}, out / f"{stem}.png",
                          f"{Path(args.sample_dir).name}: {len(df)} test comments", colors=COLORS)
    print(pd.DataFrame(table).round(3).to_string(index=False))
    print(f"wrote {stem}.png and {stem}.csv to {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
