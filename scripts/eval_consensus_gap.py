"""Score gap between each judge and the human annotators, by human consensus (bootstrap plot).

Reads data/judgements/<sample>/judgements.csv (original comments) and the sample's sample.csv, and writes to
data/judgements/<sample>/eval/:
  score_gap_by_bin.csv   mean signed and absolute gap per consensus bin and judge, with 95% bootstrap intervals
  score_gap_by_bin.png   bootstrap mean |judge - human| per consensus bin and judge, with 95% interval (line plot)

--extra adds judges on top (e.g. an experiment run on only some bins) and writes score_gap_by_bin_<name>.csv/.png
instead, leaving the main files alone. Extra judges get their own bootstrap draws (seed 1), so the --judges numbers
are identical to the main plot's.

Usage:
    python scripts/eval_consensus_gap.py                                        # detoxify + qwen3:8b-annotator
    python scripts/eval_consensus_gap.py --extra qwen3:8b-annotator-dmp --name dmp
"""
import argparse
from pathlib import Path

import pandas as pd

from ethos.judges.core import output_path
from ethos.plotting.consensus import gap_by_bin, plot_score_gap_by_bin

ROOT = Path(__file__).resolve().parents[1]
TITLES = {"detoxify": "Detoxify", "gpt-6-luna-minimal": "gpt-6-luna, minimal prompt (10 ratings)", "qwen3:8b-annotator": "qwen3:8b, annotator instructions (10 ratings)",
          "qwen3:8b-annotator-dmp": "qwen3:8b + moral profiles (10 profiles)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-dir", default="data/processed/consensus_n125")
    parser.add_argument("--judges", nargs="+", default=["detoxify", "qwen3:8b-annotator"])
    parser.add_argument("--extra", nargs="+", default=[], help="judges to add, with their own bootstrap draws")
    parser.add_argument("--name", default=None,
                        help="output suffix: writes score_gap_by_bin_<name>.png/.csv (keeps the main plot)")
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
    stem = (f"score_gap_by_bin_{args.name}" if args.name else
            "score_gap_by_bin_extra" if args.extra else "score_gap_by_bin")

    out = path.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(out / f"{stem}.csv", index=False)
    plot_score_gap_by_bin(table, {c: TITLES.get(c, c) for c in args.judges + args.extra}, out / f"{stem}.png",
                          f"{Path(args.sample_dir).name}: {len(df)} test comments")
    print(pd.DataFrame(table).round(3).to_string(index=False))
    print(f"wrote {stem}.png and {stem}.csv to {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
