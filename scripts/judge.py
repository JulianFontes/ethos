"""Run judges over every comment of a sample.

All judges and the sample they run on are defined in configs/judges/judges.yaml; judges run in the
order listed. Results go to data/judgements/<sample>/judgements.csv (one row per judge per
comment); the config is copied to its log/ folder. Re-running skips comments already done.

Usage:
  python scripts/judge.py                                   # every judge in configs/judges/judges.yaml
  python scripts/judge.py --only gpt-6-luna-base            # only judges with these names
  python scripts/judge.py --limit 2                         # first 2 comments only
  python scripts/judge.py --query "consensus < 0.7"         # only comments whose sample.csv row matches
  python scripts/judge.py --sample-dir data/processed/consensus_n125_train   # override the config's sample_dir
"""
import argparse
from pathlib import Path

from ethos.judges import run_judge
from ethos.judges.core import ROOT, load_config, load_judges, output_path, save_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "judges" / "judges.yaml")
    p.add_argument("--sample-dir", default=None, help="override the config's sample_dir")
    p.add_argument("--only", nargs="+", default=None, help="only run judges with these names")
    p.add_argument("--limit", type=int, default=None, help="only judge the first N comments")
    p.add_argument("--query", default=None, help='only judge comments matching this pandas query on sample.csv, '
                                                 'e.g. "consensus < 0.7"')
    args = p.parse_args()

    sample_dir = args.sample_dir or load_config(args.config).get("sample_dir")
    if not sample_dir:
        raise SystemExit(f"set sample_dir in {args.config} or pass --sample-dir")
    judges = [c for c in load_judges(args.config) if not args.only or c["name"] in args.only]
    if args.only and len(judges) < len(args.only):
        names = [c["name"] for c in load_judges(args.config)]
        raise SystemExit(f"unknown judge name in --only {args.only}; available: {names}")
    save_config(args.config, sample_dir)
    for cfg in judges:
        run_judge({**cfg, "query": args.query}, sample_dir, args.limit)
    print(f"judgements: {output_path(sample_dir)}")


if __name__ == "__main__":
    main()
