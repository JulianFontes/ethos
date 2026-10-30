"""Run judges over every comment version (original + variations) of a sample.

All judges and the sample they run on are defined in configs/judges.yaml; judges run in the
order listed. Results go to data/judgements/<sample>/judgements.csv (one row per judge per
version); the config is copied to its log/ folder. Re-running skips versions already done.

Usage:
  python scripts/judge.py                                   # every judge in configs/judges.yaml
  python scripts/judge.py --only qwen3:8b-dmp               # only judges with these names
  python scripts/judge.py --limit 2                         # first 2 comments only
  python scripts/judge.py --sample-dir data/processed/sample_n200   # override the config's sample_dir
"""
import argparse
from pathlib import Path

from ethos.judges import run_judge
from ethos.judges.core import ROOT, add_human_labels, load_config, load_judges, output_path, save_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "judges.yaml")
    p.add_argument("--sample-dir", default=None, help="override the config's sample_dir")
    p.add_argument("--only", nargs="+", default=None, help="only run judges with these names")
    p.add_argument("--limit", type=int, default=None, help="only judge the first N comments")
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
        run_judge(cfg, sample_dir, args.limit)
    # same threshold as the `human` judge, so human_decision always matches its decision
    human = next((c for c in load_judges(args.config) if c["type"] == "human"), {})
    add_human_labels(sample_dir, human.get("threshold", 0.5))
    print(f"judgements: {output_path(sample_dir)}")


if __name__ == "__main__":
    main()
