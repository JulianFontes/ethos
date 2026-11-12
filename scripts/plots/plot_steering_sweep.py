"""Alpha sweep figure: which steering strengths change the TOXIC / NOT TOXIC answers without breaking them.

Reads a steered run's folder (e.g. from configs/gpt_oss_20b/alpha_sweep.yaml): ratings.csv, failures.csv (failed
attempts, e.g. invalid JSON) and steering.json. Writes to --out:
  alpha_sweep.png    TOXIC share (per comment and mean), valid answers and rationale length per strength
  sweep_summary.csv  the same numbers per strength
  examples.csv       a few rationales per strength, to read how the text changes

Usage:
    python scripts/plots/plot_steering_sweep.py --data data/processed/consensus_n1000_train/steering_alpha_sweep_gpt_oss_20b \
        --out data/experiments/05_activation_steering/alpha_sweep_gpt_oss_20b
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ethos.plotting.steering import plot_alpha_sweep
from ethos.sources import write_sources


def alpha_of(condition):
    """"none" -> 0.0; "none|steer=fairness@-20.3" -> -20.3."""
    return float(condition.rsplit("@", 1)[1]) if "|steer=" in condition else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="the run's folder (ratings.csv, steering.json)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--examples", type=int, default=3, help="rationales per strength in examples.csv")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    steering = json.loads((args.data / "steering.json").read_text())
    fraction = dict(zip(steering["alphas"], steering.get("alpha_fractions", steering["alphas"])))

    r = pd.read_csv(args.data / "ratings.csv")
    r["toxic"] = r.toxic.astype(str).str.lower() == "true"
    r["alpha"] = r.condition.map(alpha_of)
    r["words"] = r.rationale.fillna("").str.split().str.len()
    failures_path = args.data / "failures.csv"
    f = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame(columns=["condition"])
    failed = f.condition.map(alpha_of).value_counts() if len(f) else pd.Series(dtype=int)

    per_comment = r.groupby(["alpha", "id"]).toxic.mean().reset_index()
    tried = sorted(set(r.alpha) | set(failed.index))         # strengths with answers or failed attempts (a strength
    summary = (r.groupby("alpha").agg(toxic_answers=("toxic", "mean"), answers=("toxic", "size"),   # where every
                                      words=("words", "median"))                                    # call failed has
               .reindex(tried).rename_axis("alpha").reset_index())                                  # no answers)
    summary["answers"] = summary.answers.fillna(0).astype(int)
    summary["toxic"] = summary.alpha.map(per_comment.groupby("alpha").toxic.mean())  # mean over comments
    summary["failed_attempts"] = summary.alpha.map(failed).fillna(0).astype(int)
    summary["valid"] = summary.answers / (summary.answers + summary.failed_attempts)
    summary["fraction"] = summary.alpha.map(lambda a: fraction.get(a, a))
    summary["unsteered"] = summary.alpha == 0
    summary["label"] = [f"{fr:+g}\n(α {a:+g})" if a else "0\n(unsteered)" for fr, a in zip(summary.fraction, summary.alpha)]
    per_comment["label"] = per_comment.alpha.map(dict(zip(summary.alpha, summary.label)))

    model = Path(steering["vectors"]).name
    title = (f"Steering strength sweep: {model}, {', '.join(steering['values'])} vector, layers "
             f"{steering['layers'][0]}–{steering['layers'][-1]}, no-value prompt ({r.id.nunique()} comments)")
    plot_alpha_sweep(summary, per_comment, args.out / "alpha_sweep.png", title)
    cols = ["fraction", "alpha", "toxic", "toxic_answers", "answers", "failed_attempts", "valid", "words"]
    summary[cols].to_csv(args.out / "sweep_summary.csv", index=False)
    examples = r.sample(frac=1, random_state=0).groupby("alpha").head(args.examples).sort_values("alpha")
    examples[["alpha", "id", "toxic", "rationale"]].to_csv(args.out / "examples.csv", index=False)
    write_sources(args.out, [args.data / "ratings.csv", args.data / "steering.json"])
    pd.set_option("display.width", 160)
    print(summary[cols].round(3).to_string(index=False))
    print(f"wrote alpha_sweep.png, sweep_summary.csv, examples.csv to {args.out}")


if __name__ == "__main__":
    main()
