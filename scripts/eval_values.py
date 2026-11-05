"""Kaleido values: how often each value appears in the judges' rationales.

Reads the `kaleido` column of judgements.csv (filled by scripts/kaleido_values.py). An item counts once per
rationale it was generated for; only values are counted (Kaleido's rights and duties are left out), and labels
are used as Kaleido wrote them (near-duplicates are not merged).

Writes to data/judgements/<sample>/eval/:
  kaleido_values.csv              number of rationales with each value: overall, base vs DMP, and per judge
  kaleido_values_overall.png      top values across all judges
  kaleido_values_by_judge.png     the same values, one panel per judge
  kaleido_values_base_vs_dmp.png  the same values, all base judges vs all DMP judges
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ethos.judges.core import load_config, output_path
from ethos.plotting.values import judge_colors, plot_values

ROOT = Path(__file__).resolve().parents[1]
TOP = 20           # values shown in kaleido_values_overall.png
TOP_BY_JUDGE = 10  # values shown in kaleido_values_by_judge.png


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
    colors = judge_colors([x for x in dict.fromkeys(j.judge) if x != "human"])   # same colors as removal_gap.png
    if "kaleido" not in j or j.kaleido.isna().all():
        print("no Kaleido values yet; run scripts/shell/run_kaleido.sh first")
        return
    j = j[j.kaleido.notna()]
    judges = list(dict.fromkeys(j.judge))

    # One row per (rationale, value), e.g. "Respect"
    items = pd.DataFrame([{"row": i, "judge": r.judge, "item": e["text"]}
                          for i, r in j.iterrows() for e in json.loads(r.kaleido) if e["type"] == "Value"]).drop_duplicates()
    # Number of rationales mentioning each value (a rationale can mention several, so counts don't add up to the total)
    counts = items.item.value_counts()
    by_judge = (items.groupby(["item", "judge"]).size().unstack(fill_value=0)
                .reindex(index=counts.index, columns=judges, fill_value=0))
    # Base vs DMP: pool the judges of each kind (e.g. qwen3:8b-base + gpt-6-luna-base)
    kind = lambda judge: "dmp" if judge.endswith("-dmp") else "base"
    by_kind = (items.assign(kind=items.judge.map(kind)).groupby(["item", "kind"]).size().unstack(fill_value=0)
               .reindex(index=counts.index, columns=["base", "dmp"], fill_value=0))
    n = {"all": len(j), **j.judge.value_counts().to_dict(), **j.judge.map(kind).value_counts().to_dict()}   # rationales

    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.concat([counts.rename("all"), by_kind, by_judge], axis=1)
    table.rename_axis("value").reset_index().to_csv(out_dir / "kaleido_values.csv", index=False)
    plot_values(counts.head(TOP), by_judge.head(TOP_BY_JUDGE), by_kind.head(TOP_BY_JUDGE), n, colors, out_dir)

    print("rationales:", n)
    print(table.head(10).to_string())
    print(f"wrote kaleido_values.csv and figures to {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
