"""Add Kaleido values, rights and duties for every judge rationale to judgements.csv.

Each rationale is given to Kaleido as-is. The new `kaleido` column holds a JSON list with one entry per item:
  {"type": "Value", "text": "Respect", "relevance": 0.98, "valence": "opposes", "valence_p": 0.95}
relevance = probability the item is relevant; valence = does the rationale's comment support, oppose or
either, with its probability. Rows without a rationale (human, detoxify) stay empty.
Re-running only fills rows that are still empty; the CSV is saved every 20 rationales.

Usage:
  python scripts/kaleido_values.py               # every rationale
  python scripts/kaleido_values.py --limit 5     # only the next 5 (quick test)
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ethos.judges.core import COLUMNS, ROOT, load_config, output_path, save_config
from ethos.llm.kaleido import Kaleido


def describe(kaleido, rationale, n):
    """Generate items for one rationale, then score each one's relevance and valence."""
    items = kaleido.generate(rationale, n)
    relevance = kaleido.score("relevance", rationale, items)
    valence = kaleido.score("valence", rationale, items)
    entries = []
    for item, rel, val in zip(items, relevance, valence):
        kind, _, name = item.partition(": ")
        label = max(val, key=val.get)
        entries.append({"type": kind, "text": name, "relevance": round(rel["Yes"], 3),
                        "valence": label.lower(), "valence_p": round(val[label], 3)})
    return json.dumps(entries)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "kaleido.yaml")
    p.add_argument("--sample-dir", default=None, help="override the config's sample_dir")
    p.add_argument("--limit", type=int, default=None, help="only do the next N rationales")
    args = p.parse_args()

    cfg = load_config(args.config)
    sample_dir = args.sample_dir or cfg["sample_dir"]
    path = output_path(sample_dir)
    j = pd.read_csv(path).reindex(columns=COLUMNS).astype({"kaleido": object})   # adds the column if missing
    todo = j.index[j.rationale.notna() & j.kaleido.isna()][:args.limit]
    if todo.empty:
        print("kaleido: nothing to do")
        return
    save_config(args.config, sample_dir)

    kaleido = Kaleido(cfg["model"])
    for k, i in enumerate(tqdm(todo, desc="kaleido", unit="rationale"), 1):
        j.loc[i, "kaleido"] = describe(kaleido, j.loc[i, "rationale"].strip(), cfg.get("n", 10))
        if k % 20 == 0 or k == len(todo):
            j.to_csv(path, index=False)
    print(f"kaleido: {len(todo)} rationales -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
