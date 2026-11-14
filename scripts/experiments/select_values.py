"""Ask an LLM (default gpt-6-luna) to pick the 5 moral values (of the 31 VISPA values) most relevant to moderating
each unique comment in the sample. Writes one JSON line per comment, and skips comments already in the output.

Usage:
    python scripts/experiments/select_values.py                # 1 comment (quick check)
    python scripts/experiments/select_values.py --limit 0      # all unique comments, 16 calls at a time
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ethos.llm import connect

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/processed/civil-comments/consensus_n2000_train/sample.csv"
PROMPT = ROOT / "prompts/value_selection/select_top5.txt"
OUT = ROOT / "data/processed/civil-comments/consensus_n2000_train/value_selection/{model}.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--limit", type=int, default=1, help="number of comments (0 = all)")
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=16, help="parallel API calls")
    args = p.parse_args()
    out = args.out or Path(str(OUT).format(model=args.model.replace(":", "_").replace("/", "_")))
    out.parent.mkdir(parents=True, exist_ok=True)

    values = sorted(f.name.removesuffix("_context_controlled.csv").replace("_", " ").title()
                    for f in (ROOT / "data/VISPA-values").glob("*_context_controlled.csv"))
    template = PROMPT.read_text().replace("{values_31}", "\n".join(f"- {v}" for v in values))
    schema = {"type": "object", "additionalProperties": False, "required": ["values"],
              "properties": {"values": {"type": "array", "items": {"type": "string", "enum": values}}}}

    comments = pd.read_csv(SAMPLE).drop_duplicates("comment_text")[["id", "comment_text"]]
    if args.limit:
        comments = comments.head(args.limit)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    todo = comments[~comments["id"].isin(done)]

    backend, client, model = connect(args.model)
    def ask(row):
        answer = backend.chat_json(client, model, system="",
                                   user=template.replace("{comment}", row.comment_text), schema=schema)
        return row, list(dict.fromkeys(answer["values"]))   # drop repeats, keep order

    with out.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(ask, row) for row in todo.itertuples()]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            row, chosen = fut.result()
            if len(chosen) != 5:
                print(f"warning: comment {row.id} got {len(chosen)} values: {chosen}")
            f.write(json.dumps({"id": row.id, "comment": row.comment_text, "values": chosen}) + "\n")
            f.flush()
    print(f"{len(todo)} new, {len(done)} already done -> {out}")


if __name__ == "__main__":
    main()
