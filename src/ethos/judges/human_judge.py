"""Human judge: the Civil Comments annotations, written as rows like any other judge.

score = share of annotators who rated the comment toxic (`toxicity` in sample.csv);
decision = "remove" when score >= threshold. Humans only labeled the original comments,
so this judge writes `version = original` rows only (no human labels exist for variations).
"""
import pandas as pd
from tqdm import tqdm

from ethos.judges.core import ROOT, append_rows, output_path, todo


def run(cfg, sample_dir, limit=None):
    name, threshold = cfg["name"], cfg.get("threshold", 0.5)
    texts = todo(sample_dir, name, limit)
    texts = texts[texts.version == "original"]
    if texts.empty:
        print(f"{name}: nothing to do")
        return

    scores = pd.read_csv(ROOT / sample_dir / "sample.csv", usecols=["id", "toxicity"]).set_index("id").toxicity
    append_rows(output_path(sample_dir), [
        {"id": row.id, "version": row.version, "text": row.text, "judge": name,
         "score": round(float(scores[row.id]), 4),
         "decision": "remove" if scores[row.id] >= threshold else "allow",
         "rationale": "", "profile": ""}
        for row in tqdm(texts.itertuples(index=False), total=len(texts), desc=name, unit="text")
    ])
