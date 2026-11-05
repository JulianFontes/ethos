"""Detoxify judge: toxicity score, and "remove" when the score is at or above a threshold.

Uses Detoxify's "unbiased" model by default (trained on the Civil Comments train split;
our sample is test-only).
"""
import torch
from detoxify import Detoxify
from tqdm import tqdm

from ethos.judges.core import append_rows, output_path, todo

BATCH_SIZE = 32


def run(cfg, sample_dir, limit=None):
    name, threshold = cfg["name"], cfg.get("threshold", 0.5)
    texts = todo(sample_dir, name, limit)
    out = output_path(sample_dir)
    if texts.empty:
        print(f"{name}: nothing to do")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Detoxify(cfg.get("checkpoint", "unbiased"), device=device)
    with tqdm(total=len(texts), desc=name, unit="text") as bar:
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts.iloc[start:start + BATCH_SIZE]
            scores = model.predict(list(batch.text))["toxicity"]
            append_rows(out, [
                {"id": row.id, "version": row.version, "text": row.text, "judge": name,
                 "score": round(float(score), 4), "decision": "remove" if score >= threshold else "allow",
                 "rationale": "", "profile": ""}
                for row, score in zip(batch.itertuples(index=False), scores)
            ])
            bar.update(len(batch))
