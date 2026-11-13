"""Sentence-BERT embeddings of every comment in data/processed/civil-comments/all_comments.

Reads comments.csv and writes embeddings/embeddings.parquet: one row per comment (index `id`), columns
emb_0 ... emb_767 (all-mpnet-base-v2, unit length, float32). ethos.data.load_embeddings() reads it, and with a sample
folder (e.g. load_embeddings(TRAIN_SAMPLE)) returns just that sample's comments, so samples need no file of their own.

The model is general-purpose, not trained on toxicity, so it does not leak the labels Detoxify was trained on.
About 9 minutes for 38.5k comments on an Apple GPU (the first run downloads the model).

Usage:
    python scripts/data/embed_comments.py
"""
import argparse

import pandas as pd

from ethos.data import EMBEDDING_MODEL, embed
from ethos.paths import COMMENTS, EMBEDDINGS, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=EMBEDDING_MODEL)
    args = p.parse_args()

    texts = pd.read_csv(COMMENTS / "comments.csv", usecols=["id", "comment_text"]).set_index("id").comment_text
    emb = embed(texts, args.model)
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    emb.astype("float32").to_parquet(EMBEDDINGS / "embeddings.parquet")
    print(f"{len(emb):,} comments x {emb.shape[1]} dimensions -> {(EMBEDDINGS / 'embeddings.parquet').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
