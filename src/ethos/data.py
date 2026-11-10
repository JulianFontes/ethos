"""Data processing for Civil Comments, shared by the scripts in scripts/: load the raw Jigsaw files, filter the
train / test comments, compute consensus, bin and sample."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "unprocessed"


# --- loading and filtering -------------------------------------------------------------------------------------

def load_comments():
    """All comments with their labels (data/unprocessed/all_data.csv)."""
    return pd.read_csv(RAW / "all_data.csv")


def load_votes():
    """One row per annotator vote on toxicity and its attributes (train comments only)."""
    return pd.read_csv(RAW / "annotations" / "toxicity_individual_annotations.csv")


def filter_split(df, split, min_annotators, top_level_only=True):
    """Comments of one split with >= min_annotators toxicity annotators, optionally top-level only, deduplicated
    by stripped text (first occurrence kept). Keeps a `_text` column (stripped text) for overlap checks."""
    d = df[(df.split == split) & (df.toxicity_annotator_count >= min_annotators)]
    if top_level_only:
        d = d[d.parent_id.isna()]
    return d.assign(_text=d.comment_text.str.strip()).drop_duplicates("_text")


def build_splits(df, min_annotators, top_level_only=True, drop_test_overlap=True):
    """Filtered train and test comments with a `consensus` column.

    drop_test_overlap removes test comments whose text also appears in train. Returns (train, test, n_dropped).
    """
    train = filter_split(df, "train", min_annotators, top_level_only)
    test = filter_split(df, "test", min_annotators, top_level_only)
    n_dropped = 0
    if drop_test_overlap:
        overlap = test._text.isin(train._text)
        n_dropped = int(overlap.sum())
        test = test[~overlap]
    train, test = (add_consensus(d.drop(columns="_text")) for d in (train, test))
    return train, test, n_dropped


# --- consensus and bins ----------------------------------------------------------------------------------------

def consensus(p):
    """Share of annotators in the majority, max(p, 1 - p): 0.5 = even split, 1 = unanimous."""
    return np.maximum(p, 1 - p)


def add_consensus(df):
    """Return df with a `consensus` column computed from `toxicity`."""
    return df.assign(consensus=consensus(df.toxicity))


def bin_column(values, edges, label_format="{lo:g}-{hi:g}"):
    """Bin values into [lo, hi) intervals given by `edges`; the last bin also includes its top edge.
    Returns a categorical Series with labels like "0.5-0.6"."""
    edges = sorted(float(e) for e in edges)
    labels = [label_format.format(lo=lo, hi=hi) for lo, hi in zip(edges[:-1], edges[1:])]
    # nudge the top edge so a value exactly on it (e.g. consensus 1.0) lands in the last bin
    return pd.cut(values, edges[:-1] + [edges[-1] + 1e-9], right=False, labels=labels)


# --- sampling --------------------------------------------------------------------------------------------------

def sample_per_bin(pool, bin_col, per_bin, seed):
    """Draw `per_bin` rows from each category of `pool[bin_col]`, in category order, with a fixed seed.
    Raises ValueError naming the bins with fewer than `per_bin` rows."""
    labels = list(pool[bin_col].cat.categories)
    available = pool[bin_col].value_counts()
    short = [label for label in labels if available.get(label, 0) < per_bin]
    if short:
        raise ValueError(f"not enough comments in bin(s) {short}; lower per_bin or change the bins")
    return pd.concat(pool[pool[bin_col] == label].sample(per_bin, random_state=seed) for label in labels)


# --- sentence embeddings ---------------------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"   # general-purpose Sentence-BERT, not toxicity-trained

def embed(texts, model_name=EMBEDDING_MODEL, batch_size=32):
    """Sentence-BERT embeddings of `texts` (a Series indexed by comment id) as a DataFrame: one row per comment,
    columns emb_0 ... emb_<d-1>, unit length (so a dot product is the cosine similarity)."""
    from sentence_transformers import SentenceTransformer  # imported here: slow to load, only this step needs it

    model = SentenceTransformer(model_name)
    vectors = model.encode(texts.tolist(), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return pd.DataFrame(vectors, index=texts.index, columns=[f"emb_{i}" for i in range(vectors.shape[1])])

def load_embeddings(sample_dir=None):
    """Sentence-BERT embeddings indexed by comment id, from data/processed/comments/embeddings/embeddings.parquet (written by
    scripts/data/embed_comments.py). With `sample_dir`, only that sample's comments, in sample.csv order."""
    from ethos.paths import COMMENTS, EMBEDDINGS
    path = EMBEDDINGS / "embeddings.parquet"
    if sample_dir is None or (ROOT / sample_dir) == COMMENTS:
        return pd.read_parquet(path)
    ids = pd.read_csv(ROOT / sample_dir / "sample.csv", usecols=["id"]).id
    emb = pd.read_parquet(path, filters=[("id", "in", ids.tolist())])   # reads only the sample's rows
    return emb.loc[ids]                                                  # in sample.csv order
