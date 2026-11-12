"""Data processing for Civil Comments, shared by the scripts in scripts/: load the raw Jigsaw files, filter the
train / test comments, compute consensus, bin and sample, embed comments, and load the value policy's training
data."""
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


# --- value policy training data --------------------------------------------------------------------------------
# The single-value run (scripts/experiments/profile_signal.py --single-values): every comment rated 10 times with each
# of the 60 taxonomy values alone, plus no value ("none").

N_VALUES, N_ANSWERS = 60, 10

def load_scores(path, include_none=False):
    """Read a single-value scores.csv (e.g. data/processed/consensus_n125_train/single_values/scores.csv, or a checkpoint
    of a larger run) and return (rows, base):

    rows  one row per (comment, option): id, moral_value, llm_toxicity, human_toxicity, consensus_bin, and the target
          distance = |llm_toxicity - human_toxicity|. The options are the 60 values, plus "none" (no value in the
          prompt) as a 61st when `include_none`, so a policy can also choose to leave the prompt as it is.
    base  the same columns for the no-value rows (moral_value "none"), one per comment; the no-value baseline
    """
    scores = pd.read_csv(ROOT / path)
    cols = ["id", "moral_value", "llm_toxicity", "human_toxicity", "consensus_bin"]
    scores = scores[cols].assign(distance=lambda d: (d.llm_toxicity - d.human_toxicity).abs())
    base = scores[scores.moral_value == "none"].set_index("id")
    rows = scores if include_none else scores[scores.moral_value != "none"]
    rows = rows.sort_values(["id", "moral_value"], ignore_index=True)

    # fail loudly if the table is not what the policy expects
    n_options = N_VALUES + include_none
    n_comments = rows.id.nunique()
    per_comment = rows.groupby("id").moral_value.nunique()
    assert (per_comment == n_options).all(), f"expected {n_options} options per comment, got {per_comment.unique()}"
    assert set(base.index) == set(rows.id), "every comment needs a no-value ('none') row"
    assert not rows.isna().any().any() and not base.isna().any().any(), "missing values in scores.csv"
    assert len(rows) == n_comments * n_options
    return rows, base


def feature_matrix(rows, embeddings):
    """Model inputs, one row per row of `rows`: the comment's embedding (emb_0 ... emb_767) followed by one column per
    option (1 for the row's value, 0 otherwise; value_none when "none" is an option). No scores: the human score and
    the LLM's answers stay out."""
    missing = set(rows.id) - set(embeddings.index)
    assert not missing, f"{len(missing)} comments have no embedding"
    emb = embeddings.loc[rows.id].to_numpy()
    values = pd.get_dummies(rows.moral_value, prefix="value", dtype=np.float32)
    return pd.DataFrame(np.hstack([emb, values.to_numpy()]), index=rows.index,
                        columns=list(embeddings.columns) + list(values.columns))
