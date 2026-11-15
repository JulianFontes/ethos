"""Similarity bank over the 60-value taxonomy: which values mean about the same thing.

Build it once with scripts/value_banks/build_similarity_bank.py, which writes
    prompts/value_banks/similarity.yaml        clusters (hand-editable after review) + build settings
    prompts/value_banks/similarity_matrix.csv  60 x 60 cosine similarity between values

Then use it from experiments:

    from ethos.value_banks import SimilarityBank
    bank = SimilarityBank.load()
    bank.cluster_of("Care")                        # cluster id
    bank.same_cluster("Care", "Compassion")        # True / False
    bank.similarity("Care", "Compassion")          # cosine similarity
    bank.neighbors("Care", k=5)                    # [(value, similarity), ...]
    bank.clusters_in(["Care", "Privacy", "Tradition"])   # set of cluster ids, e.g. a profile's breadth

The clusters in similarity.yaml are the source of truth: if the team merges, splits or moves values by hand
after reviewing them, edit the yaml and `SimilarityBank.load()` picks the edits up.
"""
import numpy as np
import pandas as pd
import yaml

from ethos.paths import ROOT

BANK_DIR = ROOT / "prompts/value_banks"
DEFINITIONS = ROOT / "prompts/value_definitions.yaml"


# --- embeddings ------------------------------------------------------------------------------------------------

def value_texts(definitions_path=DEFINITIONS):
    """({value: definition}, [text to embed per value]), in the definitions file's order."""
    defs = yaml.safe_load(open(definitions_path))
    return defs, [f"{v}: {d}" for v, d in defs.items()]


def embed(texts, backend="openai", model=None):
    """Unit-length embedding rows, one per text.

    openai  OpenAI embeddings (default text-embedding-3-large, the model the DMP paper used for its clustering)
    sbert   sentence-transformers, runs locally (default all-mpnet-base-v2); needs `uv add sentence-transformers`
    tfidf   word counts only, no model: an offline smoke test of the pipeline, NOT good enough for the real bank
    """
    if backend == "openai":
        from ethos.llm.openai_client import get_client

        model = model or "text-embedding-3-large"
        resp = get_client(model).embeddings.create(model=model, input=texts)
        x = np.array([d.embedding for d in resp.data], dtype=float)
    elif backend == "sbert":
        from sentence_transformers import SentenceTransformer

        x = SentenceTransformer(model or "all-mpnet-base-v2").encode(texts)
    elif backend == "tfidf":
        x = _tfidf(texts)
    else:
        raise ValueError(f"unknown backend {backend!r} (openai, sbert, tfidf)")
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _tfidf(texts):
    import re

    docs = [re.findall(r"[a-z]+", t.lower()) for t in texts]
    vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
    tf = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for w in d:
            tf[i, vocab[w]] += 1
    idf = np.log((1 + len(docs)) / (1 + (tf > 0).sum(0))) + 1
    return tf * idf


# --- clustering ------------------------------------------------------------------------------------------------

def cosine_similarity(x):
    """Pairwise cosine similarity of unit-length rows."""
    return np.clip(x @ x.T, -1, 1)


def cluster(sim, k):
    """Average-linkage agglomerative clustering into k clusters. Returns (labels 0..k-1, linkage matrix)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    dist = 1 - sim
    np.fill_diagonal(dist, 0)
    z = linkage(squareform(dist, checks=False), method="average")
    return fcluster(z, k, criterion="maxclust") - 1, z


def silhouette(sim, labels):
    """Per-item silhouette on cosine distance (singletons score 0). Higher = sits more clearly in its cluster."""
    dist = 1 - sim
    out = np.zeros(len(labels))
    for i in range(len(labels)):
        same = (labels == labels[i]) & (np.arange(len(labels)) != i)
        if not same.any():
            continue
        a = dist[i, same].mean()
        b = min(dist[i, labels == c].mean() for c in set(labels) if c != labels[i])
        out[i] = (b - a) / max(a, b)
    return out


def choose_k(sim, k_min=5, k_max=15):
    """(best k, {k: mean silhouette}). Expect low scores with 60 abstract values (the DMP paper's was 0.09), so
    treat this as a starting point and confirm by reading the clusters."""
    scores = {k: silhouette(sim, cluster(sim, k)[0]).mean() for k in range(k_min, k_max + 1)}
    return max(scores, key=scores.get), scores


def label_clusters(values, sim, labels):
    """Clusters as [{id, label, values}], largest first. label = the most central member (medoid)."""
    out = []
    for c in sorted(set(labels), key=lambda c: -(labels == c).sum()):
        idx = np.where(labels == c)[0]
        medoid = idx[np.argmax(sim[np.ix_(idx, idx)].mean(1))]
        out.append({"label": values[medoid], "values": sorted(values[i] for i in idx)})
    for n, c in enumerate(out, 1):
        c["id"] = n
    return out


# --- the bank --------------------------------------------------------------------------------------------------

class SimilarityBank:
    def __init__(self, clusters, matrix, meta=None):
        self.clusters = clusters  # [{id, label, values}]
        self.matrix = matrix  # DataFrame, value x value
        self.meta = meta or {}
        self._cluster_of = {v: c["id"] for c in clusters for v in c["values"]}

    @classmethod
    def load(cls, directory=BANK_DIR):
        meta = yaml.safe_load(open(directory / "similarity.yaml"))
        matrix = pd.read_csv(directory / "similarity_matrix.csv", index_col=0)
        return cls(meta.pop("clusters"), matrix, meta)

    def cluster_of(self, value):
        return self._cluster_of[value]

    def members(self, cluster_id):
        return next(c["values"] for c in self.clusters if c["id"] == cluster_id)

    def same_cluster(self, a, b):
        return self._cluster_of[a] == self._cluster_of[b]

    def similarity(self, a, b):
        return float(self.matrix.loc[a, b])

    def neighbors(self, value, k=5):
        s = self.matrix.loc[value].drop(value).sort_values(ascending=False)
        return list(s.head(k).items())

    def clusters_in(self, values):
        """Set of cluster ids touched by a list of values, e.g. the breadth of a profile or a rationale."""
        return {self._cluster_of[v] for v in values}
