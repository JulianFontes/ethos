"""Build the similarity bank: cluster the 60 taxonomy values by meaning.

Pipeline: value + one-line definition (prompts/value_definitions.yaml) -> embedding -> cosine similarity ->
average-linkage clustering. k is picked by silhouette unless --k is given.

Writes
    prompts/value_banks/similarity.yaml         clusters + build settings (small; commit it, edit it by hand after review)
    prompts/value_banks/similarity_matrix.csv   60 x 60 cosine similarity
    data/value_banks/                           embeddings cache, dendrogram.png, similarity_heatmap.png (git-ignored)

Usage:
    python scripts/value_banks/build_similarity_bank.py                       # OpenAI text-embedding-3-large, k by silhouette
    python scripts/value_banks/build_similarity_bank.py --k 10
    python scripts/value_banks/build_similarity_bank.py --backend sbert       # local, needs sentence-transformers
    python scripts/value_banks/build_similarity_bank.py --backend tfidf --out-dir /tmp/bank_test --no-figures   # offline smoke test only

Then READ the printed clusters (and the "borderline values" list) and fix the yaml by hand where they look wrong.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ethos.paths import ROOT
from ethos.value_banks import (
    DEFINITIONS, BANK_DIR, choose_k, cluster, cosine_similarity, embed, label_clusters, silhouette, value_texts,
)

CACHE = ROOT / "data/value_banks"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="openai", choices=["openai", "sbert", "tfidf"])
    ap.add_argument("--model", help="embedding model (default depends on backend)")
    ap.add_argument("--k", type=int, help="number of clusters (default: best silhouette in --k-min..--k-max)")
    ap.add_argument("--k-min", type=int, default=5)
    ap.add_argument("--k-max", type=int, default=15)
    ap.add_argument("--definitions", default=DEFINITIONS)
    ap.add_argument("--out-dir", default=BANK_DIR, type=Path, help="default prompts/value_banks")
    ap.add_argument("--refresh", action="store_true", help="re-embed even if a cached embedding file exists")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    defs, texts = value_texts(args.definitions)
    values = list(defs)

    # embeddings (cached, since the OpenAI call costs money and the result does not change)
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{args.backend}_{(args.model or 'default').replace('/', '-')}"
    cache_file = CACHE / f"embeddings_{tag}.npy"
    if cache_file.exists() and not args.refresh:
        x = np.load(cache_file)
        print(f"loaded embeddings from {cache_file.relative_to(ROOT)}")
    else:
        x = embed(texts, args.backend, args.model)
        np.save(cache_file, x)
        print(f"embedded {len(texts)} values with {tag}")
    sim = cosine_similarity(x)

    # choose k, cluster
    if args.k:
        k, scores = args.k, {}
    else:
        k, scores = choose_k(sim, args.k_min, args.k_max)
        print("silhouette by k: " + "  ".join(f"{kk}:{s:.3f}" for kk, s in scores.items()))
    labels, z = cluster(sim, k)
    sil = silhouette(sim, labels)
    clusters = label_clusters(values, sim, labels)
    print(f"\nk = {k}, mean silhouette = {sil.mean():.3f}  (low is normal for 60 abstract values; judge by reading)\n")
    for c in clusters:
        print(f"[{c['id']:>2}] {c['label']:<22} ({len(c['values'])})  " + ", ".join(c["values"]))

    # values that sit weakly in their cluster: the ones to review by hand
    cid = {v: c["id"] for c in clusters for v in c["values"]}
    sims = pd.DataFrame(sim, index=values, columns=values)
    print("\nborderline values (lowest silhouette; check these first):")
    for i in np.argsort(sil)[:8]:
        v = values[i]
        nn = sims.loc[v].drop(v).sort_values(ascending=False).head(3)
        print(f"  {v:<22} in [{cid[v]}] {sil[i]:+.2f}   closest: " + ", ".join(f"{n} [{cid[n]}] {s:.2f}" for n, s in nn.items()))

    # save
    out = ROOT / args.out_dir  # absolute paths stay as given
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "backend": args.backend,
        "model": args.model or "default",
        "linkage": "average (cosine distance)",
        "k": int(k),
        "mean_silhouette": round(float(sil.mean()), 4),
        "note": "Hand-editable: SimilarityBank.load() reads the clusters below, not the matrix.",
        "clusters": [{"id": c["id"], "label": c["label"], "values": c["values"]} for c in clusters],
    }
    (out / "similarity.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    sims.round(4).to_csv(out / "similarity_matrix.csv")
    print(f"\nwrote {out / 'similarity.yaml'}\nwrote {out / 'similarity_matrix.csv'}")

    if not args.no_figures:
        figures(values, sim, z, labels, k, CACHE)


def figures(values, sim, z, labels, k, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(7, 11))
    cut = z[len(values) - k - 1, 2] if k > 1 else 0
    dendrogram(z, labels=values, orientation="left", leaf_font_size=9, color_threshold=cut, ax=ax)
    ax.set_xlabel("cosine distance")
    ax.set_title(f"Value similarity dendrogram (cut at k={k})")
    fig.tight_layout()
    fig.savefig(out / "dendrogram.png", dpi=150)
    plt.close(fig)

    order = np.argsort(labels, kind="stable")
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(sim[np.ix_(order, order)], cmap="viridis")
    ax.set_xticks(range(len(values)), [values[i] for i in order], rotation=90, fontsize=7)
    ax.set_yticks(range(len(values)), [values[i] for i in order], fontsize=7)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title("Value similarity (values ordered by cluster)")
    fig.tight_layout()
    fig.savefig(out / "similarity_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
