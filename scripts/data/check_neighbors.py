"""Check the embeddings by eye: the nearest neighbours of a few random comments.

Picks `--n` random comments from <folder>, finds each one's `--k` nearest neighbours in the same folder by cosine
similarity (embeddings are unit length, so cosine = dot product), and writes them to
data/experiments/civil-comments/embeddings/<folder name>/neighbors.md to read, plus sources.yaml.
Each comment is shown with its id, split and human toxicity score.

Run scripts/data/embed_comments.py on the folder first.

Usage:
    python scripts/data/check_neighbors.py                                  # data/processed/civil-comments/comments, 10 comments, 5 neighbours
    python scripts/data/check_neighbors.py --folder data/processed/civil-comments/consensus_n125_train --seed 1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ethos.data import EMBEDDING_MODEL, load_embeddings
from ethos.paths import EMBEDDINGS, EXPERIMENTS, ROOT
from ethos.sources import write_sources


def comment_block(text, max_chars=None):
    """The comment as a Markdown quote, whitespace tidied (and cut to max_chars if given)."""
    text = " ".join(str(text).split())
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " [...]"
    return "> " + text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", type=Path, default=ROOT / "data/processed/civil-comments/comments",
                   help="data/processed/civil-comments/comments, or a sample folder (neighbours within that sample)")
    p.add_argument("--n", type=int, default=10, help="random comments to check")
    p.add_argument("--k", type=int, default=5, help="nearest neighbours per comment")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="default: data/experiments/civil-comments/embeddings/<folder name>/neighbors.md")
    args = p.parse_args()
    folder = ROOT / args.folder

    emb = load_embeddings(folder)
    csv = folder / "sample.csv" if (folder / "sample.csv").exists() else folder / "comments.csv"
    comments = pd.read_csv(csv, usecols=["id", "comment_text", "toxicity", "split"]).set_index("id").loc[emb.index]

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(emb), args.n, replace=False)
    vectors = emb.to_numpy()
    sims = vectors[picks] @ vectors.T                      # (n, all comments) cosine similarities
    sims[np.arange(args.n), picks] = -np.inf               # a comment is not its own neighbour

    lines = [f"# Nearest neighbours of {args.n} random comments",
             "",
             f"Embeddings: `{EMBEDDING_MODEL}` · {len(emb):,} comments in `{folder.relative_to(ROOT)}` · "
             f"{args.k} neighbours each by cosine similarity · seed {args.seed}",
             "",
             "Human = share of annotators who rated the comment toxic.",
             ""]
    for n, i in enumerate(picks, 1):
        row = comments.iloc[i]
        lines += ["---", "",
                  f"## {n}. Comment {comments.index[i]} ({row.split}, human {row.toxicity:.2f})", "",
                  comment_block(row.comment_text), "",
                  "| # | Similarity | Human | Split | Comment |",
                  "|---|---|---|---|---|"]
        for rank, j in enumerate(np.argsort(-sims[n - 1])[:args.k], 1):
            nb = comments.iloc[j]
            text = comment_block(nb.comment_text, 300)[2:].replace("|", "\\|")
            lines.append(f"| {rank} | {sims[n - 1, j]:.2f} | {nb.toxicity:.2f} | {nb.split} | {text} |")
        lines.append("")

    out = args.out or EXPERIMENTS / "embeddings" / folder.name / "neighbors.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    write_sources(out.parent, [EMBEDDINGS / "embeddings.parquet", csv])
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
