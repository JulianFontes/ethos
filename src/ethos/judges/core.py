"""Shared helpers for all judges.

Every judge writes rows to one long CSV, data/judgements/<sample>/judgements.csv:
    id, version, text, judge, score, decision, rationale, profile, kaleido, ratings
`judge` is the judge's name from configs/judges/judges.yaml (the human annotations are the `human` judge's rows);
`score` is the toxicity score (for LLM judges, ethos.scores.toxicity_score of its answers); `profile` is the moral
profile a DMP judge was given; `ratings` holds an LLM judge's individual answers (JSON list). `version` (always
"original") and `kaleido` (always empty) are kept so older judgements.csv files stay readable.
"""
import csv
import shutil
from pathlib import Path

import pandas as pd
import yaml

from ethos.paths import ROOT

COLUMNS = ["id", "version", "text", "judge", "score", "decision", "rationale", "profile", "kaleido", "ratings"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


LLM_TYPES = {"ollama_minimal", "ollama_minimal_dmp"}


def load_judges(path):
    """Judge configs from configs/judges/judges.yaml; LLM judges are merged over `llm_defaults`."""
    cfg = load_config(path)
    defaults = cfg.get("llm_defaults", {})
    return [{**defaults, **judge} if judge["type"] in LLM_TYPES else judge for judge in cfg["judges"]]


def load_texts(sample_dir):
    """The sample's comments: id, version ("original"), text."""
    texts = pd.read_csv(ROOT / sample_dir / "sample.csv", usecols=["id", "comment_text"])
    texts = texts.rename(columns={"comment_text": "text"}).assign(version="original")
    return texts[["id", "version", "text"]].sort_values("id", ignore_index=True)


def output_path(sample_dir):
    """data/judgements/<sample name>/judgements.csv"""
    return ROOT / "data" / "judgements" / Path(sample_dir).name / "judgements.csv"


def todo(sample_dir, judge, limit=None, query=None):
    """Comments this judge still has to do.

    Optionally only comments whose sample.csv row matches `query` (a pandas query, e.g. "consensus < 0.7"),
    and only the first `limit` comments.
    """
    versions = load_texts(sample_dir)
    if query:
        keep = pd.read_csv(ROOT / sample_dir / "sample.csv").query(query).id
        versions = versions[versions.id.isin(keep)]
    if limit:
        versions = versions[versions.id.isin(versions.id.unique()[:limit])]
    path = output_path(sample_dir)
    if path.exists():
        done = pd.read_csv(path, usecols=["id", "version", "judge"])
        done = set(zip(*done[done.judge == judge][["id", "version"]].to_numpy().T))
        versions = versions[[(i, v) not in done for i, v in zip(versions.id, versions.version)]]
    return versions


def append_rows(path, rows):
    """Add rows to the judgements CSV, writing the header if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and pd.read_csv(path, nrows=0).columns.tolist() != COLUMNS:
        # older file without newer columns (e.g. `profile`): add them empty before appending
        pd.read_csv(path).reindex(columns=COLUMNS).to_csv(path, index=False)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def save_config(config_path, sample_dir):
    """Copy a judge's config into the output's log/ folder."""
    log_dir = output_path(sample_dir).parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, log_dir / Path(config_path).name)
