"""Shared helpers for all judges.

Every judge writes rows to one long CSV, data/judgements/<sample>/judgements.csv:
    id, version, text, judge, score, decision, rationale, profile
`version` is "original" or one of the variation types; `judge` is the judge's name from its config
(the human annotations are the `human` judge's rows, originals only); `profile` is the value profile
a pluralistic judge was given (empty for other judges).
"""
import csv
import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
VERSIONS = ["original", "consequentialist", "emotional", "relational"]
COLUMNS = ["id", "version", "text", "judge", "score", "decision", "rationale", "profile"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


LLM_TYPES = {"ollama", "ollama_dmp"}


def load_judges(path):
    """Judge configs from configs/judges.yaml; LLM judges are merged over `llm_defaults`."""
    cfg = load_config(path)
    defaults = cfg.get("llm_defaults", {})
    return [{**defaults, **judge} if judge["type"] in LLM_TYPES else judge for judge in cfg["judges"]]


def load_versions(sample_dir):
    """One row per comment version: id, version, text (original plus each variation).

    Everything comes from variations.csv, whose `comment_text` column is the original comment.
    """
    wide = pd.read_csv(ROOT / sample_dir / "variations.csv", usecols=["id", "comment_text"] + VERSIONS[1:])
    wide = wide.rename(columns={"comment_text": "original"})
    long = wide.melt(id_vars="id", value_vars=VERSIONS, var_name="version", value_name="text")
    return long.dropna(subset=["text"]).sort_values(["id", "version"], ignore_index=True)


def output_path(sample_dir):
    """data/judgements/<sample name>/judgements.csv"""
    return ROOT / "data" / "judgements" / Path(sample_dir).name / "judgements.csv"


def todo(sample_dir, judge, limit=None):
    """Comment versions this judge still has to do (optionally only the first `limit` comments)."""
    versions = load_versions(sample_dir)
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
