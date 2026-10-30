"""Generate consequentialist / emotional / relational variations of each comment.

Model, paths and sampling settings come from a YAML config (see configs/generation.yaml).
The model must already be pulled in Ollama; the script stops with an error otherwise.

Usage:
  python scripts/generate_variations.py
  python scripts/generate_variations.py --limit 5
  python scripts/generate_variations.py --config configs/my_generation.yaml

Re-running resumes: ids already in the output file are skipped.
"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from ethos.llm.mistral_client import DEFAULT_BASE_URL, chat_json, get_client

KEYS = ["consequentialist", "emotional", "relational"]

# One CSV row per comment; `error` is empty unless a call failed.
COLUMNS = (["id", "comment_text", "model", "valid"] + KEYS
           + [f"added_words_{k}" for k in KEYS] + ["error"])

# One call per variation type; the output is a single string.
SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}

ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "prompts"
DEFAULT_CONFIG = ROOT / "configs" / "generation.yaml"


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = {"model", "input", "output"} - cfg.keys()
    if missing:
        raise SystemExit(f"{path} is missing required keys: {sorted(missing)}")
    return cfg


def format_type_prompt(kind, spec):
    """The type-specific part of the system prompt: instruction plus worked examples."""
    lines = [f"Variation type: {kind}", spec["instruction"], "",
             "Examples (for format and length only; write new context, do not reuse their wording):"]
    for ex in spec.get("examples", []):
        lines += [f"Comment: {ex['comment']}", f"Variation: {ex['variation']}", ""]
    return "\n".join(lines).strip()


def load_prompts():
    """system.txt holds the rules shared by all calls; variations.json holds an instruction
    and examples per type."""
    system = (PROMPT_DIR / "system.txt").read_text().strip()
    specs = json.loads((PROMPT_DIR / "variations.json").read_text())
    missing = [k for k in KEYS if k not in specs]
    if missing:
        raise KeyError(f"prompts/variations.json is missing: {missing}")
    return system, {k: format_type_prompt(k, specs[k]) for k in KEYS}


SYSTEM, INSTRUCTIONS = load_prompts()


def keeps_original(variant, original):
    """True if the original comment text appears verbatim (ignoring case, spacing, trailing punctuation)."""
    norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
    core = norm(original).rstrip(".!?…\"' ")
    return bool(core) and core in norm(variant)


def generate_one(client, cfg, comment, kind):
    """One variation of one type. Returns (text, valid). Retries if the original text was altered."""
    system = SYSTEM + "\n\n" + INSTRUCTIONS[kind]
    text = ""
    for _ in range(cfg.get("retries", 2) + 1):
        out = chat_json(client, cfg["model"], system, f"Comment:\n{comment}", SCHEMA,
                        temperature=cfg.get("temperature", 0.5), max_tokens=cfg.get("max_tokens", 600))
        text = out.get("text", "")
        if keeps_original(text, comment):
            return text, True
    return text, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="YAML config (default: configs/generation.yaml)")
    p.add_argument("--limit", type=int, default=None, help="only process the first N comments")
    args = p.parse_args()
    cfg = load_config(args.config)
    model = cfg["model"]

    # Fail fast if Ollama is down or the model isn't pulled, before touching any files.
    client = get_client(model, cfg.get("base_url", DEFAULT_BASE_URL))

    df = pd.read_csv(ROOT / cfg["input"], usecols=["id", "comment_text"]).dropna()
    if args.limit:
        df = df.head(args.limit)

    out_path = ROOT / cfg["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_path.parent / "generation_config.yaml")
    done = set()
    if out_path.exists():
        # keep finished rows, drop failed ones so they are retried on this run
        with out_path.open(newline="") as f:
            kept = [r for r in csv.DictReader(f) if not r["error"]]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(kept)
        done = {int(r["id"]) for r in kept}

    todo = df[~df["id"].isin(done)]
    print(f"{len(todo)} to generate ({len(done)} already done) with {model}")

    new_file = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        n_valid = n_errors = 0
        bar = tqdm(todo.itertuples(index=False), total=len(todo), unit="comment")
        for row in bar:
            rec = {"id": int(row.id), "comment_text": row.comment_text, "model": model, "valid": True}
            errors = {}
            for kind in KEYS:
                try:
                    text, ok = generate_one(client, cfg, row.comment_text, kind)
                    rec[kind] = text
                    rec[f"added_words_{kind}"] = len(text.split()) - len(row.comment_text.split())
                    rec["valid"] = rec["valid"] and ok
                except Exception as e:  # network error, refusal, bad JSON
                    rec[kind], rec["valid"] = "", False
                    errors[kind] = str(e)[:200]
            rec["error"] = "; ".join(f"{k}: {v}" for k, v in errors.items())
            writer.writerow(rec)
            f.flush()
            n_valid += rec["valid"]
            n_errors += bool(errors)
            bar.set_postfix(valid=n_valid, errors=n_errors)


if __name__ == "__main__":
    main()
