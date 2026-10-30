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

# One call per variation type. The model writes only the added context and where it goes;
# the script attaches it to the untouched original comment.
SCHEMA = {
    "type": "object",
    "properties": {
        "position": {"type": "string", "enum": ["before", "after"]},
        "context": {"type": "string"},
    },
    "required": ["position", "context"],
    "additionalProperties": False,
}

ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "prompts"
DEFAULT_CONFIG = ROOT / "configs" / "generation.yaml"


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = {"model", "input", "output", "max_added_words"} - cfg.keys()
    if missing:
        raise SystemExit(f"{path} is missing required keys: {sorted(missing)}")
    missing = [k for k in KEYS if k not in cfg["max_added_words"]]
    if missing:
        raise SystemExit(f"{path}: max_added_words is missing {missing}")
    return cfg


def split_example(comment, variation):
    """Turn a full example variation into (position, context) by locating the comment in it."""
    if variation.startswith(comment):
        return "after", variation[len(comment):].strip()
    if variation.endswith(comment):
        return "before", variation[:-len(comment)].strip()
    raise ValueError(f"example variation must start or end with its comment: {variation!r}")


def format_type_prompt(kind, spec):
    """The type-specific part of the system prompt: instruction plus worked examples."""
    lines = [f"Variation type: {kind}", spec["instruction"], "",
             "Examples (for format and length only; write new context, do not reuse their wording):"]
    for ex in spec.get("examples", []):
        position, context = split_example(ex["comment"], ex["variation"])
        output = json.dumps({"position": position, "context": context}, ensure_ascii=False)
        lines += [f"Comment: {ex['comment']}", f"Output: {output}", ""]
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


def usable_context(context, comment, max_words):
    """False if the context is empty, too long, or copies the start of the comment."""
    norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
    return (bool(norm(context)) and len(context.split()) <= max_words
            and norm(comment)[:40] not in norm(context))


def check_examples(max_added_words):
    """Stop early if a prompt example breaks its own type's word limit."""
    specs = json.loads((PROMPT_DIR / "variations.json").read_text())
    for kind in KEYS:
        for ex in specs[kind].get("examples", []):
            _, context = split_example(ex["comment"], ex["variation"])
            if len(context.split()) > max_added_words[kind]:
                raise SystemExit(f"{kind} example exceeds {max_added_words[kind]} words: {context!r}")


def generate_one(client, cfg, comment, kind):
    """One variation of one type. Returns (text, context, valid); retries unusable context."""
    max_words = cfg["max_added_words"][kind]
    system = f"{SYSTEM}\n\n{INSTRUCTIONS[kind]}\n\nWord limit: the context must be at most {max_words} words."
    position, context = "before", ""
    for _ in range(cfg.get("retries", 2) + 1):
        out = chat_json(client, cfg["model"], system, f"Comment:\n{comment}", SCHEMA,
                        temperature=cfg.get("temperature", 0.5), max_tokens=cfg.get("max_tokens", 600))
        position, context = out.get("position", "before"), out.get("context", "").strip()
        if usable_context(context, comment, max_words):
            break
    text = f"{context} {comment}" if position == "before" else f"{comment} {context}"
    return text, context, usable_context(context, comment, max_words)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="YAML config (default: configs/generation.yaml)")
    p.add_argument("--limit", type=int, default=None, help="only process the first N comments")
    args = p.parse_args()
    cfg = load_config(args.config)
    check_examples(cfg["max_added_words"])
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
                    text, context, ok = generate_one(client, cfg, row.comment_text, kind)
                    rec[kind] = text
                    rec[f"added_words_{kind}"] = len(context.split())
                    rec["valid"] = rec["valid"] and ok
                except Exception as e:  # network error, refusal, bad JSON
                    rec[kind], rec["valid"] = "", False
                    errors[kind] = str(e)[:200]
            rec["error"] = "; ".join(f"{k}: {v}" for k, v in errors.items())
            writer.writerow(rec)
            f.flush()
            n_valid += bool(rec["valid"])
            n_errors += bool(errors)
            bar.set_postfix(valid=n_valid, errors=n_errors)


if __name__ == "__main__":
    main()
