"""Base LLM judge: one isolated call per comment version, returning allow/remove plus a rationale.

Works for any model served by Ollama (config type "ollama"). `judge_texts` is the shared loop,
also used by the pluralistic judge (ethos.judges.pluralistic_judge).
Failed calls are skipped (not written) and retried on the next run.
"""
from tqdm import tqdm

from ethos.judges.core import ROOT, append_rows, output_path, todo
from ethos.llm.ollama_client import DEFAULT_BASE_URL, chat_json, get_client

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["allow", "remove"]},
        "rationale": {"type": "string"},
    },
    "required": ["decision", "rationale"],
    "additionalProperties": False,
}


def judge_texts(cfg, sample_dir, limit, prompt_for):
    """Judge every remaining comment version. `prompt_for(row)` returns (system_prompt, profile)."""
    name = cfg["name"]
    client = get_client(cfg["model"], cfg.get("base_url", DEFAULT_BASE_URL))  # fails fast if not pulled
    texts = todo(sample_dir, name, limit)
    out = output_path(sample_dir)
    if texts.empty:
        print(f"{name}: nothing to do")
        return

    failed = 0
    for row in tqdm(texts.itertuples(index=False), total=len(texts), desc=name, unit="text"):
        system, profile = prompt_for(row)
        try:
            result = chat_json(client, cfg["model"], system, f"Comment:\n{row.text}", DECISION_SCHEMA,
                               temperature=cfg.get("temperature"), max_tokens=cfg.get("max_tokens"),
                               reasoning_effort=cfg.get("reasoning_effort"))
        except Exception as e:  # connection error, bad JSON, refusal
            failed += 1
            tqdm.write(f"{name}: failed {row.id}/{row.version}: {str(e)[:120]}")
            continue
        append_rows(out, [{"id": row.id, "version": row.version, "text": row.text, "judge": name,
                           "score": "", "decision": result["decision"], "rationale": result["rationale"],
                           "profile": profile}])
    if failed:
        print(f"{name}: {failed} failed; re-run to retry them")


def run(cfg, sample_dir, limit=None):
    system = (ROOT / cfg["prompt"]).read_text().strip()
    judge_texts(cfg, sample_dir, limit, lambda row: (system, ""))
