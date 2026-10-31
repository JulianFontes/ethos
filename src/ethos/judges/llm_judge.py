"""Base LLM judge: one isolated call per comment version, returning allow/remove plus a rationale.

Works for any model served by Ollama (config type "ollama") or the OpenAI API (config type "openai"). `judge_texts` is the shared loop,
also used by the pluralistic judge (ethos.judges.pluralistic_judge).
Failed calls are skipped (not written) and retried on the next run.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ethos.judges.core import ROOT, append_rows, output_path, todo
from ethos.llm import ollama_client, openai_client
from ethos.llm.ollama_client import DEFAULT_BASE_URL

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
    llm = openai_client if cfg["type"].startswith("openai") else ollama_client
    client = llm.get_client(cfg["model"], cfg.get("base_url", DEFAULT_BASE_URL))  # fails fast if unavailable
    texts = todo(sample_dir, name, limit)
    out = output_path(sample_dir)
    if texts.empty:
        print(f"{name}: nothing to do")
        return

    def judge(row):
        system, profile = prompt_for(row)
        try:
            result = llm.chat_json(client, cfg["model"], system, f"Comment:\n{row.text}", DECISION_SCHEMA,
                                   temperature=cfg.get("temperature"), max_tokens=cfg.get("max_tokens"),
                                   reasoning_effort=cfg.get("reasoning_effort"))
        except Exception as e:  # connection error, bad JSON, refusal, rate limit
            return row, profile, e
        return row, profile, result

    # `workers` requests in flight at once (default 1); rows are written from this thread only, as they finish.
    failed = 0
    with ThreadPoolExecutor(max_workers=cfg.get("workers", 1)) as pool:
        futures = [pool.submit(judge, row) for row in texts.itertuples(index=False)]
        for future in tqdm(as_completed(futures), total=len(futures), desc=name, unit="text"):
            row, profile, result = future.result()
            if isinstance(result, Exception):
                failed += 1
                tqdm.write(f"{name}: failed {row.id}/{row.version}: {str(result)[:120]}")
                continue
            append_rows(out, [{"id": row.id, "version": row.version, "text": row.text, "judge": name,
                               "score": "", "decision": result["decision"], "rationale": result["rationale"],
                               "profile": profile}])
    if failed:
        print(f"{name}: {failed} failed; re-run to retry them")


def run(cfg, sample_dir, limit=None):
    system = (ROOT / cfg["prompt"]).read_text().strip()
    judge_texts(cfg, sample_dir, limit, lambda row: (system, ""))
