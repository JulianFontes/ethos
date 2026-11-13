"""Why do some LLM calls in a profile_signal.py run fail? Re-ask the failed ones and keep the raw responses.

The failed calls of a run are its gaps: (comment, condition) pairs with fewer answers than the rest, in comments that
are otherwise done. This asks each gap pair `--repeats` times, plus as many randomly chosen pairs that did succeed
(the control group), with exactly the request profile_signal.py sends, and records for every response: why
generation stopped (finish_reason), token counts, reasoning length, whether the content parses as the expected JSON,
and the content itself when it does not.

Needs a running vLLM server (VLLM_BASE_URL); scripts/shell/debug_failures.sh starts one in a gpu_test job.
Writes <out>/responses.csv (one row per response) and prints a summary.

Usage:
    python scripts/experiments/debug_failures.py --data data/processed/consensus_n1000_train/single_values_gpt_oss_20b \
        --model openai/gpt-oss-20b --reasoning-effort low --out <data>/debug
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ethos.judges.llm_judge import SCHEMA
from ethos.llm import vllm_client
from ethos.llm.dmp import load_values
from ethos.paths import ROOT
from ethos.steering.prompt import prompt_conditions, system_prompt   # same prompts and conditions as the run


def gap_pairs(data, samples, n_conds):
    """(id, condition) pairs with fewer than `samples` answers, in comments that are otherwise nearly complete
    (missing at most 5% of their answers, so comments still in progress when the job stopped are left out)."""
    r = pd.concat(pd.read_csv(f, usecols=["id", "condition"]) for f in sorted((data / "jobs").glob("*.csv")))
    per_pair = r.groupby(["id", "condition"]).size()
    per_comment = r.groupby("id").size()
    finished = per_comment[per_comment >= 0.95 * samples * n_conds].index
    pairs = per_pair.reindex(pd.MultiIndex.from_product([finished, per_pair.index.levels[1]]), fill_value=0)
    return pairs[pairs < samples].index.tolist(), pairs[pairs == samples].index.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="the run's data folder (with jobs/*.csv)")
    p.add_argument("--sample", type=Path, default=Path("data/processed/consensus_n1000_train/sample.csv"))
    p.add_argument("--model", required=True, help="Hugging Face id served by vLLM, e.g. openai/gpt-oss-20b")
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--samples", type=int, default=10, help="answers per comment and condition in the run")
    p.add_argument("--repeats", type=int, default=5, help="times to ask each pair")
    p.add_argument("--max-pairs", type=int, default=600, help="at most this many gap pairs (and as many controls)")
    p.add_argument("--workers", type=int, default=256)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data, out = ROOT / args.data, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    conds = prompt_conditions(load_values("prompts/profiles/taxonomy-values.txt"), single_values=True)
    texts = pd.read_csv(ROOT / args.sample).set_index("id").comment_text
    gaps, ok = gap_pairs(data, args.samples, len(conds))
    rng = random.Random(0)
    gaps = rng.sample(gaps, min(len(gaps), args.max_pairs))
    controls = rng.sample(ok, len(gaps))
    print(f"{len(gaps)} gap pairs and {len(controls)} control pairs, {args.repeats} times each")

    client = vllm_client.get_client(args.model)
    extra = ({"chat_template_kwargs": {"enable_thinking": False}} if args.reasoning_effort == "none"
             else {"reasoning_effort": args.reasoning_effort})

    def ask(task):
        group, (cid, c) = task
        system = system_prompt(conds[c])
        row = {"group": group, "id": cid, "condition": c}
        try:
            resp = client.chat.completions.create(
                model=args.model, temperature=1.0, extra_body=extra,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": f"Comment:\n{texts[cid]}"}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "output", "schema": SCHEMA, "strict": True}})
        except Exception as e:                          # the HTTP call itself failed
            return row | {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        choice, msg = resp.choices[0], resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        content = msg.content or ""
        row |= {"finish_reason": choice.finish_reason, "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens, "reasoning_chars": len(reasoning),
                "content_chars": len(content)}
        try:
            answer = json.loads(content)
            assert answer.get("evaluation") in ("TOXIC", "NOT TOXIC"), "evaluation not TOXIC / NOT TOXIC"
            return row | {"ok": True}
        except Exception as e:                          # what profile_signal.py would count as a failure
            return row | {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}",
                          "content": content[:500], "reasoning_tail": reasoning[-300:]}

    tasks = [("gap", pr) for pr in gaps for _ in range(args.repeats)] + \
            [("control", pr) for pr in controls for _ in range(args.repeats)]
    rng.shuffle(tasks)
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(tqdm(pool.map(ask, tasks), total=len(tasks), unit="call"))
    df = pd.DataFrame(rows)
    df.to_csv(out / "responses.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nFailure rate by group:")
    print(df.groupby("group").ok.agg(calls="size", failed=lambda s: int((~s).sum()),
                                     rate=lambda s: round(1 - s.mean(), 4)).to_string())
    bad = df[~df.ok]
    if len(bad):
        print("\nFailures by finish_reason:", bad.finish_reason.fillna("(no response)").value_counts().to_dict())
        print("Failure reasons:")
        print(bad.error.str[:120].value_counts().head(8).to_string())
        print("\nCompletion tokens, failed vs ok:")
        print(df.groupby("ok").completion_tokens.describe(percentiles=[.5, .9, .99]).round(0).to_string())
        print("\nFailures by condition (top 10):", bad.condition.value_counts().head(10).to_dict())
        print("\nExamples of failed content:")
        for _, r in bad.head(3).iterrows():
            print(f"--- {r.finish_reason}, {r.get('completion_tokens')} tokens: content={r.get('content')!r:.300}")
    print(f"\nwrote {(out / 'responses.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
