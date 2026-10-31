# ETHOS

Proof-of-concept pipeline for **Exploring Pluralistic LLMs for Social Impact Decision-Making** (COMPSCI 2800).
It compares a toxicity classifier, a base LLM, and a pluralistic LLM (Dynamic Moral Profiling) on content
moderation decisions for Civil Comments, including contextual variations of each comment.

## Pipeline

| Step | Script | Config | Output (`data/`) |
|---|---|---|---|
| 1. Sample comments | `scripts/extract_sample.py` | `configs/sample.yaml` | `processed/sample_n100/sample.csv` + `figures/` |
| 2. Generate variations | `scripts/shell/run_generate_variations.sh` | `configs/generation.yaml` | `processed/sample_n100/variations.csv` |
| 3. Judge | `scripts/shell/run_judge.sh` | `configs/judges.yaml` | `judgements/sample_n100/judgements.csv` |
| 4. Evaluate | `scripts/shell/run_eval.sh` | `configs/judges.yaml` | `judgements/sample_n100/eval/` |

- **Sample:** 100 top-level, deduplicated Civil Comments **test-split** comments (Detoxify was trained on train)
  with ≥30 annotators, 25 per human-toxicity bin (0–0.25, 0.25–0.5, 0.5–0.75, 0.75–1).
- **Variations:** consequentialist, emotional, and relational context added before/after each comment
  (after *On the Context Sensitivity of LLM Moral Judgment*). The original text is never modified.
- **Judges** (one row per judge per comment version in `judgements.csv`):
  - `human`: Civil Comments annotations (originals only); `score` = share of annotators rating it toxic, `decision` = remove if ≥ 0.5
  - `detoxify`: Detoxify `unbiased`, remove if score ≥ 0.5
  - `qwen3:8b-base`: base LLM, allow/remove + rationale
  - `qwen3:8b-dmp`: pluralistic LLM, conditioned on a sampled value profile (60-value taxonomy, *The Pluralistic Moral Gap*); the profile is in the `profile` column

Each step copies its config into the output folder (`log/`) for reproducibility. Re-running a step resumes where it stopped.

## Setup

```bash
uv sync                                   # Python 3.14 deps into .venv
ollama pull ministral-3:8b                # variation generator
ollama pull qwen3:8b                      # LLM judges
```

Raw data: Kaggle *Jigsaw Unintended Bias in Toxicity Classification*, `all_data.csv` in `data/unprocessed/` (`data/` is git-ignored).

## Running the code

1. **Start Ollama** (the app, or `ollama serve`). Both LLM steps call it locally.
2. **Pick the sample:** set `n`, the annotator minimum and the bins in `configs/sample.yaml`, then run
   `.venv/bin/python scripts/extract_sample.py`.
3. **Generate variations:** set the model and word limits in `configs/generation.yaml`, then run
   `scripts/shell/run_generate_variations.sh`.
4. **Judge:** set `sample_dir` and the judges in `configs/judges.yaml`, then run `scripts/shell/run_judge.sh`.
5. **Evaluate:** run `scripts/shell/run_eval.sh` for human alignment and context sensitivity
   (figures and CSVs in `data/judgements/<sample>/eval/`).

Add `--limit 3` to steps 3–4 for a quick test, or `--only <judge name>` to step 4 to run a single judge.
A stopped step picks up where it left off when you run it again. Thinking is off (`reasoning_effort: none`), so
each LLM judge takes about 15–20 min for 400 texts. Remove that line to get rationales with thinking, which takes about 2 h per judge.

## Proof-of-concept stand-ins

| Proposal | Here | Why |
|---|---|---|
| Mistral Small 4 (variations) | `ministral-3:8b` | runs on a 16 GB laptop |
| Evaluated LLM families | `qwen3:8b` | runs locally |
| Human-derived DMP prior (G0) | uniform over the 60 values | no human rationales in Civil Comments (proposal H1); see `src/ethos/llm/dmp.py` |
| Many profiles/samples per text | 1 call per text | time |

## Layout

```
configs/    sample.yaml, generation.yaml, judges.yaml
prompts/    variation + judge prompts, 60-value taxonomy
scripts/    extract_sample.py, generate_variations.py, judge.py, shell/
src/ethos/  judges/ (human, detoxify, base LLM, pluralistic), llm/ (Ollama client, DMP profiles), plotting/
notebooks/  exploratory analysis
```

**Note:** if `No module named 'ethos'` appears when calling `.venv/bin/python` directly, iCloud has hidden the
venv's `.pth` files: `chflags nohidden .venv/lib/python3.14/site-packages/*.pth` (the shell scripts handle this).
