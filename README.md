# ETHOS

Proof-of-concept for **Exploring Pluralistic LLMs for Social Impact Decision-Making** (COMPSCI 2800). It asks
whether steering an LLM toxicity moderator with moral value profiles (Dynamic Moral Profiling, DMP) brings its
judgements closer to human annotators on Civil Comments, especially on comments the annotators disagree about.

> **Content warning:** Civil Comments contains toxic language, and so do some prompts and outputs.

## Key terms

- **Human score:** share of a comment's annotators who rated it toxic.
- **Consensus:** `max(p, 1 - p)` of the human score; 0.5 = annotators split evenly, 1 = unanimous.
- **LLM toxicity score:** share of the model's repeated answers (10 per comment) that are TOXIC
  (`ethos.scores.toxicity_score`); defined like the human score.
- **Distance from humans:** `|LLM score - human score|`.

## Pipeline

| Step | Script | Config | Output (`data/`) |
|---|---|---|---|
| 1. Build train/test datasets | `scripts/data/extract_splits.py` | `configs/splits.yaml` | `processed/toplevel_min30/` |
| 2. Draw consensus-balanced samples | `scripts/data/extract_consensus_sample.py` | `configs/sample_consensus_{test,train}.yaml` | `processed/consensus_n125_{test,train}/` |
| 3. Judge | `scripts/shell/run_judge.sh` (`scripts/judge.py`) | `configs/judges.yaml` | `judgements/<sample>/judgements.csv` |
| 4. Compare judges to humans | `scripts/plots/eval_consensus_gap.py` | | `judgements/<sample>/eval/` |
| Dataset figures | `scripts/plots/plot_consensus.py`, `plot_attributes.py` | | `processed/<dataset>/figures/` |

- **Samples:** 125 top-level, deduplicated comments with ≥ 30 annotators, 25 per consensus bin (0.5–1.0 in steps
  of 0.1). The **test** sample is for evaluation (Detoxify was trained on Jigsaw's train split); the **train**
  sample is for experiments and tuning.
- **Judges** (`configs/judges.yaml`): `human`, `detoxify`, `gpt-6-luna-base` (minimal TOXIC / NOT TOXIC prompt,
  10 answers) and `gpt-6-luna-dmp-random` (the same with a random 10-value profile per answer).

## Experiments

`scripts/experiments/` holds standalone experiments on the train sample; results go to `data/experiments/`
(see its `README.md`). Plot a run with `scripts/plots/plot_profiles.py --results <run folder>`.

| Script | Question |
|---|---|
| `profile_probe.py` | On a few comments, does a random or fixed DMP profile change the answers? |
| `profile_signal.py` | Do the identity and weighting of the profiles in `prompts/profiles/` change the answers, and does each taxonomy value on its own (`--single-values`)? |

## Setup and running

```bash
./setup.sh        # uv sync, .env from .env.example, Ollama models
```

Needs [uv](https://docs.astral.sh/uv/); Ollama only for local models. Add `OPENAI_API_KEY` to `.env` for the
OpenAI judges. Raw data: Kaggle *Jigsaw Unintended Bias in Toxicity Classification* in `data/unprocessed/`
(`data/` is git-ignored).

```bash
uv run python scripts/data/extract_splits.py
uv run python scripts/data/extract_consensus_sample.py --config configs/sample_consensus_test.yaml
scripts/shell/run_judge.sh                       # --only <judge>, --limit 3 for a quick test
uv run python scripts/plots/eval_consensus_gap.py
uv run python scripts/experiments/profile_signal.py --all --single-values --out data/experiments/03_single_values
uv run python scripts/plots/plot_profiles.py --results data/experiments/03_single_values
```

Re-running a judge or experiment skips answers already saved. `scripts/shell/slurm_ollama.sh` runs any command on a
FASRC GPU node with its own Ollama server.

## Layout

```
configs/     splits, sample and judge configs
prompts/     judge prompts, 60-value taxonomy, profiles/ (hand-written moral profiles)
scripts/     data/, judge.py, plots/, experiments/, shell/
src/ethos/   paths.py, data.py, scores.py, judges/, llm/ (OpenAI and Ollama clients, DMP profiles), plotting/
notebooks/   exploratory analysis
docs/        Civil Comments annotator instructions
.archive/    retired code, configs and prompts (variations, Kaleido, older judges, greedy weight search)
```

**Note:** if `No module named 'ethos'` appears, iCloud has hidden the venv's `.pth` files:
`chflags nohidden .venv/lib/python3.14/site-packages/*.pth` (the shell scripts handle this).
