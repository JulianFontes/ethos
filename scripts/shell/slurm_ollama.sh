#!/usr/bin/env bash
# Run any project command on a GPU node with a private Ollama server (qwen3:8b) started inside the job.
# The code already talks to http://localhost:11434, so nothing else changes.
#
# Usage (from the project root):
#   sbatch scripts/shell/slurm_ollama.sh scripts/shell/run_judge.sh --only <judge name>
#   sbatch scripts/shell/slurm_ollama.sh uv run python scripts/experiments/profile_signal.py --model qwen3:8b --workers 4 --out data/experiments/<run>
#   sbatch -p gpu --time=24:00:00 scripts/shell/slurm_ollama.sh ...      # longer runs (gpu_test caps at 12h)
#
#SBATCH --job-name=ethos-ollama
#SBATCH --partition=gpu_test
#SBATCH --account=protopapas_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs/slurm

module load ncf/1.0.0-fasrc01 ollama/v0.15.6-linux_x64-ncf

# The ncf module sets OLLAMA_MODELS to a shared read-only store (/n/sw/ncf/models/ollama). To pull models it
# lacks, submit with OLLAMA_MODELS=$HOME/.ollama/models (or a netscratch path) in the environment.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"   # concurrent requests (match --workers)
export OLLAMA_HOST=127.0.0.1:11434   # the configs point at this port
export PYTHONUNBUFFERED=1            # show print() progress in the .out file as it happens
# Ollama sizes its default context from the whole A100 (40 GB), but gpu_test gives a 20 GB MIG slice; with
# 32k context x NUM_PARALLEL slots the model spills onto the CPU. Our prompts are short, so 4k is plenty.
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}"

ollama serve > "logs/slurm/ollama-${SLURM_JOB_ID:-local}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

for _ in $(seq 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done
ollama list >/dev/null || { echo "Ollama did not start (port 11434 busy?), see logs/slurm/"; exit 1; }
# The module's shared (read-only) model store already has qwen3:8b; only pull when it's missing.
ollama list | grep -q '^qwen3:8b' || ollama pull qwen3:8b

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Running: $*"
"$@"
