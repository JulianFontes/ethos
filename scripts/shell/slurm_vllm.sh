#!/usr/bin/env bash
# Run any project command on a full A100 with a private vLLM server started inside the job. The command reaches it
# with --model vllm:<model> (VLLM_BASE_URL is set below).
#
# vLLM runs in its own isolated environment (uvx), so the project's dependencies are untouched and nothing is installed
# on a Mac. The first run downloads vLLM and the model weights (cached in UV_CACHE_DIR / HF_HOME below).
#
# Usage (from the project root):
#   sbatch scripts/shell/slurm_vllm.sh uv run python scripts/experiments/profile_signal.py \
#       --model vllm:Qwen/Qwen3-8B --single-values --ids 240450 1020342 --samples 5 --workers 64 --data /tmp/vllm_test
#   VLLM_MODEL=Qwen/Qwen3-14B sbatch scripts/shell/slurm_vllm.sh ...      # another model (pass the same --model vllm:...)
#   sbatch --time=12:00:00 scripts/shell/slurm_vllm.sh ...                  # longer runs
#   scripts/shell/submit_vllm.sh --job-size 500 --sample <sample.csv> -- <command>   # split across nodes (array job)
#
#SBATCH --job-name=ethos-vllm
#SBATCH --partition=gpu
#SBATCH --account=protopapas_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs/slurm

MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-$((8000 + ${SLURM_ARRAY_TASK_ID:-0}))}"   # one port per array task, in case two share a node
# Big downloads (vLLM ~ several GB, model weights ~16 GB): keep them out of $HOME. Override if your lab uses another path.
SCRATCH="${SCRATCH_DIR:-/n/netscratch/protopapas_lab/$USER}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv-cache}"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export PYTHONUNBUFFERED=1            # show print() progress in the .out file as it happens
# Find the ethos package directly; iCloud can hide the .venv's .pth files, which breaks `import ethos`.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

nvidia-smi --query-gpu=name,memory.total --format=csv

# Our prompts are short (~300 tokens) and answers small, so a 4k context leaves the GPU for batching many requests.
uvx --python 3.12 --from vllm vllm serve "$MODEL" --port "$PORT" --max-model-len 4096 \
    --gpu-memory-utilization 0.90 > "logs/slurm/vllm-${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "Starting vLLM ($MODEL); the first run downloads it, which can take several minutes ..."
for _ in $(seq 180); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited, see logs/slurm/vllm-${SLURM_JOB_ID:-local}.log"; exit 1; }
    sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "vLLM did not start within 30 min"; exit 1; }
echo "vLLM ready. Running: $*"
"$@"
