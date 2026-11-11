#!/usr/bin/env bash
# Split a profile_signal.py run across nodes: one Slurm array task (one A100 + its own vLLM server) per job of
# --job-size comments. Each task gets its job from SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT.
#
# Usage (from the project root; extra sbatch options go in SBATCH_ARGS):
#   SBATCH_ARGS="--time=08:00:00" scripts/shell/submit_vllm.sh --job-size 500 \
#       --sample data/processed/consensus_n1000_train/sample.csv -- \
#       uv run python scripts/experiments/profile_signal.py --model vllm:Qwen/Qwen3-8B --all --single-values --workers 64 \
#       --sample data/processed/consensus_n1000_train/sample.csv --data data/processed/consensus_n1000_train/single_values_qwen3_8b
set -euo pipefail
cd "$(dirname "$0")/../.."

JOB_SIZE="" SAMPLE=""
while [[ $# -gt 0 && "$1" != "--" ]]; do
    case "$1" in
        --job-size) JOB_SIZE="$2"; shift 2 ;;
        --sample) SAMPLE="$2"; shift 2 ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done
shift   # the --
[[ -n "$JOB_SIZE" && -n "$SAMPLE" && $# -gt 0 ]] || { echo "usage: $0 --job-size N --sample <csv> -- <command>"; exit 1; }

N_COMMENTS=$(PYTHONPATH=src uv run python -c "import pandas as pd; print(len(pd.read_csv('$SAMPLE')))")
N_JOBS=$(( (N_COMMENTS + JOB_SIZE - 1) / JOB_SIZE ))
echo "$N_COMMENTS comments / $JOB_SIZE per job = $N_JOBS jobs (array 0-$((N_JOBS - 1)))"
sbatch --array=0-$((N_JOBS - 1)) ${SBATCH_ARGS:-} scripts/shell/slurm_vllm.sh "$@"
echo "When the jobs finish (or any time, to see progress), combine them with the same command plus --merge."
