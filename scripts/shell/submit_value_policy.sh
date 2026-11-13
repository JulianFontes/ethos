#!/usr/bin/env bash
# Run scripts/experiments/value_policy.py as a CPU job on the cluster, with its log in <out>/logs/.
# Every argument is passed on to value_policy.py; --out decides where results and the log go.
#
# Usage (from anywhere):
#   scripts/shell/submit_value_policy.sh \
#     --scores data/processed/civil-comments/consensus_n1000_train/single_values_qwen3_8b_fp8/checkpoints/n0500/scores.csv \
#     --out data/experiments/civil-comments/04_value_policy/decision_tree_qwen3_8b_fp8_n500
# Slurm settings can be changed with environment variables, e.g. PARTITION=shared TIME=04:00:00.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD

out=data/experiments/civil-comments/04_value_policy/decision_tree   # value_policy.py's default
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  [[ ${args[i]} == --out ]] && out=${args[i + 1]}
done
mkdir -p "$out/logs"   # Slurm needs the log folder to exist before the job starts

sbatch -p "${PARTITION:-test}" --account="${ACCOUNT:-protopapas_lab}" -c 2 --mem="${MEM:-32G}" -t "${TIME:-02:00:00}" \
  -J value-policy -o "$out/logs/value-policy-%j.out" \
  --wrap "cd '$root' && PYTHONUNBUFFERED=1 PYTHONPATH='$root/src' .venv/bin/python scripts/experiments/value_policy.py $(printf '%q ' "${args[@]}")"
echo "log: $out/logs/value-policy-<job id>.out"
