#!/usr/bin/env bash
# Check that switching steering in a running vLLM server (no restart) matches starting a server with that steering,
# on gpt-oss-20b, on one gpu_test slice (~25 min: 4 server starts). Run after changing the plugin or the swap code.
# See scripts/experiments/check_steering_swap.py. Results: data/experiments/05_activation_steering/swap_check/
#
# Usage (from anywhere): scripts/shell/check_steering_swap.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
out=data/experiments/05_activation_steering/swap_check
mkdir -p "$out/logs"

sbatch -p "${PARTITION:-gpu_test}" --account="${ACCOUNT:-protopapas_lab}" --gres=gpu:1 -c 8 --mem=64G -t "${TIME:-01:30:00}" \
  -J steering-swap-check -o "$out/logs/swap-check-%j.out" \
  --wrap "cd '$root' && PYTHONPATH='$root/src' .venv/bin/python scripts/experiments/check_steering_swap.py --out '$out'"
echo "log: $out/logs/swap-check-<job id>.out"
