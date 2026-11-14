#!/usr/bin/env bash
# Train the linear softmax steering-strength policy (src/ethos/alpha_policy.py) on one activation steering run and
# plot its training / validation loss against constant alphas. Runs locally in seconds (no cluster job).
#
# Usage (from anywhere):
#   scripts/shell/run_alpha_policy.sh                       # the 500-comment Llama run
#   RUN=data/processed/civil-comments/boundary_n100_train/steer_toxicity_llama3_8b_l10-25 \
#     OUT=data/experiments/civil-comments/05_activation_steering/alpha_policy_llama3_8b_boundary_n100 \
#     scripts/shell/run_alpha_policy.sh --features prior  # other runs; extra arguments go to the policy
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN=${RUN:-data/processed/civil-comments/boundary_n500_train/steer_toxicity_llama3_8b_l10-25_a0.3}
OUT=${OUT:-data/experiments/civil-comments/05_activation_steering/alpha_policy_llama3_8b_boundary_n500}

PYTHONPATH=src .venv/bin/python -m ethos.alpha_policy --scores "$RUN/scores.csv" --out "$OUT" "$@"
