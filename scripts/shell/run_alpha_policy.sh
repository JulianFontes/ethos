#!/usr/bin/env bash
# Train and evaluate the steering-strength policy (scripts/experiments/alpha_policy.py; library: src/ethos/alpha_policy.py,
# plots: src/ethos/plotting/alpha_policy.py). Every setting (run, output folder, model, inputs, folds, epochs) is in the
# config. Runs locally in about a minute.
#   scripts/shell/run_alpha_policy.sh                          # configs/llama3_8b/alpha_policy.yaml
#   CONFIG=configs/llama3_8b/my_policy.yaml scripts/shell/run_alpha_policy.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=src .venv/bin/python scripts/experiments/alpha_policy.py --config "${CONFIG:-configs/llama3_8b/alpha_policy.yaml}"
