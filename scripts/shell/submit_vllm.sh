#!/usr/bin/env bash
# Submit a distributed vLLM run from a cluster login node; all settings are in the YAML config.
#
# Usage (from anywhere):
#   scripts/shell/submit_vllm.sh --config configs/vllm_run.yaml --dry-run    # show the plan
#   scripts/shell/submit_vllm.sh --config configs/vllm_run.yaml
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"   # iCloud can hide the venv's .pth files on a Mac
uv run python scripts/experiments/submit_vllm.py "$@"
