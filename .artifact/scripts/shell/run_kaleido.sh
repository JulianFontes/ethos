#!/usr/bin/env bash
# Add Kaleido values (configs/kaleido.yaml) for every judge rationale to judgements.csv.
#
# Usage:
#   scripts/shell/run_kaleido.sh                                       # every rationale
#   scripts/shell/run_kaleido.sh --limit 5
set -euo pipefail

cd "$(dirname "$0")/../.."   # project root (this script lives in scripts/shell/)
# Find the ethos package directly; iCloud can hide the .venv's .pth files, which breaks `import ethos`.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# HF_TOKEN (Kaleido is a gated model) from .env, if present.
if [[ -f .env ]]; then set -a; source .env; set +a; fi

echo "Running Kaleido ..."
uv run python scripts/kaleido_values.py "$@"
