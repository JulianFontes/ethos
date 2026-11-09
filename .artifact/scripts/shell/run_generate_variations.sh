#!/usr/bin/env bash
# Extract the sample, then generate context variations for it.
#
# Usage:
#   scripts/shell/run_generate_variations.sh                # full run with configs/generation.yaml
#   scripts/shell/run_generate_variations.sh --limit 5      # extra args are passed to generate_variations.py
#
# The generation config's `input` must point at the sample config's `output_dir`/sample.csv.
set -euo pipefail

cd "$(dirname "$0")/../.."   # project root (this script lives in scripts/shell/)
# Find the ethos package directly; iCloud can hide the .venv's .pth files, which breaks `import ethos`.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=.venv/bin/python

echo "Generating variations ..."
"$PYTHON" scripts/generate_variations.py --config configs/generation.yaml "$@"
