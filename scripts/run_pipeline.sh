#!/usr/bin/env bash
# Extract the sample, then generate context variations for it.
#
# Usage:
#   scripts/run_pipeline.sh                # full run with configs/sample.yaml + configs/generation.yaml
#   scripts/run_pipeline.sh --limit 5      # extra args are passed to generate_variations.py
#
# The generation config's `input` must point at the sample config's `output_dir`/sample.csv.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=.venv/bin/python

echo "== Extracting sample"
"$PYTHON" scripts/extract_sample.py --config configs/sample.yaml

echo "== Generating variations"
"$PYTHON" scripts/generate_variations.py --config configs/generation.yaml "$@"
