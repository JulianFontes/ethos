#!/usr/bin/env bash
# Run the evaluations on the judgements (configs/judges.yaml picks the sample):
# human alignment, then context sensitivity. Figures and CSVs go to data/judgements/<sample>/eval/.
#
# Usage:
#   scripts/shell/run_eval.sh                                          # sample_dir from configs/judges.yaml
#   scripts/shell/run_eval.sh --sample-dir data/processed/sample_n100  # extra args are passed to both scripts
set -euo pipefail

cd "$(dirname "$0")/../.."   # project root (this script lives in scripts/shell/)
# Find the ethos package directly; iCloud can hide the .venv's .pth files, which breaks `import ethos`.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=.venv/bin/python

echo "Evaluating human alignment ..."
"$PYTHON" scripts/eval_alignment.py "$@"

echo "Evaluating context sensitivity ..."
"$PYTHON" scripts/eval_context.py "$@"
