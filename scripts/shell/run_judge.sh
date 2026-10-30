#!/usr/bin/env bash
# Run the judges (configs/judges.yaml) over every comment version of the sample.
#
# Usage:
#   scripts/shell/run_judge.sh                                         # all judges
#   scripts/shell/run_judge.sh --only qwen3:8b-dmp                     # extra args are passed to judge.py
#   scripts/shell/run_judge.sh --limit 2
set -euo pipefail

cd "$(dirname "$0")/../.."   # project root (this script lives in scripts/shell/)
# Find the ethos package directly; iCloud can hide the .venv's .pth files, which breaks `import ethos`.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Running judges ..."
uv run python scripts/judge.py "$@"
