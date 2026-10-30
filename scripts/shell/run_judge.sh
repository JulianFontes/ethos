#!/usr/bin/env bash
# Run the judges (configs/judges.yaml) over every comment version of the sample.
#
# Usage:
#   scripts/shell/run_judge.sh                                         # all judges
#   scripts/shell/run_judge.sh --only qwen3:8b-dmp                     # extra args are passed to judge.py
#   scripts/shell/run_judge.sh --limit 2
set -euo pipefail

cd "$(dirname "$0")/../.."   # project root (this script lives in scripts/shell/)
PYTHON=.venv/bin/python

echo "Running judges ..."
"$PYTHON" scripts/judge.py "$@"
