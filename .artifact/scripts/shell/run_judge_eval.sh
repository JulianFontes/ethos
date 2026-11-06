#!/usr/bin/env bash
# Run the judges, then the evaluations on their judgements.
#
# Usage:
#   scripts/shell/run_judge_eval.sh                                   # everything, sample_dir from configs/judges.yaml
#   scripts/shell/run_judge_eval.sh --only qwen3:8b-dmp --limit 3     # extra args are passed to judge.py
#
# --sample-dir is also passed to the evaluation; other args only go to the judges.
set -euo pipefail

cd "$(dirname "$0")"   # this script lives in scripts/shell/ next to the two it runs

eval_args=()
args=("$@")
for i in "${!args[@]}"; do
  if [[ "${args[$i]}" == "--sample-dir" ]]; then
    eval_args=(--sample-dir "${args[$((i + 1))]}")
  fi
done

./run_judge.sh "$@"
./run_eval.sh ${eval_args[@]+"${eval_args[@]}"}
