#!/usr/bin/env bash
# Capture the judge's internal state for comments, unsteered and steered (scripts/experiments/capture_steered_states.py),
# on one gpu_test slice. Results and log go to --out.
#
# Usage (from anywhere):
#   scripts/shell/capture_steered_states.sh --out data/experiments/05_activation_steering/steered_states_toxicity [options]
# Options are passed on to capture_steered_states.py (e.g. --n 100 --capture 17 --layers 7-19 --alpha 68.6).
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
args=("$@") out= layers=7-19
for ((i = 0; i < ${#args[@]}; i++)); do
  [[ ${args[i]} == --out ]] && out=${args[i + 1]}
  [[ ${args[i]} == --layers ]] && layers=${args[i + 1]}
done
[[ -n $out ]] || { echo "need --out" >&2; exit 1; }
mkdir -p "$out/logs"
steer_layers=$(.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from ethos.steering.pairs import parse_layers; print(','.join(map(str, parse_layers('$layers'))))")

sbatch -p "${PARTITION:-gpu_test}" --account="${ACCOUNT:-protopapas_lab}" --gres=gpu:1 -c 8 --mem=64G -t "${TIME:-01:00:00}" \
  -J capture-steered -o "$out/logs/capture-%j.out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$root'
scratch=/n/netscratch/protopapas_lab/Lab/\$USER
export UV_CACHE_DIR=\$scratch/uv-cache HF_HOME=\$scratch/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0   # in-process worker: plugin RPCs need no pickling
export ETHOS_STEER_LAYERS=$steer_layers                                    # zero steering vectors on these layers
PYTHONPATH='$root/src' uv run --no-project --managed-python --python 3.12 --with vllm==0.33.0 \
  --with-editable ./plugins/vllm_steer python scripts/experiments/capture_steered_states.py $(printf '%q ' "${args[@]}")
EOF
echo "results and log: $out/"
