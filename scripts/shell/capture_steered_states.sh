#!/usr/bin/env bash
# Capture the judge's internal state for comments, unsteered and steered (scripts/experiments/capture_steered_states.py),
# on one gpu_test slice, then plot the shift by layer (scripts/plots/plot_steered_states.py). --out is the run's data
# folder: states and log go to <out>/hidden_states/, the plot to <out>/figures/shift_by_layer.png.
#
# Usage (from anywhere):
#   scripts/shell/capture_steered_states.sh --out data/processed/boundary_n100_train/steer_toxicity_gpt_oss_20b_l7-19 [options]
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
mkdir -p "$out/hidden_states/logs"
steer_layers=$(.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from ethos.steering.pairs import parse_layers; print(','.join(map(str, parse_layers('$layers'))))")

sbatch -p "${PARTITION:-gpu_test}" --account="${ACCOUNT:-protopapas_lab}" --gres=gpu:1 -c 8 --mem=64G -t "${TIME:-01:00:00}" \
  -J capture-steered -o "$out/hidden_states/logs/capture-%j.out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$root'
scratch=/n/netscratch/protopapas_lab/Lab/\$USER
export UV_CACHE_DIR=\$scratch/uv-cache HF_HOME=\$scratch/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0   # in-process worker: plugin RPCs need no pickling
export ETHOS_STEER_LAYERS=$steer_layers                                    # zero steering vectors on these layers
PYTHONPATH='$root/src' uv run --no-project --managed-python --python 3.12 --with vllm==0.33.0 \
  --with-editable ./plugins/vllm_steer python scripts/experiments/capture_steered_states.py $(printf '%q ' "${args[@]}")
.venv/bin/python scripts/plots/plot_steered_states.py --data '$out'                 # <out>/figures/shift_by_layer.png
EOF
echo "states and log: $out/hidden_states/; plot: $out/figures/shift_by_layer.png"
