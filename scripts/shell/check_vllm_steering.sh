#!/usr/bin/env bash
# Check the vLLM steering plugin (plugins/vllm_steer) against our HF steering on Qwen3-8B, on one gpu_test slice:
# vLLM-captured states must give the same directions, and vLLM-steered next tokens must match HF's. Prints PASS / FAIL
# per value; the job fails on FAIL. Run after changing the plugin. See scripts/experiments/check_vllm_steering.py.
#
# Usage (from anywhere): scripts/shell/check_vllm_steering.sh
# Results: data/experiments/civil-comments/05_activation_steering/vispa_equivalence/vllm/ (log in ../logs/)
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
out=data/experiments/civil-comments/05_activation_steering/vispa_equivalence
mkdir -p "$out/logs"

sbatch -p "${PARTITION:-gpu_test}" --account="${ACCOUNT:-protopapas_lab}" --gres=gpu:1 -c 8 --mem=64G -t "${TIME:-01:00:00}" \
  -J vllm-steer-check -o "$out/logs/vllm-steer-check-%j.out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$root'
scratch=/n/netscratch/protopapas_lab/Lab/\$USER
export UV_CACHE_DIR=\$scratch/uv-cache HF_HOME=\$scratch/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0   # in-process worker: plugin RPCs need no pickling
nvidia-smi -L
echo "=== vLLM: capture states and steer (plugin) ==="
PYTHONPATH='$root/src' uv run --no-project --managed-python --python 3.12 --with vllm==0.33.0 --with-editable ./plugins/vllm_steer \
  python scripts/experiments/check_vllm_steering.py --run
echo "=== compare with HF steering ==="
PYTHONPATH='$root/src' .venv/bin/python scripts/experiments/check_vllm_steering.py --compare
EOF
echo "log: $out/logs/vllm-steer-check-<job id>.out"
