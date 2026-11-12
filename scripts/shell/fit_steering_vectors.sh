#!/usr/bin/env bash
# Fit PCA steering vectors for a model on one gpu_test slice:
#   1. capture_states.py (vLLM + plugins/vllm_steer): last-token states of every VISPA pair sentence
#   2. fit_steering_vectors.py (ethos env): directions per value and layer, meta.json, steering_layers.png
# The captured states are temporary (netscratch) and deleted after step 2.
#
# Usage (from anywhere):
#   scripts/shell/fit_steering_vectors.sh --model openai/gpt-oss-20b --layers 7-19 [--values fairness justice ...]
# Vectors go to data/processed/steering_vectors/<model name>/ (log in its logs/). Pairs: VISPA's data/value/*.csv
# (set PAIRS=<dir>; default: the pluralistic-vispa checkout, whose data/ is unchanged since the trusted e038eff).
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
model= layers= values=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) model=$2; shift 2 ;;
    --layers) layers=$2; shift 2 ;;
    --values) shift; while [[ $# -gt 0 && $1 != --* ]]; do values+=("$1"); shift; done ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done
[[ -n $model && -n $layers ]] || { echo "need --model and --layers" >&2; exit 1; }
pairs=${PAIRS:-/n/home12/$USER/projects/pluralistic-vispa/data/value}
out=data/processed/steering_vectors/${model##*/}
mkdir -p "$out/logs"
value_args=${values[*]:+--values ${values[*]}}

sbatch -p "${PARTITION:-gpu_test}" --account="${ACCOUNT:-protopapas_lab}" --gres=gpu:1 -c 8 --mem=64G -t "${TIME:-02:00:00}" \
  -J fit-steering -o "$out/logs/fit-steering-%j.out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$root'
scratch=/n/netscratch/protopapas_lab/Lab/\$USER
states=\$scratch/steering_states/${model##*/}-\$SLURM_JOB_ID
export UV_CACHE_DIR=\$scratch/uv-cache HF_HOME=\$scratch/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0   # in-process worker: plugin RPCs need no pickling
nvidia-smi -L
echo "=== 1. capture states in vLLM ==="
PYTHONPATH='$root/src' uv run --no-project --managed-python --python 3.12 --with vllm==0.33.0 --with-editable ./plugins/vllm_steer \
  python scripts/experiments/capture_states.py --model '$model' --pairs '$pairs' --layers '$layers' --out "\$states" $value_args
echo "=== 2. fit directions ==="
PYTHONPATH='$root/src' .venv/bin/python scripts/experiments/fit_steering_vectors.py --states "\$states" --out '$out'
rm -rf "\$states"
EOF
echo "vectors: $out/   log: $out/logs/fit-steering-<job id>.out"
