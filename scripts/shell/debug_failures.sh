#!/usr/bin/env bash
# Find out why calls in a vLLM run fail: start the run's vLLM server on a gpu_test slice (same settings as
# scripts/experiments/submit_vllm.py) and run scripts/experiments/debug_failures.py against it.
# Results and logs go to <data>/debug/.
#
# Usage (from anywhere):
#   scripts/shell/debug_failures.sh data/processed/civil-comments/consensus_n1000_train/single_values_gpt_oss_20b openai/gpt-oss-20b low
# Extra arguments after the first three go to debug_failures.py (e.g. --repeats 3). MAX_MODEL_LEN=8192 to try a longer
# context than the run's 4096.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
data=$1 model=$2 effort=$3; shift 3
out=$data/debug
mkdir -p "$out/logs"

sbatch -p "${PARTITION:-gpu_test}" --account=protopapas_lab --gres=gpu:1 -c 8 --mem=64G -t "${TIME:-01:00:00}" \
  -J debug-failures -o "$out/logs/debug-%j.out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$root'
scratch=/n/netscratch/protopapas_lab/Lab/\$USER
export UV_CACHE_DIR=\$scratch/uv-cache HF_HOME=\$scratch/huggingface PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
port=\$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export VLLM_BASE_URL=http://127.0.0.1:\$port/v1
uvx --managed-python --python 3.12 --from vllm==0.33.0 vllm serve '$model' --port \$port \
  --max-model-len ${MAX_MODEL_LEN:-4096} --gpu-memory-utilization 0.90 --max-num-seqs 256 \
  > '$out/logs/vllm-'\$SLURM_JOB_ID.log 2>&1 &
server=\$!
trap 'kill \$server 2>/dev/null || true' EXIT
for i in \$(seq 180); do
  curl -sf http://127.0.0.1:\$port/health >/dev/null && break
  kill -0 \$server 2>/dev/null || { echo "vLLM exited, see $out/logs/vllm-\$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
echo "vLLM ready"
PYTHONPATH='$root/src:$root/scripts/experiments' .venv/bin/python scripts/experiments/debug_failures.py \
  --data '$data' --model '$model' --reasoning-effort '$effort' --out '$out' $*
EOF
echo "results and logs: $out/"
