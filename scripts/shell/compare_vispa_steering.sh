#!/usr/bin/env bash
# Regression test for our PCA steering (src/ethos/steering.py), on one gpu_test slice (~5 min). Run after changing it.
# Refits the directions for 3 VISPA values on Qwen3-8B and steers 3 prompts; fails unless the directions, steered top
# tokens and top logits match the reference in data/experiments/civil-comments/05_activation_steering/vispa_equivalence/
# (ours_tf4.57.6.npz). That reference was checked against VISPA's own code (e038eff + RepE 5455d8a): same directions
# (cosine 1.000000) and same steered outputs; see compare_vispa_tf4.57.6.json.
#
# To redo the VISPA comparison itself, point VISPA at a checkout of e038eff with RepE in third_party/ and a venv with
# transformers 4.57.6 and scikit-learn at $VISPA/../venv (see compare_vispa_steering.py).
#
# Usage (from anywhere): scripts/shell/compare_vispa_steering.sh        (log in <out>/logs/)
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
vispa=${VISPA:-}
out=data/experiments/civil-comments/05_activation_steering/vispa_equivalence
mkdir -p "$out/logs"

sbatch -p "${PARTITION:-gpu_test}" --account="${ACCOUNT:-protopapas_lab}" --gres=gpu:1 -c 4 --mem=64G -t "${TIME:-01:00:00}" \
  -J steering-test -o "$out/logs/steering-test-%j.out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$root'
export HF_HOME=/n/netscratch/protopapas_lab/Lab/\$USER/huggingface PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
nvidia-smi -L
if [ -n '$vispa' ]; then
  echo "=== ours vs VISPA (transformers 4.57); rewrites the reference ==="
  '$vispa/../venv/bin/python' scripts/experiments/compare_vispa_steering.py --vispa-root '$vispa' --out '$out'
fi
echo "=== regression test: ours vs the reference ==="
PYTHONPATH='$root/src' .venv/bin/python scripts/experiments/compare_vispa_steering.py --out '$out'
EOF
echo "log: $out/logs/steering-test-<job id>.out"
