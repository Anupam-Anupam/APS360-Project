#!/bin/bash
#SBATCH --job-name=aps360-qwen3-1p7b-grpo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Driver job for APS360 GRPO on Qwen3-1.7B-Base.
# Training updates run on Tinker; this node hosts the verifier (vLLM).
# Set --account / partition for your cluster before submitting.

set -euo pipefail

echo "=== APS360 Qwen3-1.7B Math GRPO ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
date

if [ -f "$HOME/.tinker_env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.tinker_env"
fi
if [ -z "${TINKER_API_KEY:-}" ]; then
  echo "ERROR: TINKER_API_KEY not set" >&2
  exit 1
fi

# Activate your training venv (edit path as needed)
if [ -f "$HOME/envs/aps360/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/envs/aps360/bin/activate"
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-./log_qwen3_1p7b_math_grpo}"
mkdir -p "$LOG_DIR"

python scripts/train_grpo_qwen.py \
  log_path="$LOG_DIR" \
  model_name="Qwen/Qwen3-1.7B-Base" \
  math_only=True

echo "=== Done ==="
date
