#!/bin/bash
#SBATCH --job-name=aps360-qwen3-1p7b-grpo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Alliance / Fir GRPO driver for Qwen/Qwen3-1.7B (Table 1: n_nodes=1, n_gpu=4).
# Training updates run on Tinker; this node hosts the verifier (vLLM 0.25.1).
# Set --account / --partition for your allocation before submitting, e.g.:
#   #SBATCH --account=rrg-zhijing
#   #SBATCH --gpus-per-node=h100:4

set -euo pipefail

echo "=== APS360 Qwen3-1.7B Math GRPO ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
date

# Secrets: TINKER_API_KEY + WANDB_API_KEY (see .env.example)
if [ -f "$HOME/.tinker_env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.tinker_env"
fi
if [ -z "${TINKER_API_KEY:-}" ] || [ "${TINKER_API_KEY}" = "REPLACE_ME" ]; then
  echo "ERROR: TINKER_API_KEY not set. Put it in ~/.tinker_env (chmod 600)." >&2
  exit 1
fi
if [ -z "${WANDB_API_KEY:-}" ] || [ "${WANDB_API_KEY}" = "REPLACE_ME" ]; then
  echo "WARNING: WANDB_API_KEY not set — training will run without W&B." >&2
else
  echo "W&B: key present (project APS360-Qwen3-GRPO)"
fi

# Cluster modules (Fir / Alliance Canada style; no-op if module unavailable)
if command -v module >/dev/null 2>&1; then
  module load python/3.11 cuda/12.2 gcc opencv arrow 2>/dev/null || true
fi

# Activate training venv (edit path as needed)
if [ -f "$HOME/envs/aps360/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/envs/aps360/bin/activate"
elif [ -f "$HOME/envs/macroscope/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/envs/macroscope/bin/activate"
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-./log_qwen3_1p7b_math_grpo}"
mkdir -p "$LOG_DIR"

echo "Repo: $REPO_ROOT"
echo "Log dir: $LOG_DIR"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-all} (count via torch below)"
python - <<'PY' || true
import torch
print(f"PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, n_gpu={torch.cuda.device_count()}")
PY

python scripts/train_grpo_qwen3.py \
  log_path="$LOG_DIR" \
  model_name="Qwen/Qwen3-1.7B" \
  dataset_path="anupamc/math-dataset" \
  math_only=False \
  n_nodes=1 \
  n_gpu=4 \
  train_batch_size=128 \
  max_prompt_length=384 \
  max_response_length=128 \
  learning_rate=5e-7 \
  ppo_micro_batch_size_per_gpu=4 \
  ppo_mini_batch_size=64 \
  clip_ratio_low=0.3 \
  clip_ratio_high=0.3 \
  temperature=1.0 \
  rollout_n=8 \
  kl_coeff=0.01 \
  total_epochs=19 \
  wandb_project="APS360-Qwen3-GRPO" \
  wandb_name="qwen3-1.7b-math-grpo"

echo "=== Done ==="
date
