#!/bin/bash
#SBATCH --job-name=macroscope-1gpu
#SBATCH --account=def-zhijing_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train-1gpu-%j.out
#SBATCH --error=logs/train-1gpu-%j.err

# ── Single-GPU production training job for Fir (H100 80GB) ───────────────────
# Submit: sbatch run_training_1gpu_fir.slurm
# Override epochs: sbatch --export=ALL,TOTAL_EPOCHS=5 run_training_1gpu_fir.slurm
# Override model:  sbatch --export=ALL,MODEL_OVERRIDE=Qwen2.5-7B run_training_1gpu_fir.slurm
#
# Memory budget (H100 80GB, Qwen2.5-7B):
#   weights bf16  : ~14 GB  (held by both FSDP and vLLM, alternating)
#   FSDP grads    : ~14 GB
#   optimizer     :  0 GB GPU (offloaded to CPU, ~28 GB RAM — hence mem=120G)
#   vLLM KV cache : ~32 GB  (0.4 × 80 GB)
#   headroom      :  ~8 GB
#   CPU RAM       : 80 GB used by slurm + ~28 GB Adam states ≈ 108 GB → 120G safe

set -e

echo "=== Job Info ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "GPUs     : $SLURM_GPUS_ON_NODE"
date
echo ""

# ── Modules ──────────────────────────────────────────────────────────────────
# Load in one command so Lmod can resolve conflicts cleanly.
module load python/3.11 cuda/12.6 gcc opencv arrow/22.0.0 cudnn nccl scipy-stack

echo "CUDA    : $EBROOTCUDA"
echo "cuDNN   : $EBROOTCUDNN"
echo "NCCL    : $EBROOTNCCL"

# ── Virtual environment ───────────────────────────────────────────────────────
# Prefer SCRATCH (Lustre, avoids NFS disconnects) over HOME.
VENV_ACTIVATE="$SCRATCH/MaCroScope/envs/verl_train/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
  source "$VENV_ACTIVATE"
else
  echo "ERROR: venv not found at $VENV_ACTIVATE"
  echo "Run install_fir.sh first (on a login node or interactive node)."
  exit 1
fi

echo "Python  : $(python --version)"
echo "venv    : $VIRTUAL_ENV"

# ── LD_LIBRARY_PATH ───────────────────────────────────────────────────────────
# Build from EBROOT* variables — do NOT append the existing value so that
# CUDA 12.9.x libraries from the system cannot sneak in.
export LD_LIBRARY_PATH=\
${EBROOTCUDNN}/lib64:\
${EBROOTCUDA}/lib64:\
${EBROOTCUDA}/extras/CUPTI/lib64:\
${EBROOTNCCL}/lib:\
${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cusparselt/lib

# ── GPU / Ray settings ────────────────────────────────────────────────────────
# Let SLURM control which GPU is visible; prevent Ray from overriding it.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ── NCCL settings ────────────────────────────────────────────────────────────
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800000

# ── Weights & Biases ──────────────────────────────────────────────────────────
# Set WANDB_API_KEY before submitting, or disable logging:
#   export WANDB_MODE=disabled
# W&B is available on Fir login nodes; compute nodes may be firewalled.
# If W&B upload fails, set WANDB_MODE=offline and sync later:
#   wandb sync $SCRATCH/MaCroScope/logs/wandb/run-*
: "${WANDB_API_KEY:=}"          # leave empty to use whatever is already exported
export WANDB_CACHE_DIR="$SCRATCH/MaCroScope/.wandb_cache"
export WANDB_DIR="$SCRATCH/MaCroScope/logs"

# ── Working directory ─────────────────────────────────────────────────────────
cd "$SCRATCH/MaCroScope"
mkdir -p logs

export WORKING_DIR=$(pwd)
export HEAD_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

echo ""
echo "Working Dir : $WORKING_DIR"
echo "Head IP     : $HEAD_IP"
echo ""

# ── Start Ray (local single-node cluster) ────────────────────────────────────
ray start \
  --head \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --num-gpus=1 \
  --num-cpus=$SLURM_CPUS_PER_TASK \
  --block &

RAY_PID=$!
sleep 8   # give Ray time to initialise before job submission

# ── Launch training ───────────────────────────────────────────────────────────
bash train_general_reasoner_1gpu.sh \
  ${TOTAL_EPOCHS:+--total_epochs $TOTAL_EPOCHS} \
  ${MODEL_OVERRIDE:+--model_name $MODEL_OVERRIDE}

# ── Shutdown ──────────────────────────────────────────────────────────────────
ray stop || true
wait $RAY_PID 2>/dev/null || true

echo ""
echo "=== Job Complete ==="
date
