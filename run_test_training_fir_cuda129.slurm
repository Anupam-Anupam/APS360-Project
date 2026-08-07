#!/bin/bash
# Option B: Use CUDA 12.9 so the whole stack (torch 2.10, vllm 0.16) matches.
# Run in parallel with run_test_training_fir.slurm (Option A, 12.6) and compare logs.
#SBATCH --job-name=test-cuda129
#SBATCH --account=def-zhijing_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --output=logs/test-129-%j.out
#SBATCH --error=logs/test-129-%j.err

set -e

echo "=== Test Job Info (Option B: CUDA 12.9) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

# Load modules: CUDA 12.9 first (try 12.9.1 if 12.9 not available), then cudnn
module load python/3.11 gcc opencv arrow/22.0.0 nccl scipy-stack
if module load cuda/12.9 2>/dev/null; then
  echo "Loaded cuda/12.9"
elif module load cuda/12.9.1 2>/dev/null; then
  echo "Loaded cuda/12.9.1"
else
  echo "ERROR: No cuda/12.9 or cuda/12.9.1 module. Try: module avail cuda"
  exit 1
fi
module load cudnn

# Activate venv from SCRATCH if present (avoids "Transport endpoint is not connected" on compute nodes).
# One-time: cp -a ~/envs/verl_train $SCRATCH/MaCroScope/envs/
VENV_ACTIVATE="$SCRATCH/MaCroScope/envs/verl_train/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
  source "$VENV_ACTIVATE"
else
  source ~/envs/verl_train/bin/activate
fi

# CUDA 12.9 only (no append)
export LD_LIBRARY_PATH=$EBROOTCUDNN/lib64:$EBROOTCUDA/lib64:$EBROOTCUDA/extras/CUPTI/lib64:$EBROOTNCCL/lib:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cusparselt/lib

# Prevent Ray from overriding SLURM's GPU allocation
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# NCCL config
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800000

cd $SCRATCH/MaCroScope
mkdir -p logs

# Get head node IP
HEAD_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export HEAD_IP
export WORKING_DIR=$(pwd)

echo "Head IP: $HEAD_IP"
echo "Working Dir: $WORKING_DIR"
echo "Python: $(python --version)"
echo "CUDA: $EBROOTCUDA"
echo ""

# Start Ray head node
ray start --head --port=6379 --dashboard-host=0.0.0.0 --block &

# Wait a moment for Ray to start
sleep 5

# Run test training (test_training.sh will pass 12.9 paths via EBROOT*)
bash test_training.sh

# Stop Ray
ray stop

echo ""
echo "=== Test Complete (Option B) ==="
date
