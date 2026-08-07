#!/bin/bash
#SBATCH --job-name=test-training
#SBATCH --account=def-zhijing_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err

set -e

echo "=== Test Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

# Load only CUDA 12.6 (avoid 12.9.x from polluting Ray workers)
module purge
module load python/3.11 cuda/12.6 gcc opencv arrow/22.0.0 cudnn nccl scipy-stack

# Activate venv from SCRATCH if present (avoids "Transport endpoint is not connected" on compute nodes when home is NFS).
# One-time: cp -a ~/envs/verl_train $SCRATCH/MaCroScope/envs/
VENV_ACTIVATE="$SCRATCH/MaCroScope/envs/verl_train/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
  source "$VENV_ACTIVATE"
else
  source ~/envs/verl_train/bin/activate
fi

# Set environment variables: CUDA 12.6 only (do not append existing LD_LIBRARY_PATH to avoid pulling in 12.9.x)
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
echo ""

# Start Ray head node
ray start --head --port=6379 --dashboard-host=0.0.0.0 --block &

# Wait a moment for Ray to start
sleep 5

# Run test training
bash test_training.sh

# Stop Ray
ray stop

echo ""
echo "=== Test Job Complete ==="
date
