#!/bin/bash
# Run this ON A COMPUTE NODE (interactive or batch) with the same modules as your
# training job, so CUDA_HOME is set and vLLM builds against the same CUDA/PyTorch.
# Usage:
#   srun --account=def-zhijing_gpu --gres=gpu:1 --mem=16G --time=1:00:00 --pty bash
#   cd $SCRATCH/MaCroScope && bash fix_vllm_match_torch.sh

set -e

# Same modules as run_test_training_fir.slurm (needed for CUDA_HOME and compiler)
module load python/3.11 cuda/12.6 gcc opencv arrow/22.0.0 cudnn nccl scipy-stack

# CUDA_HOME required for building vLLM's C++/CUDA extensions
export CUDA_HOME="${EBROOTCUDA:-/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.6.2}"
if [[ ! -d "$CUDA_HOME" ]]; then
  echo "ERROR: CUDA_HOME not found at $CUDA_HOME. Load the cuda module and ensure EBROOTCUDA is set."
  exit 1
fi
echo "Using CUDA_HOME=$CUDA_HOME"

source ~/envs/verl_train/bin/activate

# Build deps (vLLM build needs numpy; runtime needs psutil)
pip install --quiet "numpy" "psutil" "packaging" "setuptools" "wheel"

# Build vLLM from source so it links against current torch (2.5.1). 0.8.3 matches README; if it rejects torch 2.5.1, we fall back to latest.
echo "Building and installing vLLM from source (this will take several minutes)..."
if ! pip install vllm==0.8.3 --no-cache-dir --no-build-isolation; then
  echo "vllm==0.8.3 failed (e.g. torch version); trying latest vllm from source..."
  pip install vllm --no-cache-dir --no-build-isolation
fi

echo ""
echo "Verifying torch and vllm..."
python -c "import torch; print('torch:', torch.__version__)"
python -c "from vllm import LLM; import vllm; print('vllm:', vllm.__version__)"
echo "Done. Re-run your test training job."
