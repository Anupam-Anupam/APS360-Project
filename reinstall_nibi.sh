#!/bin/bash
# Reinstall MaCroScope environment on Nibi from scratch.
#
# Copy your project to Nibi first (from your laptop, in the project directory):
#   cd /path/to/MaCroScope
#   scp -r . anupam@nibi.alliancecan.ca:\$SCRATCH/MaCroScope
#   (Or from project dir: scp -r . anupam@nibi.alliancecan.ca:\$SCRATCH/MaCroScope)
#
# Run on Nibi (login or interactive compute node with GPU for flash-attn build):
#   cd $SCRATCH/MaCroScope   # or wherever you copied the project
#   bash reinstall_nibi.sh
#
# Optional: pass repo root if not running from it:
#   bash /path/to/reinstall_nibi.sh
#
# For flash-attn and vLLM builds, an interactive GPU node is recommended:
#   srun --account=def-zhijing_gpu --gres=gpu:1 --mem=32G --time=2:00:00 --pty bash
#   cd $SCRATCH/MaCroScope && bash reinstall_nibi.sh

set -e

REPO_ROOT="${1:-${SCRATCH:-$HOME/scratch}/MaCroScope}"
VENV_DIR="$HOME/envs/verl_train"

echo "=========================================="
echo "MaCroScope reinstall on Nibi"
echo "=========================================="
echo "Repo root: $REPO_ROOT"
echo "Venv:      $VENV_DIR"
echo ""

# Load modules (same as run_test_training_nibi.slurm)
echo "=== Loading modules ==="
module load python/3.11 cuda/12.6 gcc opencv arrow/22.0.0 cudnn nccl scipy-stack
echo "Python: $(which python3)"
echo ""

# Ensure we're in repo root and verl exists (must be copied with rsync)
echo "=== Repo and verl ==="
cd "$REPO_ROOT"
if [ ! -f main_ppo.py ] || [ ! -f train_general_reasoner.sh ]; then
  echo "ERROR: Repo root not found (no main_ppo.py). Copy project with scp, then pass path: bash reinstall_nibi.sh /path/to/MaCroScope"
  exit 1
fi
if [ ! -d verl ] || [ ! -f verl/setup.py ]; then
  echo "ERROR: verl directory missing or incomplete. From your laptop, copy the full project (including the verl folder) to Nibi with scp."
  exit 1
fi
echo ""

# Remove existing venv (backup optional)
if [ -d "$VENV_DIR" ]; then
  BACKUP="$HOME/envs/verl_train.bak.$(date +%Y%m%d_%H%M%S)"
  echo "=== Backing up existing venv to $BACKUP ==="
  mv "$VENV_DIR" "$BACKUP"
  echo ""
fi

# Create fresh venv
echo "=== Creating virtual environment ==="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel
echo ""

# 1. PyTorch (README: cu124; Nibi has cuda/12.6, cu124 wheels are compatible)
echo "=== Installing PyTorch 2.4.0 (CUDA 12.4) ==="
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
echo ""

# 2. Flash Attention (needs CUDA; build can be slow)
echo "=== Installing flash-attn (this may take several minutes) ==="
pip install flash-attn --no-build-isolation
echo ""

# 3. verl (editable from repo)
echo "=== Installing verl (editable) ==="
pip install -e ./verl
echo ""

# 4. vLLM
echo "=== Installing vllm==0.8.3 ==="
pip install vllm==0.8.3
echo ""

# 5. FlashInfer
echo "=== Installing flashinfer-python ==="
pip install flashinfer-python
echo ""

# 6. math-verify
echo "=== Installing math-verify ==="
pip install math-verify
echo ""

# Optional but useful for training
echo "=== Installing extra deps ==="
pip install numpy psutil packaging hydra-core omegaconf
echo ""

# Verify
echo "=========================================="
echo "Verification"
echo "=========================================="
source "$VENV_DIR/bin/activate"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import flash_attn; print('flash_attn: OK')"
python -c "import verl; print('verl: OK')"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import flashinfer; print('flashinfer: OK')"
python -c "from math_verify.metric import math_metric; print('math_verify: OK')"
echo ""
echo "=== Reinstall complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Then run: bash verify_setup.sh (from repo root) and/or submit your test job."
echo "Data/models: ensure $HOME/scratch/macroscope/models and data are present (see README)."
echo "  e.g. huggingface-cli download Qwen/Qwen3-4B-Base --local-dir \$HOME/scratch/macroscope/models/Qwen3-4B-Base"
echo "  e.g. huggingface-cli download TIGER-Lab/general-verifier --local-dir \$HOME/scratch/macroscope/models/general-verifier"
echo "  e.g. python data_preprocess.py --local-dir \$HOME/scratch/macroscope/data/webinstruct-verified"
