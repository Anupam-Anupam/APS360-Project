#!/bin/bash
# Comprehensive MaCroScope environment install script for Fir (Alliance Canada)
#
# WHERE TO RUN:
#   Login node is fine for everything except flash-attn compilation.
#   flash-attn needs nvcc (available via module load cuda/12.6) but NOT a running GPU.
#   So the login node works for the full install.
#
# QUICK START:
#   # 1. Copy project to Fir (from laptop):
#   #    rsync -av --exclude='.git' /path/to/MaCroScope/ anupam@fir.alliancecan.ca:$SCRATCH/MaCroScope/
#   #
#   # 2. SSH into Fir login node and run:
#   #    cd $SCRATCH/MaCroScope
#   #    bash install_fir.sh
#   #
#   # 3. Download models and data (see bottom of this script).
#
# WHAT THIS INSTALLS:
#   torch 2.5.1+cu124    — H100-compatible, avoids triton_key bug present in 2.4.x
#   flash-attn 2.7.x     — compiled from source against CUDA 12.6 headers
#   vllm 0.8.5           — compatible with torch 2.5.1, no triton_key dependency
#   flashinfer           — required by vLLM for H100 kernels
#   verl (editable)      — the RL training framework (from ./verl)
#   ray[default]         — distributed actor framework
#   transformers / datasets / accelerate / deepspeed — HuggingFace + training stack
#   math-verify          — reward function dependency
#   wandb                — experiment tracking
#
# VERSION RATIONALE:
#   torch 2.5.1+cu124:  cu124 wheels are forward-compatible with CUDA 12.6 (12.4 ≤ 12.6).
#                       2.5.1 ships triton 3.0.0 which no longer has triton_key,
#                       eliminating the "cannot import name triton_key" crash seen with 2.4.0.
#   vllm 0.8.5:         Last 0.8.x release; supports torch>=2.4,<2.6; no triton_key usage.
#   flashinfer 0.2.x:   Provides optimised attention kernels vLLM uses on H100.

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT="${1:-${SCRATCH}/MaCroScope}"
VENV_DIR="$SCRATCH/MaCroScope/envs/verl_train"
PYTHON_VERSION="3.11"

echo "============================================================"
echo "MaCroScope install for Fir (H100 + CUDA 12.6)"
echo "============================================================"
echo "Repo root : $REPO_ROOT"
echo "Venv      : $VENV_DIR"
echo ""

# Verify we're on a Fir-family node
if [[ ! "$HOSTNAME" == *"fir"* ]] && [[ ! "$HOSTNAME" == *"login"* ]] && [[ ! "$HOSTNAME" == *"int"* ]]; then
  echo "WARNING: hostname is '$HOSTNAME' — does not look like a Fir login/interactive node."
  echo "Proceeding anyway. If module load fails, SSH into fir.alliancecan.ca first."
fi

# ── Check repo ────────────────────────────────────────────────────────────────
echo "=== Checking repo ==="
cd "$REPO_ROOT" || {
  echo "ERROR: Cannot find repo at $REPO_ROOT."
  echo "Copy the project first: rsync -av --exclude=.git /local/MaCroScope/ \$SCRATCH/MaCroScope/"
  exit 1
}
if [ ! -f main_ppo.py ] || [ ! -f train_general_reasoner.sh ]; then
  echo "ERROR: main_ppo.py or train_general_reasoner.sh not found."
  echo "Are you in the project root? (current dir: $(pwd))"
  exit 1
fi
if [ ! -d verl ] || [ ! -f verl/setup.py ]; then
  echo "ERROR: ./verl submodule missing or incomplete."
  echo "The verl/ directory must be copied alongside the rest of the project."
  exit 1
fi
echo "Repo looks good."
echo ""

# ── Load modules ──────────────────────────────────────────────────────────────
echo "=== Loading modules ==="
# Load all at once so Lmod can resolve inter-module conflicts cleanly.
module load python/${PYTHON_VERSION} cuda/12.6 gcc opencv arrow/22.0.0 cudnn nccl scipy-stack
echo "Python  : $(which python3)"
echo "nvcc    : $(nvcc --version 2>/dev/null | grep release || echo 'n/a')"
echo "CUDA    : $EBROOTCUDA"
echo ""

# ── Create virtual environment in SCRATCH ─────────────────────────────────────
# IMPORTANT: always create the venv in $SCRATCH, not $HOME.
# Slurm compute nodes can lose the NFS mount to $HOME mid-job; $SCRATCH (Lustre)
# stays connected. The slurm script also looks here:
#   $SCRATCH/MaCroScope/envs/verl_train/bin/activate
echo "=== Creating virtual environment ==="
if [ -d "$VENV_DIR" ]; then
  BACKUP="${VENV_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
  echo "Existing venv found — backing up to $BACKUP"
  mv "$VENV_DIR" "$BACKUP"
fi
mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "Venv activated: $VIRTUAL_ENV"

# Upgrade pip from the Alliance wheelhouse (--no-index = no network needed for this step).
pip install --no-index --upgrade pip setuptools wheel
echo ""

# ── 1. PyTorch 2.5.1 (CUDA 12.4 wheel — forward-compatible with CUDA 12.6) ───
echo "=== [1/8] PyTorch 2.5.1+cu124 ==="
# cu124 wheels work under CUDA 12.6 (CUDA is backward-compatible).
# We need 2.5.1 specifically:
#   - torch 2.5.x → triton 3.0.0 (triton_key removed → no crash with vllm 0.8.5)
#   - torch 2.5.x is the minimum recommended for H100 clusters (per Alliance docs)
pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

python -c "
import torch
print('  torch      :', torch.__version__)
print('  CUDA built :', torch.version.cuda)
print('  triton     :', __import__('triton').__version__)
"
echo ""

# ── 2. Flash Attention ────────────────────────────────────────────────────────
echo "=== [2/8] flash-attn (compile from source — may take 10-20 min) ==="
# --no-build-isolation: use the torch/numpy already in the venv rather than
# letting pip create a fresh build environment that might pull a different torch.
# CUDA headers come from module load cuda/12.6 ($EBROOTCUDA).
export CUDA_HOME="${EBROOTCUDA}"
MAX_JOBS=${SLURM_CPUS_PER_TASK:-8}   # use all allocated CPUs for parallel compile
export MAX_JOBS
pip install flash-attn --no-build-isolation
python -c "import flash_attn; print('  flash-attn :', flash_attn.__version__)"
echo ""

# ── 3. verl (editable install from repo) ─────────────────────────────────────
echo "=== [3/8] verl (editable install from ./verl) ==="
pip install -e "${REPO_ROOT}/verl"
python -c "import verl; print('  verl       : OK')"
echo ""

# ── 4. vLLM ──────────────────────────────────────────────────────────────────
echo "=== [4/8] vLLM 0.8.5 ==="
# vllm 0.8.5 requirements: torch>=2.4,<2.6 → satisfied by 2.5.1.
# Unlike 0.8.3, it does NOT import triton_key from triton.compiler.compiler.
# If 0.8.5 is yanked/unavailable, fall back to latest 0.8.x:
if ! pip install "vllm==0.8.5" 2>/dev/null; then
  echo "  vllm==0.8.5 unavailable; trying latest 0.8.x ..."
  pip install "vllm>=0.8.4,<0.9.0"
fi
python -c "import vllm; print('  vllm       :', vllm.__version__)"
echo ""

# ── 5. FlashInfer ─────────────────────────────────────────────────────────────
echo "=== [5/8] flashinfer (for H100 attention kernels used by vLLM) ==="
# flashinfer.ai provides pre-compiled wheels indexed by CUDA + torch version.
# cu124/torch2.5 matches our stack (CUDA 12.4 wheel, torch 2.5.x).
pip install flashinfer \
  -i https://flashinfer.ai/whl/cu124/torch2.5/ \
  --extra-index-url https://pypi.org/simple
python -c "import flashinfer; print('  flashinfer : OK')"
echo ""

# ── 6. HuggingFace / training stack ──────────────────────────────────────────
echo "=== [6/8] transformers / datasets / accelerate / deepspeed ==="
# Try Alliance wheelhouse first (--no-index) for common packages; fall back to PyPI.
pip install --no-index \
  transformers \
  datasets \
  accelerate \
  deepspeed \
  2>/dev/null || \
pip install \
  "transformers>=4.47.0" \
  "datasets>=3.0.0" \
  "accelerate>=1.0.0" \
  "deepspeed>=0.16.0"

python -c "
import transformers, datasets, accelerate, deepspeed
print('  transformers:', transformers.__version__)
print('  datasets    :', datasets.__version__)
print('  accelerate  :', accelerate.__version__)
print('  deepspeed   :', deepspeed.__version__)
"
echo ""

# ── 7. Ray and auxiliary ──────────────────────────────────────────────────────
echo "=== [7/8] ray[default], hydra-core, omegaconf, tensordict, peft ==="
# ray[default] installs the full Ray cluster (scheduler, actor runtime, dashboard).
# Try wheelhouse first; fall back to PyPI for packages not in it.
pip install --no-index ray 2>/dev/null || pip install "ray[default]>=2.30.0"
pip install --no-index hydra-core omegaconf tensordict peft 2>/dev/null || \
  pip install "hydra-core>=1.3.0" "omegaconf>=2.3.0" "tensordict>=0.5.0" peft

python -c "
import ray, hydra, omegaconf, tensordict, peft
print('  ray        :', ray.__version__)
print('  hydra-core :', hydra.__version__)
print('  omegaconf  :', omegaconf.__version__)
print('  tensordict :', tensordict.__version__)
print('  peft       :', peft.__version__)
"
echo ""

# ── 8. Remaining dependencies ─────────────────────────────────────────────────
echo "=== [8/8] math-verify, wandb, and utilities ==="
# math-verify: reward function used in compute_score.py
# wandb: experiment tracking (set WANDB_API_KEY before training)
# codetiming: timing utility used inside verl
pip install \
  math-verify \
  wandb \
  codetiming \
  psutil \
  packaging

python -c "
from math_verify.metric import math_metric
print('  math-verify: OK')
import wandb
print('  wandb      :', wandb.__version__)
"
echo ""

# ── Full verification ─────────────────────────────────────────────────────────
echo "============================================================"
echo "FULL VERIFICATION"
echo "============================================================"
python << 'PYEOF'
import sys
errors = []

checks = [
    ("torch",         lambda: __import__("torch").__version__),
    ("triton",        lambda: __import__("triton").__version__),
    ("flash_attn",    lambda: __import__("flash_attn").__version__),
    ("vllm",          lambda: __import__("vllm").__version__),
    ("flashinfer",    lambda: "OK"),
    ("verl",          lambda: "OK"),
    ("ray",           lambda: __import__("ray").__version__),
    ("transformers",  lambda: __import__("transformers").__version__),
    ("datasets",      lambda: __import__("datasets").__version__),
    ("accelerate",    lambda: __import__("accelerate").__version__),
    ("deepspeed",     lambda: __import__("deepspeed").__version__),
    ("hydra",         lambda: __import__("hydra").__version__),
    ("tensordict",    lambda: __import__("tensordict").__version__),
    ("math_verify",   lambda: (__import__("math_verify.metric", fromlist=["math_metric"]) and "OK")),
    ("wandb",         lambda: __import__("wandb").__version__),
]

for name, fn in checks:
    try:
        ver = fn()
        print(f"  OK  {name:<14} {ver}")
    except Exception as e:
        print(f"  ERR {name:<14} {e}")
        errors.append(name)

print()
if errors:
    print(f"FAILED packages: {errors}")
    sys.exit(1)
else:
    print("All packages verified successfully.")
PYEOF

echo ""
echo "============================================================"
echo "INSTALL COMPLETE"
echo "============================================================"
echo ""
echo "Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Before running training, download models and data:"
echo ""
echo "  # Actor model (Qwen2.5-7B)"
echo "  huggingface-cli download Qwen/Qwen2.5-7B \\"
echo "    --local-dir \$SCRATCH/macroscope/models/Qwen2.5-7B"
echo ""
echo "  # Verifier model"
echo "  huggingface-cli download TIGER-Lab/general-verifier \\"
echo "    --local-dir \$SCRATCH/macroscope/models/general-verifier"
echo ""
echo "  # Dataset (requires internet — run on login node)"
echo "  python data_preprocess.py \\"
echo "    --local-dir \$SCRATCH/macroscope/data/webinstruct-verified"
echo ""
echo "Then submit the training job:"
echo "  sbatch run_training_1gpu_fir.slurm"
echo ""
echo "Or run the preflight check first:"
echo "  bash preflight_check.sh"
echo ""
