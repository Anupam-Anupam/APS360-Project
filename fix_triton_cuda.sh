#!/bin/bash
# Fix Triton/CUDA PTX version mismatch
# CUDA 12.6.2 supports PTX 8.5 max, but Triton generates PTX 8.6

set -e

echo "=== TRITON/CUDA VERSION FIX ==="
echo ""

# Load modules
module load python/3.11 cuda/12.6 gcc 2>/dev/null || true
source ~/envs/verl_train/bin/activate 2>/dev/null || {
    echo "ERROR: Activate verl_train venv first"
    exit 1
}

echo "Current versions:"
python << 'PYEOF'
import torch
import triton
print(f"PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
print(f"Triton: {triton.__version__}")
PYEOF

echo ""
echo "=== FIXING TRITON VERSION ==="
echo "CUDA 12.6.2 requires Triton that generates PTX 8.5 or lower"
echo "Triton 3.0.0+ generates PTX 8.6 (incompatible)"
echo "Triton 2.1.0 generates PTX 8.5 (compatible)"
echo ""

# Check if we can downgrade Triton
echo "Attempting to install compatible Triton version..."
pip install "triton>=2.1.0,<3.0.0" --force-reinstall --no-deps 2>&1 | tail -20

echo ""
echo "Verifying fix:"
python << 'PYEOF'
import triton
print(f"Triton version: {triton.__version__}")
# Check if it's compatible
major, minor = map(int, triton.__version__.split('.')[:2])
if major < 3:
    print("✓ Triton version should generate PTX 8.5 (compatible)")
else:
    print("✗ Triton version may still generate PTX 8.6")
PYEOF

echo ""
echo "=== ALTERNATIVE: Use enforce_eager ==="
echo "If downgrading Triton doesn't work, use enforce_eager=True"
echo "This disables Triton compilation entirely (slower but works)"
echo ""
