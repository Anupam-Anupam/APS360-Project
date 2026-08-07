#!/bin/bash
# Check versions of CUDA, PyTorch, Triton to diagnose PTX mismatch

echo "=== VERSION CHECK ==="
echo ""

# Load modules
module load python/3.11 cuda/12.6 gcc 2>/dev/null || true
source ~/envs/verl_train/bin/activate 2>/dev/null || true

echo "CUDA Module:"
module show cuda/12.6 2>&1 | grep -E "Version|PATH" | head -5
echo ""

if command -v nvcc &> /dev/null; then
    echo "NVCC Version:"
    nvcc --version | grep "release"
    echo ""
fi

echo "Python Packages:"
python << 'PYEOF'
import sys
try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
except Exception as e:
    print(f"torch: ERROR - {e}")

try:
    import triton
    print(f"triton: {triton.__version__}")
except Exception as e:
    print(f"triton: ERROR - {e}")

try:
    import vllm
    print(f"vllm: {vllm.__version__}")
except Exception as e:
    print(f"vllm: ERROR - {e}")

try:
    import transformers
    print(f"transformers: {transformers.__version__}")
except Exception as e:
    print(f"transformers: ERROR - {e}")
PYEOF

echo ""
echo "=== PTX COMPATIBILITY ==="
echo "CUDA 12.6.2 supports PTX up to version 8.5"
echo "If Triton generates PTX 8.6, we need to either:"
echo "  1. Downgrade Triton to version that generates PTX 8.5"
echo "  2. Use enforce_eager=True to disable Triton compilation"
echo "  3. Upgrade CUDA (if available on cluster)"
echo ""
