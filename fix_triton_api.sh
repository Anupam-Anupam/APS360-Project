#!/bin/bash
# Fix Triton API incompatibility: triton_key import error
# This addresses the root cause, not a workaround

set -e

echo "=== TRITON API COMPATIBILITY FIX ==="
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

# Check if triton_key exists
try:
    from triton.compiler.compiler import triton_key
    print("✓ triton_key is available")
except ImportError:
    print("✗ triton_key is NOT available (this is the problem)")
PYEOF

echo ""
echo "=== DIAGNOSIS ==="
echo "The error 'cannot import name triton_key' means:"
echo "  - Triton 3.0+ removed the triton_key function"
echo "  - PyTorch/vLLM still try to import it"
echo "  - We need Triton < 3.0.0 that has triton_key"
echo ""

echo "=== FIXING: Downgrade Triton to 2.1.x (has triton_key, generates PTX 8.5) ==="
echo "This version:"
echo "  - Has triton_key API (compatible with PyTorch/vLLM)"
echo "  - Generates PTX 8.5 (compatible with CUDA 12.6.2)"
echo ""
echo "Note: Triton 2.2.0+ removed triton_key, so we need 2.1.x specifically"
echo ""

# Try to install 2.1.0 specifically, or the latest 2.1.x if available
echo "Attempting to install Triton 2.1.0 from wheelhouse..."
pip install "triton==2.1.0" --force-reinstall --no-deps 2>&1 | tail -30

# If 2.1.0 not available, try 2.1.x range
if [ $? -ne 0 ]; then
    echo ""
    echo "2.1.0 not available in wheelhouse, trying latest 2.1.x..."
    pip install "triton>=2.1.0,<2.2.0" --force-reinstall --no-deps 2>&1 | tail -30
    
    # If still not available, try from PyPI (may need to build from source)
    if [ $? -ne 0 ]; then
        echo ""
        echo "2.1.x not available in wheelhouse, trying PyPI..."
        pip install "triton==2.1.0" --force-reinstall --no-deps --index-url https://pypi.org/simple 2>&1 | tail -30
    fi
fi

echo ""
echo "=== VERIFICATION ==="
python << 'PYEOF'
import triton
print(f"Triton version: {triton.__version__}")

# Check if triton_key exists now
try:
    from triton.compiler.compiler import triton_key
    print("✓ triton_key is now available - FIXED!")
except ImportError:
    print("✗ triton_key still not available - may need different version")
    # Try to find what's available
    try:
        import triton.compiler.compiler as compiler
        print(f"Available in compiler module: {dir(compiler)[:10]}")
    except:
        pass
PYEOF

echo ""
echo "=== NEXT STEPS ==="
echo "1. If triton_key is available, you can remove enforce_eager=True"
echo "2. Test with a new job to verify the fix"
echo ""
