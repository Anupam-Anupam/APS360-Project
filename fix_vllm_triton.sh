#!/bin/bash
# Fix vLLM/Triton incompatibility by upgrading vLLM instead of downgrading Triton
# This keeps Triton 2.3.0 and upgrades vLLM to a compatible version

set -e

echo "=== FIXING vLLM/Triton INCOMPATIBILITY ==="
echo "Strategy: Keep Triton 2.3.0, upgrade vLLM to compatible version"
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
try:
    import vllm
    vllm_version = vllm.__version__
except:
    vllm_version = "NOT INSTALLED"

print(f"PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
print(f"Triton: {triton.__version__}")
print(f"vLLM: {vllm_version}")

# Check if triton_key exists
try:
    from triton.compiler.compiler import triton_key
    print("✓ triton_key is available in Triton")
except ImportError:
    print("✗ triton_key is NOT available in Triton (this is the problem)")
PYEOF

echo ""
echo "=== DIAGNOSIS ==="
echo "The error 'cannot import name triton_key' means:"
echo "  - Triton 2.3.0 doesn't have triton_key (removed in 2.2.0+)"
echo "  - vLLM 0.8.4 still tries to import it"
echo "  - We need to upgrade vLLM to a version that doesn't use triton_key"
echo ""

echo "=== FIXING: Upgrade vLLM to latest version compatible with Triton 2.3.0 ==="
echo "vLLM 0.8.4+ should have fixes for newer Triton versions"
echo ""

# Check what vLLM version is currently installed
CURRENT_VLLM=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "Current vLLM: $CURRENT_VLLM"

# Try to upgrade vLLM to latest 0.8.x or 0.9.x
echo ""
echo "Attempting to upgrade vLLM to latest version..."
echo "This may take a few minutes..."

# First, try to upgrade from wheelhouse (if available)
pip install --upgrade vllm --no-deps 2>&1 | tail -30

# If that doesn't work or doesn't upgrade, try from PyPI
NEW_VLLM=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
if [ "$NEW_VLLM" = "$CURRENT_VLLM" ] || [ "$NEW_VLLM" = "unknown" ]; then
    echo ""
    echo "Wheelhouse upgrade didn't change version, trying PyPI..."
    pip install --upgrade vllm --no-deps --index-url https://pypi.org/simple 2>&1 | tail -30
fi

echo ""
echo "=== VERIFICATION ==="
python << 'PYEOF'
import triton
try:
    import vllm
    print(f"vLLM version: {vllm.__version__}")
    
    # Try to import the problematic module to see if it still uses triton_key
    try:
        # This will fail if vLLM still tries to import triton_key
        from vllm.compilation.backends import VllmBackend
        print("✓ vLLM compilation backend imports successfully")
    except ImportError as e:
        if "triton_key" in str(e):
            print("✗ vLLM still tries to use triton_key - may need newer version")
        else:
            print(f"  Import error (may be expected): {e}")
    except Exception as e:
        print(f"  Other error (may be expected): {e}")
        
except Exception as e:
    print(f"✗ vLLM import failed: {e}")
PYEOF

echo ""
echo "=== NEXT STEPS ==="
echo "1. If vLLM upgraded successfully, test with a new job"
echo "2. If still failing, we may need to patch vLLM to remove triton_key usage"
echo "3. Alternative: Check if vLLM 0.9.x or newer has better Triton 2.3.0 support"
echo ""
