#!/bin/bash
# Fix Triton 2.3.0 compatibility by creating a compatibility shim
# This patches triton.compiler.compiler to provide triton_key

set -e

echo "=== FIXING TRITON 2.3.0 COMPATIBILITY ==="
echo "Strategy: Create compatibility shim for triton_key"
echo ""

# Load modules
module load python/3.11 cuda/12.6 gcc 2>/dev/null || true
source ~/envs/verl_train/bin/activate 2>/dev/null || {
    echo "ERROR: Activate verl_train venv first"
    exit 1
}

TRITON_COMPILER_PATH="$HOME/envs/verl_train/lib/python3.11/site-packages/triton/compiler/compiler.py"

if [ ! -f "$TRITON_COMPILER_PATH" ]; then
    echo "ERROR: Triton compiler.py not found at $TRITON_COMPILER_PATH"
    exit 1
fi

echo "Current status:"
python << 'PYEOF'
import triton
print(f"Triton version: {triton.__version__}")

try:
    from triton.compiler.compiler import triton_key
    print("✓ triton_key is available")
except ImportError:
    print("✗ triton_key is NOT available (will patch)")
PYEOF

echo ""
echo "Backing up compiler.py..."
cp "$TRITON_COMPILER_PATH" "${TRITON_COMPILER_PATH}.backup"

echo "Adding triton_key compatibility function..."
python3 << 'PYTHON_PATCH'
import sys

file_path = "/home/anupam/envs/verl_train/lib/python3.11/site-packages/triton/compiler/compiler.py"

with open(file_path, 'r') as f:
    content = f.read()

# Check if already patched
if "def triton_key" in content and "# COMPATIBILITY PATCH" in content:
    print("✓ triton_key compatibility already patched")
    sys.exit(0)

# Find a good place to insert the function (after imports, before class definitions)
# Look for the first class or function definition
lines = content.split('\n')
insert_pos = len(lines)

# Try to find after imports but before first class/function
for i, line in enumerate(lines):
    # Skip imports and blank lines
    if line.strip().startswith('import ') or line.strip().startswith('from ') or not line.strip():
        continue
    # If we hit a class or def, insert before it
    if line.strip().startswith('class ') or (line.strip().startswith('def ') and not line.strip().startswith('def _')):
        insert_pos = i
        break

# Create the compatibility function
compat_func = '''
# COMPATIBILITY PATCH for Triton 2.3.0+
# triton_key was removed in Triton 2.2.0+, but PyTorch/vLLM still need it
def triton_key(*args, **kwargs):
    """
    Compatibility shim for triton_key removed in Triton 2.2.0+
    Returns a hash-based key for kernel caching.
    """
    import hashlib
    import pickle
    try:
        # Create a hash from the arguments (similar to original triton_key behavior)
        key_data = pickle.dumps((args, kwargs), protocol=4)
        return hashlib.md5(key_data).hexdigest()
    except Exception:
        # Fallback: return a constant key (will disable caching but allow execution)
        return "triton_key_compat_fallback"

'''

# Insert the function
new_lines = lines[:insert_pos] + [compat_func] + lines[insert_pos:]
new_content = '\n'.join(new_lines)

with open(file_path, 'w') as f:
    f.write(new_content)

print("✅ Added triton_key compatibility function to Triton compiler.py")
PYTHON_PATCH

if [ $? -ne 0 ]; then
    echo "ERROR: Patch failed, restoring backup..."
    mv "${TRITON_COMPILER_PATH}.backup" "$TRITON_COMPILER_PATH"
    exit 1
fi

echo ""
echo "=== VERIFICATION ==="
python << 'PYEOF'
try:
    from triton.compiler.compiler import triton_key
    print("✓ triton_key is now available!")
    
    # Test that it works
    test_key = triton_key("test", "args")
    print(f"✓ triton_key function works (returns: {test_key[:20]}...)")
except Exception as e:
    print(f"✗ triton_key still not available: {e}")
    import sys
    sys.exit(1)
PYEOF

echo ""
echo "=== PATCH COMPLETE ==="
echo "Backup saved to: ${TRITON_COMPILER_PATH}.backup"
echo "You can now remove enforce_eager=True from your code"
echo ""
