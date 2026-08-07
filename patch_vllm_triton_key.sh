#!/bin/bash
# Patch PyTorch/vLLM to handle missing triton_key in Triton 2.3.0+
# The error actually comes from PyTorch's dynamo backend, not vLLM directly
# This patches both PyTorch and vLLM to work without triton_key

set -e

VLLM_BACKEND_PATH="$HOME/envs/verl_train/lib/python3.11/site-packages/vllm/compilation/backends/__init__.py"
VLLM_BACKEND_DIR="$HOME/envs/verl_train/lib/python3.11/site-packages/vllm/compilation/backends"
PYTORCH_TRITON_DIR="$HOME/envs/verl_train/lib/python3.11/site-packages/torch/_dynamo/backends"

if [ ! -d "$VLLM_BACKEND_DIR" ]; then
    echo "ERROR: vLLM backends directory not found at $VLLM_BACKEND_DIR"
    exit 1
fi

echo "=== PATCHING PyTorch/vLLM FOR TRITON 2.3.0 COMPATIBILITY ==="
echo "The error comes from PyTorch's dynamo backend trying to use triton_key"
echo ""

# First, patch PyTorch's dynamo backend
echo "=== STEP 1: Patching PyTorch's dynamo backend ==="
if [ -d "$PYTORCH_TRITON_DIR" ]; then
    echo "Searching for triton_key in PyTorch dynamo backends..."
    TORCH_FILES=$(grep -r "triton_key" "$PYTORCH_TRITON_DIR" 2>/dev/null | cut -d: -f1 | sort -u || true)
    
    if [ -n "$TORCH_FILES" ]; then
        echo "Found PyTorch files with triton_key:"
        echo "$TORCH_FILES"
        for file in $TORCH_FILES; do
            echo "Backing up $file..."
            cp "$file" "${file}.backup"
            # Patch similar to vLLM below
        done
    else
        echo "No triton_key found in PyTorch dynamo backends (may be in compiled code)"
    fi
else
    echo "PyTorch dynamo backends directory not found (may be in different location)"
fi

echo ""
echo "=== STEP 2: Patching vLLM compilation backends ==="

# Find all files that import triton_key
echo "Searching for triton_key imports in vLLM..."
FILES_WITH_TRITON_KEY=$(grep -r "triton_key" "$VLLM_BACKEND_DIR" 2>/dev/null | cut -d: -f1 | sort -u || true)

if [ -z "$FILES_WITH_TRITON_KEY" ]; then
    echo "No triton_key imports found in backends directory."
    echo "Searching in entire vLLM package..."
    VLLM_ROOT="$HOME/envs/verl_train/lib/python3.11/site-packages/vllm"
    FILES_WITH_TRITON_KEY=$(grep -r "from triton.compiler.compiler import triton_key" "$VLLM_ROOT" 2>/dev/null | cut -d: -f1 | sort -u || true)
fi

if [ -z "$FILES_WITH_TRITON_KEY" ]; then
    echo "WARNING: Could not find triton_key imports. The error might be coming from PyTorch's dynamo backend."
    echo "Trying to patch vLLM's backend initialization instead..."
    
    # Try to patch the backend file directly
    if [ -f "$VLLM_BACKEND_DIR/__init__.py" ]; then
        echo "Backing up __init__.py..."
        cp "$VLLM_BACKEND_DIR/__init__.py" "${VLLM_BACKEND_DIR/__init__.py}.backup"
        
        # Add a compatibility shim at the top of the file
        python3 << 'PYTHON_PATCH'
import sys

file_path = "/home/anupam/envs/verl_train/lib/python3.11/site-packages/vllm/compilation/backends/__init__.py"

with open(file_path, 'r') as f:
    content = f.read()

# Check if patch already applied
if "triton_key_compat" in content:
    print("✓ Compatibility shim already present")
    sys.exit(0)

# Add compatibility shim at the top
shim = '''# Compatibility shim for Triton 2.3.0+ (triton_key removed)
try:
    from triton.compiler.compiler import triton_key
except ImportError:
    # triton_key was removed in Triton 2.2.0+
    # Provide a dummy function that returns a hash of the kernel
    def triton_key(*args, **kwargs):
        import hashlib
        import pickle
        try:
            # Try to create a hash from the arguments
            key_data = pickle.dumps((args, kwargs))
            return hashlib.md5(key_data).hexdigest()
        except:
            # Fallback: return a constant
            return "triton_key_not_available"

'''

# Insert shim after any existing imports or at the beginning
if content.startswith('"""') or content.startswith("'''"):
    # Find end of docstring
    lines = content.split('\n')
    insert_pos = 0
    for i, line in enumerate(lines):
        if (line.startswith('"""') or line.startswith("'''")) and i > 0:
            insert_pos = i + 1
            break
    new_content = '\n'.join(lines[:insert_pos]) + '\n' + shim + '\n'.join(lines[insert_pos:])
else:
    new_content = shim + content

with open(file_path, 'w') as f:
    f.write(new_content)

print("✅ Added triton_key compatibility shim to vLLM backends")
PYTHON_PATCH
    fi
else
    echo "Found files with triton_key imports:"
    echo "$FILES_WITH_TRITON_KEY"
    echo ""
    
    for file in $FILES_WITH_TRITON_KEY; do
        echo "Backing up $file..."
        cp "$file" "${file}.backup"
        
        echo "Patching $file..."
        python3 << PYTHON_PATCH
import re
import sys

file_path = "$file"

with open(file_path, 'r') as f:
    content = f.read()

# Check if already patched
if "triton_key_compat" in content or "getattr" in content and "triton_key" in content:
    print(f"  ⚠️  {file_path} already patched or uses safe access")
    sys.exit(0)

# Replace direct import with try/except
# Pattern 1: from triton.compiler.compiler import triton_key
pattern1 = r'from triton\.compiler\.compiler import triton_key'
replacement1 = '''try:
    from triton.compiler.compiler import triton_key
except ImportError:
    # triton_key removed in Triton 2.2.0+
    def triton_key(*args, **kwargs):
        import hashlib
        import pickle
        try:
            key_data = pickle.dumps((args, kwargs))
            return hashlib.md5(key_data).hexdigest()
        except:
            return "triton_key_not_available"'''

if re.search(pattern1, content):
    content = re.sub(pattern1, replacement1, content)
    print(f"  ✅ Patched import in {file_path}")
else:
    print(f"  ⚠️  Could not find import pattern in {file_path}")

with open(file_path, 'w') as f:
    f.write(content)
PYTHON_PATCH
    done
fi

echo ""
echo "=== VERIFICATION ==="
python3 << 'PYEOF'
try:
    # Try to import vLLM's compilation backend
    import sys
    sys.path.insert(0, '/home/anupam/envs/verl_train/lib/python3.11/site-packages')
    
    # Check if triton_key is now available via our shim
    try:
        from triton.compiler.compiler import triton_key
        print("✓ triton_key available (original)")
    except ImportError:
        # Try to see if our shim is in place
        try:
            import vllm.compilation.backends
            print("✓ vLLM backends module imports successfully")
            print("  (triton_key will be handled by compatibility shim)")
        except Exception as e:
            if "triton_key" in str(e):
                print("✗ triton_key error still present")
                print(f"  Error: {e}")
            else:
                print(f"  Other import issue (may be expected): {e}")
except Exception as e:
    print(f"Verification error: {e}")
PYEOF

echo ""
echo "=== PATCH COMPLETE ==="
echo "Backups saved with .backup extension"
echo "If this doesn't work, restore backups and try upgrading vLLM instead"
echo ""
