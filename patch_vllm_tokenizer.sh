#!/bin/bash
# Patch vLLM tokenizer to handle missing all_special_tokens_extended attribute
# This fixes compatibility with transformers 5.2.0 and Qwen2Tokenizer

set -e

VLLM_TOKENIZER_PATH="$HOME/envs/verl_train/lib/python3.11/site-packages/vllm/transformers_utils/tokenizer.py"

if [ ! -f "$VLLM_TOKENIZER_PATH" ]; then
    echo "ERROR: vLLM tokenizer file not found at $VLLM_TOKENIZER_PATH"
    exit 1
fi

echo "Backing up original tokenizer.py..."
cp "$VLLM_TOKENIZER_PATH" "${VLLM_TOKENIZER_PATH}.backup"

echo "Applying patch to handle missing all_special_tokens_extended..."

# Create a Python script to apply the patch
python3 << 'PYTHON_PATCH'
import re
import sys

file_path = "/home/anupam/envs/verl_train/lib/python3.11/site-packages/vllm/transformers_utils/tokenizer.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

# Find the line that accesses all_special_tokens_extended (around line 83 based on error)
patched = False
new_lines = []

for i, line in enumerate(lines):
    # Look for direct access to all_special_tokens_extended
    if 'all_special_tokens_extended' in line and 'getattr' not in line:
        # Replace direct access with safe getattr
        # Pattern: tokenizer.all_special_tokens_extended)
        new_line = re.sub(
            r'(\w+)\.all_special_tokens_extended\)',
            r'getattr(\1, "all_special_tokens_extended", getattr(\1, "all_special_tokens", [])))',
            line
        )
        # Also handle cases without closing paren
        new_line = re.sub(
            r'(\w+)\.all_special_tokens_extended\b',
            r'getattr(\1, "all_special_tokens_extended", getattr(\1, "all_special_tokens", []))',
            new_line
        )
        if new_line != line:
            print(f"✅ Patching line {i+1}: {line.strip()[:60]}...")
            patched = True
        new_lines.append(new_line)
    else:
        new_lines.append(line)

if not patched:
    print("⚠️  Could not find all_special_tokens_extended to patch")
    print("File might already be patched or structure is different")
    # Show context around line 83 for debugging
    if len(lines) > 80:
        print("\nContext around line 83:")
        for i in range(max(0, 78), min(len(lines), 88)):
            print(f"{i+1:4d}: {lines[i].rstrip()}")
    sys.exit(1)

with open(file_path, 'w') as f:
    f.writelines(new_lines)

print("✅ Patch applied successfully!")

PYTHON_PATCH

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ vLLM tokenizer patched successfully!"
    echo "Backup saved to: ${VLLM_TOKENIZER_PATH}.backup"
    echo ""
    echo "To verify the patch, run:"
    echo "  python3 -c \"from vllm.transformers_utils.tokenizer import get_cached_tokenizer; print('Import OK')\""
else
    echo "❌ Patch failed. Restoring backup..."
    mv "${VLLM_TOKENIZER_PATH}.backup" "$VLLM_TOKENIZER_PATH"
    exit 1
fi
