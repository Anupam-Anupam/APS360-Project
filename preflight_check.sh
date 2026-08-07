#!/bin/bash
# Preflight check script - catch errors BEFORE submitting to GPU cluster
# Run this on the login node to validate everything before sbatch

set -e

echo "=========================================="
echo "PREFLIGHT CHECK - CATCH ERRORS BEFORE GPU"
echo "=========================================="
echo ""

# Load modules
module load python/3.11 cuda/12.6 gcc opencv arrow/22.0.0 cudnn nccl scipy-stack 2>&1 || {
    echo "ERROR: Failed to load modules"
    exit 1
}

# Activate venv
source ~/envs/verl_train/bin/activate || {
    echo "ERROR: Failed to activate verl_train venv"
    exit 1
}

cd "$SCRATCH/MaCroScope/MaCroScope" 2>/dev/null || cd "$SCRATCH/MaCroScope" || {
    echo "ERROR: Cannot find working directory"
    exit 1
}

echo "Working directory: $(pwd)"
echo ""

# 1. Check Python syntax
echo "=== 1. Python Syntax Check ==="
python -m py_compile main_ppo.py && echo "✓ main_ppo.py syntax OK" || {
    echo "✗ main_ppo.py has syntax errors"
    exit 1
}

python -m py_compile compute_score.py && echo "✓ compute_score.py syntax OK" || {
    echo "✗ compute_score.py has syntax errors"
    exit 1
}

python -m py_compile verifier.py && echo "✓ verifier.py syntax OK" || {
    echo "✗ verifier.py has syntax errors"
    exit 1
}
echo ""

# 2. Check imports (without GPU)
echo "=== 2. Import Check (No GPU Required) ==="
python << 'PYEOF'
import sys
errors = []

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except Exception as e:
    errors.append(f"✗ torch: {e}")

try:
    import verl
    print("✓ verl")
except Exception as e:
    errors.append(f"✗ verl: {e}")

try:
    import hydra
    print("✓ hydra")
except Exception as e:
    errors.append(f"✗ hydra: {e}")

try:
    import omegaconf
    print("✓ omegaconf")
except Exception as e:
    errors.append(f"✗ omegaconf: {e}")

try:
    import ray
    print(f"✓ ray {ray.__version__}")
except Exception as e:
    errors.append(f"✗ ray: {e}")

try:
    from math_verify.metric import math_metric
    print("✓ math_verify")
except Exception as e:
    errors.append(f"✗ math_verify: {e}")

try:
    from tensordict import TensorDict
    print("✓ tensordict")
except Exception as e:
    errors.append(f"✗ tensordict: {e}")

# vLLM and flash_attn may fail on login node (no GPU), that's OK
try:
    import vllm
    print(f"✓ vllm {vllm.__version__}")
except Exception as e:
    print(f"  WARNING: vllm import failed (expected on login node): {e}")

try:
    import flash_attn
    print("✓ flash_attn")
except Exception as e:
    print(f"  WARNING: flash_attn import failed (expected on login node): {e}")

if errors:
    print("\nCRITICAL ERRORS:")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Import check failed"
    exit 1
fi
echo ""

# 3. Check for runtime env conflicts
echo "=== 3. Runtime Environment Conflict Check ==="
python << 'PYEOF'
import re
import sys

# Read main_ppo.py
with open('main_ppo.py', 'r') as f:
    main_ppo_content = f.read()

# Read train_general_reasoner.sh
with open('train_general_reasoner.sh', 'r') as f:
    train_sh_content = f.read()

# Find env vars in ray.init() in main_ppo.py
ray_init_env_vars = set()
ray_init_match = re.search(r'ray\.init\(runtime_env=\{.*?\'env_vars\':\s*\{([^}]+)\}', main_ppo_content, re.DOTALL)
if ray_init_match:
    env_vars_block = ray_init_match.group(1)
    for match in re.finditer(r"'([^']+)':\s*'[^']*'", env_vars_block):
        ray_init_env_vars.add(match.group(1))

# Find env vars in ray job submit runtime-env-json
job_env_vars = set()
job_env_match = re.search(r'--runtime-env-json=.*?"env_vars":\s*\{([^}]+)\}', train_sh_content, re.DOTALL)
if job_env_match:
    env_vars_block = job_env_match.group(1)
    for match in re.finditer(r'"([^"]+)":\s*"[^"]*"', env_vars_block):
        job_env_vars.add(match.group(1))

# Check for conflicts
conflicts = ray_init_env_vars.intersection(job_env_vars)
if conflicts:
    print(f"✗ RUNTIME ENV CONFLICT DETECTED!")
    print(f"  The following env vars are set in BOTH ray.init() AND ray job submit:")
    for var in conflicts:
        print(f"    - {var}")
    print(f"\n  Fix: Remove these from main_ppo.py's ray.init() call")
    print(f"  (They're already set in train_general_reasoner.sh's runtime-env-json)")
    sys.exit(1)
else:
    print("✓ No runtime env conflicts detected")
    print(f"  ray.init() env vars: {sorted(ray_init_env_vars) if ray_init_env_vars else '(none)'}")
    print(f"  job submit env vars: {sorted(job_env_vars) if job_env_vars else '(none)'}")
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Runtime env conflict check failed"
    exit 1
fi
echo ""

# 4. Check function signatures match
echo "=== 4. Function Signature Check ==="
if grep -q "^def compute_score" compute_score.py; then
    echo "✓ compute_score function found"
else
    echo "✗ compute_score function MISSING"
    exit 1
fi

if grep -q "custom_reward_function.name=compute_score" train_general_reasoner.sh; then
    echo "✓ custom_reward_function.name matches function name"
else
    echo "✗ custom_reward_function.name doesn't match function name"
    exit 1
fi

# Check verifier import
if grep -q "from verifier import RewardModelWorker" main_ppo.py || grep -q "import verifier" main_ppo.py; then
    echo "✓ verifier import in main_ppo.py"
else
    echo "✗ verifier import MISSING in main_ppo.py"
    exit 1
fi
echo ""

# 5. Validate Hydra config path exists
echo "=== 5. Hydra Config Path Check ==="
CONFIG_PATH=$(grep -oP "config_path='\K[^']+" main_ppo.py | head -1)
if [ -n "$CONFIG_PATH" ]; then
    if [ -d "$CONFIG_PATH" ]; then
        echo "✓ Config directory exists: $CONFIG_PATH"
        if [ -f "$CONFIG_PATH/ppo_trainer.yaml" ] || [ -f "$CONFIG_PATH/ppo_trainer.yml" ]; then
            echo "  ✓ ppo_trainer config file found"
        else
            echo "  WARNING: ppo_trainer.yaml not found (may be in verl submodule)"
        fi
    else
        echo "  WARNING: Config directory not found: $CONFIG_PATH (may be in verl submodule)"
    fi
else
    echo "  WARNING: Could not extract config_path from main_ppo.py"
fi
echo ""

# 6. Check paths exist
echo "=== 6. Path Existence Check ==="
MODEL_PATH="$HOME/scratch/macroscope/models/Qwen3-4B-Base"
VERIFIER_PATH="$HOME/scratch/macroscope/models/general-verifier"
DATASET_PATH="$HOME/scratch/macroscope/data/webinstruct-verified"

[ -d "$MODEL_PATH" ] && echo "✓ Model path exists" || {
    echo "✗ Model path MISSING: $MODEL_PATH"
    exit 1
}

[ -d "$VERIFIER_PATH" ] && echo "✓ Verifier path exists" || {
    echo "✗ Verifier path MISSING: $VERIFIER_PATH"
    exit 1
}

[ -d "$DATASET_PATH" ] && echo "✓ Dataset path exists" || {
    echo "✗ Dataset path MISSING: $DATASET_PATH"
    exit 1
}

[ -f "$DATASET_PATH/train.parquet" ] && echo "✓ train.parquet exists" || {
    echo "✗ train.parquet MISSING"
    exit 1
}

[ -f "$DATASET_PATH/test.parquet" ] && echo "✓ test.parquet exists" || {
    echo "✗ test.parquet MISSING"
    exit 1
}
echo ""

# 7. Validate train_general_reasoner.sh critical fixes
echo "=== 7. Critical Fixes Check ==="
if grep -q "critic.model.path=\$HDFS_MODEL_PATH/\$MODEL_NAME" train_general_reasoner.sh; then
    echo "✓ critic.model.path fix present"
else
    echo "✗ critic.model.path fix MISSING"
    exit 1
fi

if grep -q "custom_reward_function.name=compute_score" train_general_reasoner.sh; then
    echo "✓ custom_reward_function.name fix present"
else
    echo "✗ custom_reward_function.name fix MISSING"
    exit 1
fi
echo ""

# 8. Test Ray job submit syntax (dry run - doesn't actually submit)
echo "=== 8. Ray Job Submit Syntax Check ==="
python << 'PYEOF'
import json
import re
import sys

# Read train_general_reasoner.sh
with open('train_general_reasoner.sh', 'r') as f:
    content = f.read()

# Extract runtime-env-json
json_match = re.search(r"--runtime-env-json='([^']+)'", content)
if json_match:
    json_str = json_match.group(1)
    try:
        # Replace single quotes with double quotes for valid JSON
        json_str = json_str.replace("'", '"')
        # Replace variable references with placeholders for validation
        json_str = re.sub(r'\$\{[^}]+\}', '"PLACEHOLDER"', json_str)
        runtime_env = json.loads(json_str)
        print("✓ runtime-env-json is valid JSON")
        if 'env_vars' in runtime_env:
            print(f"  ✓ Found {len(runtime_env['env_vars'])} env vars")
    except json.JSONDecodeError as e:
        print(f"✗ runtime-env-json is INVALID JSON: {e}")
        sys.exit(1)
else:
    print("  WARNING: Could not find runtime-env-json in train_general_reasoner.sh")
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Ray job submit syntax check failed"
    exit 1
fi
echo ""

# 9. Check for common mistakes
echo "=== 9. Common Mistakes Check ==="

# Check for hardcoded CUDA_VISIBLE_DEVICES in runtime-env
if grep -q '"CUDA_VISIBLE_DEVICES"' train_general_reasoner.sh; then
    echo "✗ WARNING: CUDA_VISIBLE_DEVICES found in runtime-env (should be removed)"
    echo "  Use RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 instead"
else
    echo "✓ No hardcoded CUDA_VISIBLE_DEVICES in runtime-env"
fi

# Check WORKING_DIR is set
if grep -q 'WORKING_DIR' train_general_reasoner.sh && ! grep -q 'WORKING_DIR=\$' train_general_reasoner.sh; then
    echo "  WARNING: WORKING_DIR may not be set (check slurm script sets it)"
else
    echo "✓ WORKING_DIR handling looks OK"
fi

# Check HEAD_IP is set
if grep -q 'HEAD_IP' train_general_reasoner.sh && ! grep -q 'HEAD_IP=\$' train_general_reasoner.sh; then
    echo "  WARNING: HEAD_IP may not be set (check slurm script sets it)"
else
    echo "✓ HEAD_IP handling looks OK"
fi
echo ""

# 10. Test compute_score function can be imported
echo "=== 10. Custom Function Import Test ==="
python << 'PYEOF'
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("compute_score_module", "compute_score.py")
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
    if hasattr(module, 'compute_score'):
        print("✓ compute_score function can be imported")
        # Check function signature
        import inspect
        sig = inspect.signature(module.compute_score)
        params = list(sig.parameters.keys())
        expected = ['data_source', 'solution_str', 'ground_truth', 'extra_info']
        if params == expected or params == expected[:3]:
            print(f"  ✓ Function signature OK: {params}")
        else:
            print(f"  WARNING: Unexpected signature: {params} (expected: {expected})")
    else:
        print("✗ compute_score function not found in module")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed to import compute_score: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Custom function import test failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "PREFLIGHT CHECK COMPLETE"
echo "=========================================="
echo ""
echo "✓ All checks passed! Safe to submit to GPU cluster."
echo ""
echo "Note: Some checks (like vLLM/flash_attn imports) may show warnings"
echo "on the login node - that's expected. They'll work on compute nodes."
echo ""
