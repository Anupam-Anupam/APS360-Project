#!/bin/bash
# Comprehensive verification script for Nibi and Fir clusters
# Run this on both clusters to verify everything is ready

set -e

echo "=========================================="
echo "COMPREHENSIVE SETUP VERIFICATION"
echo "=========================================="
echo ""

# Detect cluster
if [[ "$HOSTNAME" == *"nibi"* ]]; then
    CLUSTER="NIBI"
    WORK_DIR="${SCRATCH:-$HOME/scratch}/MaCroScope"
elif [[ "$HOSTNAME" == *"fir"* ]] || [[ "$HOSTNAME" == *"login"* ]]; then
    CLUSTER="FIR"
    WORK_DIR="$SCRATCH/MaCroScope"
else
    CLUSTER="UNKNOWN"
    WORK_DIR="$SCRATCH/MaCroScope"
fi

echo "Cluster: $CLUSTER"
echo "Working Directory: $WORK_DIR"
echo ""

# Load modules
echo "=== Loading Modules ==="
module load python/3.11 cuda/12.6 gcc opencv arrow/22.0.0 cudnn nccl scipy-stack 2>&1 || {
    echo "WARNING: Some modules may not be available on login node"
}
echo ""

# Activate venv
echo "=== Activating Virtual Environment ==="
source ~/envs/verl_train/bin/activate || {
    echo "ERROR: Failed to activate verl_train venv"
    exit 1
}
echo "Python: $(python --version)"
echo "Python path: $(which python)"
echo ""

# 1. FILE EXISTENCE AND PERMISSIONS
echo "=== 1. FILE EXISTENCE ==="
cd "$WORK_DIR" || {
    echo "ERROR: Cannot cd to $WORK_DIR"
    exit 1
}

FILES=(
    "train_general_reasoner.sh"
    "run_training_${CLUSTER,,}.slurm"
    "compute_score.py"
    "main_ppo.py"
    "verifier.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists ($(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null) bytes)"
    else
        echo "✗ $file MISSING"
        exit 1
    fi
done
echo ""

# 2. CRITICAL FIXES IN train_general_reasoner.sh
echo "=== 2. CRITICAL FIXES IN train_general_reasoner.sh ==="
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

if grep -q "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" train_general_reasoner.sh; then
    echo "✓ RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES present"
else
    echo "✗ RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES MISSING"
    exit 1
fi

if grep -q "NCCL_SOCKET_IFNAME" train_general_reasoner.sh; then
    echo "✓ NCCL_SOCKET_IFNAME present"
else
    echo "✗ NCCL_SOCKET_IFNAME MISSING"
    exit 1
fi
echo ""

# 3. SLURM SCRIPT CONFIGURATION
echo "=== 3. SLURM SCRIPT CONFIGURATION ==="
SLURM_FILE="run_training_${CLUSTER,,}.slurm"

if grep -q "NCCL_SOCKET_IFNAME" "$SLURM_FILE"; then
    echo "✓ NCCL_SOCKET_IFNAME in slurm script"
else
    echo "✗ NCCL_SOCKET_IFNAME MISSING in slurm script"
    exit 1
fi

if grep -q "NCCL_IB_DISABLE" "$SLURM_FILE"; then
    echo "✓ NCCL_IB_DISABLE in slurm script"
else
    echo "✗ NCCL_IB_DISABLE MISSING in slurm script"
    exit 1
fi

if grep -q "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" "$SLURM_FILE"; then
    echo "✓ RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES in slurm script"
else
    echo "✗ RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES MISSING in slurm script"
    exit 1
fi

if [ "$CLUSTER" == "NIBI" ]; then
    if grep -q "gpus-per-node=h100:4" "$SLURM_FILE"; then
        echo "✓ gpus-per-node=h100:4 (full H100s, not MIG)"
    else
        echo "✗ gpus-per-node=h100:4 MISSING"
        exit 1
    fi
fi
echo ""

# 4. PYTHON PACKAGE IMPORTS
echo "=== 4. PYTHON PACKAGE IMPORTS ==="
python << 'PYEOF'
import sys
errors = []

# Core packages
try:
    import torch
    print(f"✓ torch {torch.__version__}")
    if not torch.cuda.is_available():
        print("  WARNING: CUDA not available (expected on login node)")
except Exception as e:
    errors.append(f"✗ torch: {e}")

try:
    import vllm
    print(f"✓ vllm {vllm.__version__}")
except Exception as e:
    errors.append(f"✗ vllm: {e}")

try:
    import flash_attn
    print(f"✓ flash_attn")
except Exception as e:
    errors.append(f"✗ flash_attn: {e}")

try:
    from math_verify.metric import math_metric
    print("✓ math_verify")
except Exception as e:
    errors.append(f"✗ math_verify: {e}")

try:
    import tensordict
    print(f"✓ tensordict")
except Exception as e:
    errors.append(f"✗ tensordict: {e}")

try:
    import ray
    print(f"✓ ray {ray.__version__}")
except Exception as e:
    errors.append(f"✗ ray: {e}")

try:
    import hydra
    print(f"✓ hydra-core")
except Exception as e:
    errors.append(f"✗ hydra-core: {e}")

try:
    import omegaconf
    print(f"✓ omegaconf")
except Exception as e:
    errors.append(f"✗ omegaconf: {e}")

try:
    import verl
    print(f"✓ verl")
except Exception as e:
    errors.append(f"✗ verl: {e}")

# Critical dependencies
try:
    import cachetools
    print("✓ cachetools")
except Exception as e:
    errors.append(f"✗ cachetools: {e}")

try:
    from PIL import Image
    print("✓ pillow (PIL)")
except Exception as e:
    print(f"  WARNING: pillow not available (expected on login node, provided by scipy-stack module)")

if errors:
    print("\nCRITICAL ERRORS:")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Python import check failed"
    exit 1
fi
echo ""

# 5. CODE FILE SYNTAX AND IMPORTS
echo "=== 5. CODE FILE SYNTAX CHECK ==="
python -m py_compile compute_score.py && echo "✓ compute_score.py syntax OK" || {
    echo "✗ compute_score.py syntax ERROR"
    exit 1
}

python -m py_compile main_ppo.py && echo "✓ main_ppo.py syntax OK" || {
    echo "✗ main_ppo.py syntax ERROR"
    exit 1
}

python -m py_compile verifier.py && echo "✓ verifier.py syntax OK" || {
    echo "✗ verifier.py syntax ERROR"
    exit 1
}
echo ""

# 6. FUNCTION SIGNATURES
echo "=== 6. FUNCTION SIGNATURES ==="
if grep -q "^def compute_score" compute_score.py; then
    echo "✓ compute_score function found"
else
    echo "✗ compute_score function MISSING"
    exit 1
fi

if grep -q "from verifier import RewardModelWorker" main_ppo.py || grep -q "import verifier" main_ppo.py; then
    echo "✓ verifier import in main_ppo.py"
else
    echo "✗ verifier import MISSING in main_ppo.py"
    exit 1
fi
echo ""

# 7. PATH EXISTENCE
echo "=== 7. PATH EXISTENCE ==="
MODEL_PATH="$HOME/scratch/macroscope/models/Qwen3-4B-Base"
VERIFIER_PATH="$HOME/scratch/macroscope/models/general-verifier"
DATASET_PATH="$HOME/scratch/macroscope/data/webinstruct-verified"
LOG_PATH="$HOME/scratch/macroscope/logs"
CHECKPOINT_PATH="$HOME/scratch/macroscope/checkpoints"

if [ -d "$MODEL_PATH" ]; then
    echo "✓ Model path exists: $MODEL_PATH"
    if [ -f "$MODEL_PATH/config.json" ] || [ -f "$MODEL_PATH/config.yaml" ]; then
        echo "  ✓ Model config file found"
    else
        echo "  WARNING: Model config file not found (may be in subdirectory)"
    fi
else
    echo "✗ Model path MISSING: $MODEL_PATH"
    exit 1
fi

if [ -d "$VERIFIER_PATH" ]; then
    echo "✓ Verifier path exists: $VERIFIER_PATH"
else
    echo "✗ Verifier path MISSING: $VERIFIER_PATH"
    exit 1
fi

if [ -d "$DATASET_PATH" ]; then
    echo "✓ Dataset path exists: $DATASET_PATH"
    if [ -f "$DATASET_PATH/train.parquet" ]; then
        echo "  ✓ train.parquet found"
    else
        echo "  ✗ train.parquet MISSING"
        exit 1
    fi
    if [ -f "$DATASET_PATH/test.parquet" ]; then
        echo "  ✓ test.parquet found"
    else
        echo "  ✗ test.parquet MISSING"
        exit 1
    fi
else
    echo "✗ Dataset path MISSING: $DATASET_PATH"
    exit 1
fi

mkdir -p "$LOG_PATH" && echo "✓ Log path exists/created: $LOG_PATH"
mkdir -p "$CHECKPOINT_PATH" && echo "✓ Checkpoint path exists/created: $CHECKPOINT_PATH"
echo ""

# 8. ENVIRONMENT VARIABLES IN SCRIPT
echo "=== 8. ENVIRONMENT VARIABLES IN train_general_reasoner.sh ==="
if grep -q "HDFS_DATA_PATH=\$HOME/scratch/macroscope/data" train_general_reasoner.sh; then
    echo "✓ HDFS_DATA_PATH set correctly"
else
    echo "✗ HDFS_DATA_PATH incorrect"
    exit 1
fi

if grep -q "HDFS_MODEL_PATH=\$HOME/scratch/macroscope/models" train_general_reasoner.sh; then
    echo "✓ HDFS_MODEL_PATH set correctly"
else
    echo "✗ HDFS_MODEL_PATH incorrect"
    exit 1
fi

if grep -q "MODEL_NAME=Qwen3-4B-Base" train_general_reasoner.sh; then
    echo "✓ MODEL_NAME set correctly"
else
    echo "✗ MODEL_NAME incorrect"
    exit 1
fi

if grep -q "VERIFIER_NAME=general-verifier" train_general_reasoner.sh; then
    echo "✓ VERIFIER_NAME set correctly"
else
    echo "✗ VERIFIER_NAME incorrect"
    exit 1
fi
echo ""

# 9. RAY RUNTIME ENV CONFIG
echo "=== 9. RAY RUNTIME ENV CONFIG ==="
if grep -q '"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"' train_general_reasoner.sh; then
    echo "✓ RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES in runtime-env"
else
    echo "✗ RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES MISSING in runtime-env"
    exit 1
fi

if grep -q '"NCCL_SOCKET_IFNAME":' train_general_reasoner.sh; then
    echo "✓ NCCL_SOCKET_IFNAME in runtime-env"
else
    echo "✗ NCCL_SOCKET_IFNAME MISSING in runtime-env"
    exit 1
fi
echo ""

# 10. HYDRA CONFIG PATH
echo "=== 10. HYDRA CONFIG PATH ==="
if grep -q "config_path='./verl/verl/trainer/config'" main_ppo.py; then
    echo "✓ Hydra config_path set"
    if [ -d "verl/verl/trainer/config" ]; then
        echo "  ✓ Config directory exists"
        if [ -f "verl/verl/trainer/config/ppo_trainer.yaml" ] || [ -f "verl/verl/trainer/config/ppo_trainer.yml" ]; then
            echo "  ✓ ppo_trainer config file found"
        else
            echo "  WARNING: ppo_trainer.yaml not found (may be in verl submodule)"
        fi
    else
        echo "  WARNING: Config directory not found (may be in verl submodule)"
    fi
else
    echo "✗ Hydra config_path incorrect"
    exit 1
fi
echo ""

# 11. VERL PATCHES
echo "=== 11. VERL PATCHES ==="
if [ -f "verl/verl/workers/fsdp_workers.py" ]; then
    if grep -q "AutoModelForVision2Seq" verl/verl/workers/fsdp_workers.py; then
        echo "✓ fsdp_workers.py has AutoModelForVision2Seq (patch may be needed)"
    else
        echo "  INFO: AutoModelForVision2Seq not found (patch may be needed)"
    fi
else
    echo "  INFO: fsdp_workers.py not found (may be in verl submodule)"
fi

# Check for vLLM MIG patch
VLLM_CUDA_PY="$HOME/envs/verl_train/lib/python3.11/site-packages/vllm/platforms/cuda.py"
if [ -f "$VLLM_CUDA_PY" ]; then
    if grep -q "MIG" "$VLLM_CUDA_PY" || grep -q "ValueError" "$VLLM_CUDA_PY"; then
        echo "✓ vLLM cuda.py may have MIG patch (check manually)"
    else
        echo "  INFO: vLLM MIG patch status unknown"
    fi
else
    echo "  INFO: vLLM cuda.py not found at expected path"
fi
echo ""

# 12. JOB STATUS
echo "=== 12. JOB STATUS ==="
if command -v squeue &> /dev/null; then
    echo "Current jobs:"
    squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "  No jobs found or squeue unavailable"
else
    echo "  squeue command not available"
fi
echo ""

# 13. MODULE VERSIONS
echo "=== 13. MODULE VERSIONS ==="
if command -v module &> /dev/null; then
    echo "Python module:"
    module show python/3.11 2>&1 | grep -E "Version|PATH" | head -3 || true
    echo ""
    echo "CUDA module:"
    module show cuda/12.6 2>&1 | grep -E "Version|PATH" | head -3 || true
else
    echo "  module command not available"
fi
echo ""

# 14. FINAL SUMMARY
echo "=========================================="
echo "VERIFICATION COMPLETE"
echo "=========================================="
echo ""
echo "✓ All critical checks passed!"
echo ""
echo "Next steps:"
echo "  1. Ensure jobs are submitted: sbatch run_training_${CLUSTER,,}.slurm"
echo "  2. Monitor with: squeue -u $USER"
echo "  3. Check logs in: $LOG_PATH"
echo ""
