#!/bin/bash
# Single-GPU version of train_general_reasoner.sh
# Same algorithm and hyperparameter structure, tuned for 1x H100 80GB.
#
# Key differences from the multi-GPU script:
#   - trainer.n_gpus_per_node=1, trainer.nnodes=1
#   - actor_rollout_ref.rollout.tensor_model_parallel_size=1 (REQUIRED for 1 GPU)
#   - Batch sizes scaled down to fit single GPU memory
#   - ACTOR_OPTIMIZER_OFFLOAD defaults to True (saves ~28 GB GPU memory for 7B model)
#   - rollout.enforce_eager=True, rollout.free_cache_engine=True (better 1-GPU memory management)
#   - ROLLOUT_GPU_MEMORY_UTIL=0.4 (leaves room for FSDP training pass)
#   - Does NOT use Ray job submit against a remote cluster; wraps around a local Ray node.
#     The calling slurm script (run_training_1gpu_fir.slurm) starts ray head and exports HEAD_IP.

set -x
export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1
export RAY_rpc_timeout_seconds=3600

# ── Paths ──────────────────────────────────────────────────────────────────────
# Use $SCRATCH on Alliance clusters; fall back to $HOME/scratch (e.g. Nibi).
BASE_PATH="${SCRATCH:-$HOME/scratch}/macroscope"
export HDFS_DATA_PATH=$BASE_PATH/data
export HDFS_MODEL_PATH=$BASE_PATH/models
export HDFS_CHECKPOINT_PATH=$BASE_PATH/checkpoints
export HDFS_LOG_PATH=$BASE_PATH/logs

export PROJECT_NAME=${PROJECT_NAME:-"General-Reasoner-1GPU"}
# Set WANDB_API_KEY in the environment, or pass WANDB_MODE=disabled to skip.
export WANDB_OFFICIAL=1

if [ -z "$RUN_NAME" ]; then
    RUN_NAME=general-reasoner-1gpu
fi

# ── Default hyperparameters (scaled for 1x H100 80GB) ─────────────────────────
# Training batch: 64 prompts × 8 rollouts = 512 sequences per update step.
# PPO mini-batch: 32 sequences → 16 micro-steps with ppo_micro_batch=1.
# H100 80GB can fit Qwen2.5-7B in bf16 (≈14 GB weights + 14 GB grads) while
# leaving 40 GB for the vLLM KV cache when optimizer states are on CPU.
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=16
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=4096          # Reduced from 8192; saves ~16 GB KV cache on 1 GPU
LEARNING_RATE=5e-7
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE=1            # 1 sequence per GPU micro-step (most memory-safe)
CLIP_RATIO=0.3
KL_LOSS_COEF=0.0001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=8       # Keep small to avoid OOM during log-prob computation
ROLLOUT_N=8                       # Same as production (8 rollouts per prompt)
KL_COEF=0.001
TOTAL_EPOCHS=30
DATASET_NAME=webinstruct-verified
ROLLOUT_GPU_MEMORY_UTIL=0.4       # 0.4×80 GB = 32 GB for vLLM KV cache; rest for FSDP
ACTOR_OPTIMIZER_OFFLOAD=True      # Offload Adam states to CPU (~28 GB freed on GPU)
ACTOR_PARAMETER_OFFLOAD=False     # Keep weights on GPU for faster training
MODEL_NAME=Qwen2.5-7B
VERIFIER_NAME=general-verifier
N_GPUS_PER_NODE=1
NNODES=1

# ── Suffix/run-name generation (mirrors train_general_reasoner.sh) ─────────────
generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local model_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --val_batch_size) suffix+="_valbatch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_max_prompt$2"; shift 2 ;;
      --max_response_length) suffix+="_max_response$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --ppo_micro_batch_size) shift 2 ;;
      --kl_loss_coef) suffix+="_klcoef$2"; shift 2 ;;
      --entropy_coeffient) suffix+="_entcoef$2"; shift 2 ;;
      --clip_ratio) suffix+="_clipratio$2"; shift 2 ;;
      --kl_loss_type) suffix+="_kltype$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --log_prob_micro_batch_size) suffix+="_logprobbatch$2"; shift 2 ;;
      --rollout_n) suffix+="_rollout$2"; shift 2 ;;
      --kl_coef) suffix+="_klcontrol$2"; shift 2 ;;
      --total_epochs) suffix+="_epochs$2"; shift 2 ;;
      --rollout_gpu_memory_util) shift 2 ;;
      --actor_optimizer_offload) shift 2 ;;
      --actor_parameter_offload) shift 2 ;;
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --model_name) suffix+="_$2"; model_provided=true; shift 2 ;;
      *) shift ;;
    esac
  done

  if [ "$dataset_provided" = false ]; then
    suffix+="_$DATASET_NAME"
  fi
  if [ "$model_provided" = false ]; then
    suffix+="_$MODEL_NAME"
  fi

  echo "$suffix"
}

echo "Arguments received: $@"
SUFFIX=$(generate_suffix "$@")
RUN_NAME="$RUN_NAME$SUFFIX"
LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME.log"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --val_batch_size) VAL_BATCH_SIZE="$2"; shift 2 ;;
    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --ppo_micro_batch_size) PPO_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --entropy_coeffient) ENTROPY_COEFFIENT="$2"; shift 2 ;;
    --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
    --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --log_prob_micro_batch_size) LOG_PROB_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --kl_coef) KL_COEF="$2"; shift 2 ;;
    --actor_optimizer_offload) ACTOR_OPTIMIZER_OFFLOAD="$2"; shift 2 ;;
    --actor_parameter_offload) ACTOR_PARAMETER_OFFLOAD="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# ── Pre-flight path checks ────────────────────────────────────────────────────
if [ ! -d "$HDFS_MODEL_PATH/$MODEL_NAME" ]; then
  echo "ERROR: Actor model not found: $HDFS_MODEL_PATH/$MODEL_NAME"
  echo "  huggingface-cli download Qwen/Qwen2.5-7B --local-dir $HDFS_MODEL_PATH/$MODEL_NAME"
  exit 1
fi
if [ ! -d "$HDFS_MODEL_PATH/$VERIFIER_NAME" ]; then
  echo "ERROR: Verifier model not found: $HDFS_MODEL_PATH/$VERIFIER_NAME"
  echo "  huggingface-cli download TIGER-Lab/general-verifier --local-dir $HDFS_MODEL_PATH/$VERIFIER_NAME"
  exit 1
fi
if [ ! -f "$HDFS_DATA_PATH/$DATASET_NAME/train.parquet" ]; then
  echo "ERROR: Train data not found: $HDFS_DATA_PATH/$DATASET_NAME/train.parquet"
  echo "  python data_preprocess.py --local-dir $HDFS_DATA_PATH/$DATASET_NAME"
  exit 1
fi

mkdir -p "$HDFS_LOG_PATH"
mkdir -p "$HDFS_CHECKPOINT_PATH"

# ── Echo config ───────────────────────────────────────────────────────────────
echo "Training with the following parameters (SINGLE GPU):"
echo "  Train Batch Size:         $TRAIN_BATCH_SIZE (prompts per step)"
echo "  Val Batch Size:           $VAL_BATCH_SIZE"
echo "  Max Prompt Length:        $MAX_PROMPT_LENGTH"
echo "  Max Response Length:      $MAX_RESPONSE_LENGTH"
echo "  Learning Rate:            $LEARNING_RATE"
echo "  PPO Mini Batch Size:      $PPO_MINI_BATCH_SIZE"
echo "  PPO Micro Batch Size:     $PPO_MICRO_BATCH_SIZE"
echo "  KL Loss Coef:             $KL_LOSS_COEF"
echo "  KL Loss Type:             $KL_LOSS_TYPE"
echo "  Temperature:              $TEMPERATURE"
echo "  Rollout N:                $ROLLOUT_N"
echo "  KL Coef:                  $KL_COEF"
echo "  Total Epochs:             $TOTAL_EPOCHS"
echo "  Dataset:                  $DATASET_NAME"
echo "  Model:                    $MODEL_NAME"
echo "  Verifier:                 $VERIFIER_NAME"
echo "  Optimizer Offload:        $ACTOR_OPTIMIZER_OFFLOAD"
echo "  Parameter Offload:        $ACTOR_PARAMETER_OFFLOAD"
echo "  Rollout GPU Mem Util:     $ROLLOUT_GPU_MEMORY_UTIL"
echo "  n_gpus_per_node:          $N_GPUS_PER_NODE"
echo "  nnodes:                   $NNODES"
echo "  Run Name:                 $RUN_NAME"
echo "  Log File:                 $LOG_FILE_PATH"

max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)

# ── Build LD_LIBRARY_PATH for Ray workers ────────────────────────────────────
# Must be explicit (no appending) to prevent CUDA 12.9.x sneaking in.
if [ -n "${EBROOTCUDA}" ] && [ -n "${EBROOTCUDNN}" ] && [ -n "${EBROOTNCCL}" ]; then
  RAY_LD_LIBRARY_PATH="${EBROOTCUDNN}/lib64:${EBROOTCUDA}/lib64:${EBROOTCUDA}/extras/CUPTI/lib64:${EBROOTNCCL}/lib:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cusparselt/lib"
else
  RAY_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
fi

# HEAD_IP and WORKING_DIR are set by the calling slurm script.
HEAD_IP=${HEAD_IP:-$(hostname -I | awk '{print $1}')}
WORKING_DIR=${WORKING_DIR:-$(pwd)}

# ── Submit Ray job and wait for completion ────────────────────────────────────
SUBMIT_OUT=$(HYDRA_FULL_ERROR=1 ray job submit --address=${HEAD_IP}:6379 \
    --entrypoint-num-cpus=1 \
    --runtime-env-json='{
         "working_dir": "'${WORKING_DIR}'",
         "excludes": [".git", "__pycache__", "*.pyc", "logs", "evaluation", "tinker_scripts", "assets"],
         "env_vars": {
           "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
           "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
           "NCCL_DEBUG": "INFO",
           "NCCL_SOCKET_IFNAME": "^lo,docker0",
           "LD_LIBRARY_PATH": "'"${RAY_LD_LIBRARY_PATH}"'"
         }
      }' \
    -- python -m main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=./compute_score.py \
    custom_reward_function.name=compute_score \
    reward_model.enable=True \
    reward_model.model.path=$HDFS_MODEL_PATH/$VERIFIER_NAME \
    reward_model.strategy=verifier \
    reward_model.reward_manager=naive \
    reward_model.micro_batch_size=0 \
    data.train_files=[$HDFS_DATA_PATH/$DATASET_NAME/train.parquet] \
    data.val_files=[$HDFS_DATA_PATH/$DATASET_NAME/test.parquet] \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.clip_ratio_c=10 \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAMETER_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    critic.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
    critic.ppo_micro_batch_size_per_gpu=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS 2>&1)

echo "$SUBMIT_OUT" | tee "$LOG_FILE_PATH"

# Wait for the Ray job to finish so the slurm script does not call ray stop early.
RAY_JOB_ID=$(echo "$SUBMIT_OUT" | sed -n "s/.*Job '\\([^']*\\)' submitted successfully.*/\\1/p" | head -1)
if [ -n "$RAY_JOB_ID" ]; then
  echo "Waiting for Ray job $RAY_JOB_ID ..."
  export RAY_ADDRESS="http://${HEAD_IP}:8265"
  ray job logs -f "$RAY_JOB_ID" 2>&1 | tee -a "$LOG_FILE_PATH" &
  LOG_PID=$!
  while true; do
    STATUS_OUT=$(ray job status "$RAY_JOB_ID" 2>/dev/null || true)
    echo "$STATUS_OUT" | grep -qE 'SUCCEEDED|FAILED|STOPPED' && break
    sleep 10
  done
  kill $LOG_PID 2>/dev/null || true
  echo ""
  echo "Final job status:"
  ray job status "$RAY_JOB_ID" 2>/dev/null || true
else
  echo "WARNING: Could not parse Ray job ID from submit output."
  echo "Training may still be running. Check Ray dashboard at http://${HEAD_IP}:8265"
fi

echo ""
echo "=== TRAINING COMPLETE ==="
echo "Log: $LOG_FILE_PATH"
echo "Checkpoints: $HDFS_CHECKPOINT_PATH/$RUN_NAME"
