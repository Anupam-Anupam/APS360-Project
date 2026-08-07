#!/bin/bash
# Same as train_general_reasoner.sh (main_ppo + same Hydra config) but 1 GPU, small batches, 1 epoch.
# Use this before the full multi-node job. Works on FIR (SCRATCH) or Nibi (HOME/scratch).

set -x
export NCCL_DEBUG=INFO
export RAY_DEDUP_LOGS=1
export RAY_rpc_timeout_seconds=3600

export PROJECT_NAME="General-Reasoner-TEST"
# Set WANDB_API_KEY in env for logging, or WANDB_MODE=disabled to skip
export WANDB_OFFICIAL=1

# Use SCRATCH on FIR, else $HOME/scratch (e.g. Nibi)
BASE_PATH="${SCRATCH:-$HOME/scratch}/macroscope"
export HDFS_DATA_PATH=$BASE_PATH/data
export HDFS_MODEL_PATH=$BASE_PATH/models
export HDFS_CHECKPOINT_PATH=$BASE_PATH/checkpoints
export HDFS_LOG_PATH=$BASE_PATH/logs

RUN_NAME=test-run-$(date +%s)

# MINIMAL TEST PARAMETERS - fast execution
TRAIN_BATCH_SIZE=8          # Tiny batch (full job uses 1024)
VAL_BATCH_SIZE=4            # Tiny validation batch
MAX_PROMPT_LENGTH=512       # Shorter prompts
MAX_RESPONSE_LENGTH=1024    # Shorter responses
LEARNING_RATE=5e-7
PPO_MINI_BATCH_SIZE=4       # Tiny mini batch
PPO_MICRO_BATCH_SIZE=1      # 1 per GPU
CLIP_RATIO=0.3
KL_LOSS_COEF=0.0001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=4 # Small
ROLLOUT_N=2                 # Only 2 rollouts (full uses 8)
KL_COEF=0.001
TOTAL_EPOCHS=1              # Just 1 epoch for testing
DATASET_NAME=webinstruct-verified
ROLLOUT_GPU_MEMORY_UTIL=0.6
ACTOR_OPTIMIZER_OFFLOAD=False
ACTOR_PARAMETER_OFFLOAD=False
MODEL_NAME=Qwen3-4B-Base
VERIFIER_NAME=general-verifier
N_GPUS_PER_NODE=1           # Single GPU test
NNODES=1                     # Single node test

CUDA_DEVS="0"
max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)

echo "=== TEST RUN CONFIGURATION ==="
echo "This is a MINIMAL test - will run fast to catch errors"
echo "Train Batch Size: $TRAIN_BATCH_SIZE (full: 1024)"
echo "Total Epochs: $TOTAL_EPOCHS (full: 30)"
echo "N GPUs: $N_GPUS_PER_NODE (full: 4)"
echo "N Nodes: $NNODES (full: 2)"
echo ""

mkdir -p $HDFS_LOG_PATH
mkdir -p $HDFS_CHECKPOINT_PATH

# Fail fast if model/data paths are missing (otherwise we only see errors in Ray job logs).
if [ ! -d "$HDFS_MODEL_PATH/$MODEL_NAME" ]; then
  echo "ERROR: Actor model path not found: $HDFS_MODEL_PATH/$MODEL_NAME"
  echo "Create it or symlink (e.g. huggingface-cli download Qwen/Qwen3-4B-Base --local-dir $HDFS_MODEL_PATH/$MODEL_NAME)"
  exit 1
fi
if [ ! -d "$HDFS_MODEL_PATH/$VERIFIER_NAME" ]; then
  echo "ERROR: Verifier model path not found: $HDFS_MODEL_PATH/$VERIFIER_NAME"
  exit 1
fi
if [ ! -f "$HDFS_DATA_PATH/$DATASET_NAME/train.parquet" ]; then
  echo "ERROR: Train data not found: $HDFS_DATA_PATH/$DATASET_NAME/train.parquet"
  exit 1
fi

# Get HEAD_IP and WORKING_DIR from environment (set by slurm script)
HEAD_IP=${HEAD_IP:-$(hostname -I | awk '{print $1}')}
WORKING_DIR=${WORKING_DIR:-$(pwd)}

# Force Ray workers to use only CUDA 12.6 (no 12.9.x) so torch loads correctly.
# Build from module vars; if unset (e.g. not run from Slurm), fall back to current LD_LIBRARY_PATH.
# Use VIRTUAL_ENV so scratch venv works; no other CUDA path so 12.9.x is never loaded.
if [ -n "${EBROOTCUDA}" ] && [ -n "${EBROOTCUDNN}" ] && [ -n "${EBROOTNCCL}" ] && [ -n "${VIRTUAL_ENV}" ]; then
  RAY_LD_LIBRARY_PATH="${EBROOTCUDNN}/lib64:${EBROOTCUDA}/lib64:${EBROOTCUDA}/extras/CUPTI/lib64:${EBROOTNCCL}/lib:${VIRTUAL_ENV}/lib/python3.11/site-packages/nvidia/cusparselt/lib"
else
  RAY_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
fi

# Submit job and capture output so we can parse job id and wait for completion.
# Without waiting, the script would exit and run_test_training_fir.slurm would run
# "ray stop", killing the cluster before the training job runs.
SUBMIT_OUT=$(HYDRA_FULL_ERROR=1 ray job submit --address=${HEAD_IP}:6379 \
    --entrypoint-num-cpus=1 \
    --runtime-env-json='{
         "working_dir": "'${WORKING_DIR}'",
         "excludes": [".git", "__pycache__", "*.pyc", "logs", "evaluation", "tinker_scripts", "assets", "merged_model", "output-*.json", "downloaded_lora", "downloaded_lora_4b"],
         "env_vars": {
           "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
           "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
           "NCCL_DEBUG": "INFO",
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
    actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    critic.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
    critic.ppo_micro_batch_size_per_gpu=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=999 \
    trainer.test_freq=999 \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS 2>&1)
echo "$SUBMIT_OUT" | tee $HDFS_LOG_PATH/$RUN_NAME.log

# Parse job id and wait for the job to finish so we don't ray stop before training runs.
# Stream the job's stdout/stderr (model loading, training, errors) into our log so we can see why models might not load.
RAY_JOB_ID=$(echo "$SUBMIT_OUT" | sed -n "s/.*Job '\\([^']*\\)' submitted successfully.*/\\1/p" | head -1)
if [ -n "$RAY_JOB_ID" ]; then
  echo "Waiting for Ray job $RAY_JOB_ID to finish (job logs stream below)..."
  export RAY_ADDRESS="http://${HEAD_IP}:8265"
  ray job logs -f "$RAY_JOB_ID" 2>&1 | tee -a $HDFS_LOG_PATH/$RUN_NAME.log &
  LOG_PID=$!
  while true; do
    STATUS_OUT=$(ray job status "$RAY_JOB_ID" 2>/dev/null || true)
    echo "$STATUS_OUT" | grep -qE 'SUCCEEDED|FAILED|STOPPED' && break
    sleep 5
  done
  kill $LOG_PID 2>/dev/null || true
else
  echo "Could not parse Ray job id from submit output; not waiting."
fi

echo ""
echo "=== TEST COMPLETE ==="
echo "Check the log: $HDFS_LOG_PATH/$RUN_NAME.log"
echo "If this passed, your full job should work too!"
