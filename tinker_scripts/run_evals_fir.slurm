#!/bin/bash
#SBATCH --job-name=eval-mmlupro
#SBATCH --account=def-zhijing_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

# Load modules
module load python/3.11 cuda/12.2 gcc arrow

# Activate virtual environment
source ~/envs/macroscope/bin/activate

echo "=== Environment Info ==="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

cd $SCRATCH/MaCroScope

MODEL_PATH="./merged_model_qw3_8b"

echo "=== Running MMLU-Pro Evaluation ==="
# Patch tensor_parallel_size to 1 for single GPU
sed -i 's/tensor_parallel_size=4/tensor_parallel_size=1/g' evaluation/eval_mmlupro.py

python -m evaluation.eval_mmlupro \
    --model_path $MODEL_PATH \
    --output_file output-mmlupro.json

echo "=== Job Complete ==="
date
