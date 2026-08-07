#!/bin/bash
# Run all evaluations on the merged model
# Usage: bash tinker_scripts/run_all_evals.sh <model_path>

set -e

MODEL_PATH=${1:-"./merged_model"}

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    echo "Usage: bash tinker_scripts/run_all_evals.sh <model_path>"
    exit 1
fi

echo "=========================================="
echo "Running All Evaluations"
echo "Model: $MODEL_PATH"
echo "=========================================="

# Direct evaluations (MMLU-Pro, SuperGPQA, BBEH)
echo ""
echo "1. Running MMLU-Pro evaluation..."
python -m evaluation.eval_mmlupro \
    --model_path "$MODEL_PATH" \
    --output_file "output-mmlupro-$(basename $MODEL_PATH).json"

echo ""
echo "2. Running SuperGPQA evaluation..."
python -m evaluation.eval_supergpqa \
    --model_path "$MODEL_PATH" \
    --output_file "output-supergpqa-$(basename $MODEL_PATH).json"

echo ""
echo "3. Running BBEH evaluation..."
python -m evaluation.eval_bbeh \
    --model_path "$MODEL_PATH" \
    --output_file "output-bbeh-$(basename $MODEL_PATH).json"

echo ""
echo "=========================================="
echo "Direct Evaluations Complete!"
echo "=========================================="
echo ""
echo "For math-related tasks (MATH, Olympiad, Minerva, GSM8K, AMC, AIME):"
echo "1. Start vLLM server:"
echo "   vllm serve $MODEL_PATH --tensor-parallel-size 1"
echo ""
echo "2. In another terminal, run:"
echo "   python -m evaluation.simple-evals.run_simple_evals_qwen --model <model_name>"
echo ""
echo "Results saved to:"
echo "  - output-mmlupro-$(basename $MODEL_PATH).json"
echo "  - output-supergpqa-$(basename $MODEL_PATH).json"
echo "  - output-bbeh-$(basename $MODEL_PATH).json"
