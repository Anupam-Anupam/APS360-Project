# Math evaluation suite for APS360 (Qwen3-1.7B)

Serve your merged model, then run math evals:

```bash
vllm serve ./merged-qwen3-1p7b-math-grpo --served-model-name aps360-qwen3-1.7b-math-grpo

python -m evaluation.simple_evals.run_simple_evals_qwen \
  --model aps360-qwen3-1.7b-math-grpo \
  --evals gsm8k,math,aime24,amc
```

Baseline:

```bash
python -m evaluation.simple_evals.run_simple_evals_qwen --model Qwen3-1.7B-Base --evals gsm8k,math
```
