# APS360 — Domain-Specific GRPO for Mathematical Reasoning (Qwen3-1.7B)

Zero-RL training of **`Qwen/Qwen3-1.7B-Base`** with **Group Relative Policy Optimization (GRPO)** on a **mathematics-only** slice of [WebInstruct-verified](https://huggingface.co/datasets/TIGER-Lab/WebInstruct-verified).

Rewards come from an LLM-as-a-judge verifier ([TIGER-Lab/general-verifier](https://huggingface.co/TIGER-Lab/general-verifier)). Policy updates run via the [Tinker](https://thinkingmachines.ai) API; one local GPU hosts the verifier.

**Research question:** Does a math-only GRPO signal beat broader cross-domain RL on math benchmarks (GSM8K, MATH, AIME, MMLU-Pro math), and how much out-of-domain ability (e.g. GPQA) is lost?

## Repo layout

```
scripts/
  train_grpo_qwen.py      # main GRPO training loop (Qwen3-1.7B)
  merge_lora.py           # merge Tinker LoRA into base for local eval
evaluation/
  eval_mmlupro.py         # MMLU-Pro via vLLM
  simple_evals/           # GSM8K, MATH, AIME, AMC, Minerva, Olympiad, GPQA
slurm/
  run_train_qwen3_1p7b.slurm
requirements.txt
```

## Setup

```bash
# 1) Install tinker-cookbook (provides tinker_cookbook helpers)
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
pip install -e tinker-cookbook

# 2) Project deps
pip install -r requirements.txt

# 3) Secrets
export TINKER_API_KEY="tml-..."
# optional
export WANDB_API_KEY="..."
```

## Train

```bash
python scripts/train_grpo_qwen.py \
  log_path=./log_qwen3_1p7b_math_grpo \
  model_name=Qwen/Qwen3-1.7B-Base \
  math_only=True \
  batch_size=128 \
  group_size=8 \
  learning_rate=4e-5 \
  total_epochs=1
```

Or submit:

```bash
sbatch slurm/run_train_qwen3_1p7b.slurm
```

### Default training hyperparameters

| Setting | Value |
|---------|--------|
| Base model | `Qwen/Qwen3-1.7B-Base` |
| Dataset | WebInstruct-verified, **math-only filter** |
| LoRA rank | 32 |
| Batch size | 128 questions / step |
| Group size | 8 rollouts / question |
| Learning rate | 4e-5 |
| Max prompt / response | 1024 / 4096 tokens |
| Loss | importance sampling (GRPO-style advantages) |
| Verifier | `TIGER-Lab/general-verifier` |

Checkpoints are recorded in `log_*/checkpoints.jsonl` as `tinker://…` paths. Download the final sampler weights with the Tinker CLI, then merge:

```bash
python scripts/merge_lora.py \
  --base Qwen/Qwen3-1.7B-Base \
  --adapter /path/to/sampler_weights_final \
  --out ./merged-qwen3-1.7b-math-grpo
```

## Evaluate

**MMLU-Pro (vLLM):**

```bash
python -m evaluation.eval_mmlupro \
  --model_path ./merged-qwen3-1.7b-math-grpo \
  --tensor_parallel_size 1 \
  --output_file outputs-mmlupro.json
```

**Math suite (served chat API):**

```bash
# Serve your merged model, then:
python -m evaluation.simple_evals.run_simple_evals_qwen \
  --model-path ./merged-qwen3-1.7b-math-grpo \
  --evals gsm8k,math,aime24,amc
```

## Acknowledgements

- Training loop adapted from [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- Verifier protocol from [General-Reasoner](https://github.com/TIGER-AI-Lab/General-Reasoner)
- Math evals adapted from [simple_evals](https://github.com/openai/simple_evals) / [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)
