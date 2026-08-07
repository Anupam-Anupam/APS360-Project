# APS360 — Domain-Specific GRPO for Mathematical Reasoning (Qwen3-1.7B)

Zero-RL training of **`Qwen/Qwen3-1.7B`** with **Group Relative Policy Optimization (GRPO)** on a **mathematics-only** slice of [WebInstruct-verified](https://huggingface.co/datasets/TIGER-Lab/WebInstruct-verified).

Rewards come from an LLM-as-a-judge verifier ([TIGER-Lab/general-verifier](https://huggingface.co/TIGER-Lab/general-verifier)). Policy updates run via the [Tinker](https://thinkingmachines.ai) API; local GPUs host the verifier (vLLM).

**Research question:** Does a math-only GRPO signal beat broader cross-domain RL on math benchmarks (GSM8K, MATH, AIME, MMLU-Pro), and how much out-of-domain ability is lost?

## Repo layout

```
scripts/
  train_grpo_qwen3.py     # main GRPO training loop (Qwen3-1.7B)
  prepare_math_data.py    # optional: export math parquet
  merge_lora.py           # merge Tinker LoRA into base for local eval
evaluation/
  eval_mmlupro.py         # MMLU-Pro via vLLM
  simple_evals/           # GSM8K, MATH, AIME, AMC, Minerva, Olympiad
slurm/
  run_train_qwen3_1p7b.slurm
requirements.txt
```

## Setup

```bash
# 1) Install tinker-cookbook (provides tinker_cookbook helpers)
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
pip install -e tinker-cookbook

# 2) Project deps (pins vLLM to Table 1)
pip install -r requirements.txt

# 3) Secrets
export TINKER_API_KEY="tml-..."
# optional
export WANDB_API_KEY="..."
```

## Train

```bash
python scripts/train_grpo_qwen3.py \
  log_path=./log_qwen3_1p7b_math_grpo \
  model_name=Qwen/Qwen3-1.7B \
  math_only=True
```

Or submit (1 node × 4 GPUs for the verifier):

```bash
sbatch slurm/run_train_qwen3_1p7b.slurm
```

### Default training hyperparameters (Table 1)

| Parameter | Value |
|-----------|--------|
| `n_nodes` | 1 |
| `n_gpu` | 4 |
| `backbone_hgf_id` | `Qwen/Qwen3-1.7B` |
| `vllm_version` | 0.25.1 |
| `train_batch_size` | 128 |
| `max_prompt_length` | 384 |
| `max_response_length` | 128 |
| `learning_rate` | 5e-7 |
| `ppo_micro_batch_size_per_gpu` | 4 |
| `ppo_mini_batch_size` | 64 |
| `clip_ratio_low` / `clip_ratio_high` | 0.3 / 0.3 (Tinker thresholds 0.7 / 1.3) |
| `temperature` | 1.0 |
| `rollout_n` | 8 |
| `kl_coeff` | 0.01 |
| Dataset | WebInstruct-verified, **math-only filter** |
| LoRA rank | 32 |
| Loss | PPO + group-relative advantages + KL-vs-base on advantages |

Checkpoints are recorded in `log_*/checkpoints.jsonl` as `tinker://…` paths. Download the final sampler weights with the Tinker CLI, then merge:

```bash
python scripts/merge_lora.py \
  --base Qwen/Qwen3-1.7B \
  --adapter /path/to/sampler_weights_final \
  --out ./merged-qwen3-1.7b-math-grpo
```

## Evaluate

**MMLU-Pro (vLLM):**

```bash
python -m evaluation.eval_mmlupro \
  --model_path ./merged-qwen3-1.7b-math-grpo \
  --tensor_parallel_size 4 \
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
