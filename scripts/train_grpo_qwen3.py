"""
APS360 — GRPO (Zero-RL) training for Qwen/Qwen3-1.7B on mathematics-only data.

Uses the Tinker API for LoRA updates and a local vLLM verifier
(TIGER-Lab/general-verifier) for LLM-as-a-judge rewards.

Hyperparameters align with the APS360 training settings table
(n_nodes=1, n_gpu=4, train_batch_size=128, max_prompt=384,
max_response=128, lr=5e-7, rollout_n=8, clip_ratio=0.3, kl=0.01).

Adapted from tinker-cookbook rl_loop.py and the General-Reasoner verifier protocol.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import Future

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

VERIFIER_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)
VERIFIER_PASS_TAG = "Final Decision: Yes"

# Keep WebInstruct rows whose category looks mathematical.
MATH_CATEGORY_KEYWORDS = (
    "math",
    "mathematics",
    "algebra",
    "geometry",
    "calculus",
    "statistics",
    "probability",
    "arithmetic",
    "number theory",
    "combinatorics",
)


def extract_last_boxed(text: str) -> str | None:
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str) -> str | None:
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    ]
    last_match = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()
    for stop_word in ("</s>", "<|im_end|>", "<|endoftext|>"):
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[: -len(stop_word)].strip()
    return last_match


def extract_solution(solution_str: str) -> str | None:
    boxed = extract_last_boxed(solution_str)
    if boxed:
        return boxed
    return extract_last_final_answer(solution_str)


def is_math_example(example: dict) -> bool:
    """Filter WebInstruct-verified down to mathematics-only questions."""
    for key in ("category", "topic", "subject", "domain", "field"):
        val = example.get(key)
        if val is None:
            continue
        text = str(val).lower()
        if any(k in text for k in MATH_CATEGORY_KEYWORDS):
            return True
    # Fallback: some releases only tag difficulty; keep if question looks math-heavy
    q = str(example.get("question", "")).lower()
    math_markers = ("\\frac", "\\sum", "equation", "prove", "calculate", "integral", "derivative")
    return any(m in q for m in math_markers)


class GeneralVerifier:
    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.7,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=2048)

    def _truncate_response(self, response: str) -> str:
        if response is None:
            return ""
        return self.tokenizer.decode(self.tokenizer.encode(response)[-1024:])

    def verify_batch(
        self, questions: list[str], ground_truths: list[str], responses: list[str]
    ) -> list[float]:
        student_answers = [extract_solution(response) for response in responses]
        ground_truths = [self._truncate_response(g) for g in ground_truths]
        student_answers = [self._truncate_response(s or "") for s in student_answers]
        messages = [
            VERIFIER_PROMPT_TEMPLATE.format(
                question=q, ground_truth=g, student_answer=s
            )
            for q, g, s in zip(questions, ground_truths, student_answers)
        ]
        outputs = self.llm.generate(messages, sampling_params=self.sampling_params)
        verifier_responses = [o.outputs[0].text.strip() for o in outputs]
        rewards: list[float] = []
        for vr, gt, sa in zip(verifier_responses, ground_truths, student_answers):
            try:
                if VERIFIER_PASS_TAG in vr:
                    sa_len = len(self.tokenizer.encode(sa))
                    gt_len = len(self.tokenizer.encode(gt))
                    difference = min(abs(sa_len - gt_len), 10)
                    rewards.append(1.0 - difference * 0.05)
                else:
                    rewards.append(0.0)
            except Exception as e:
                logger.warning(f"Verifier error: {e}")
                rewards.append(0.0)
        return rewards


def incorporate_kl_penalty_sync(
    data: list[types.Datum],
    base_sampling_client: tinker.SamplingClient,
    kl_penalty_coef: float,
) -> dict[str, float]:
    """Adjust advantages in-place with KL vs base model (sync Tinker API)."""
    full_seqs = [
        datum.model_input.append_int(int(datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data
    ]
    base_futures = [base_sampling_client.compute_logprobs(seq) for seq in full_seqs]
    base_logprobs = [fut.result() for fut in base_futures]

    sampled_logprobs = [d.loss_fn_inputs["logprobs"].to_torch() for d in data]
    masks = [d.loss_fn_inputs["mask"].to_torch().float() for d in data]
    logprob_diffs = []
    adj_masks = []
    for base_lp, sampled_lp, mask in zip(base_logprobs, sampled_logprobs, masks):
        # base_lp[0] is None (no prev token); align with target positions via [1:]
        base_t = torch.tensor(
            [0.0 if x is None else float(x) for x in base_lp[1:]], dtype=torch.float32
        )
        n = min(len(base_t), len(sampled_lp), len(mask))
        base_t = base_t[:n]
        sampled_lp = sampled_lp[:n]
        mask = mask[:n]
        logprob_diffs.append((sampled_lp - base_t) * mask)
        adj_masks.append(mask)

    mask_sum = sum(float(m.sum()) for m in adj_masks)
    if mask_sum <= 0:
        return {"kl_policy_base": 0.0}
    avg_logp_diff = sum(float(d.sum()) for d in logprob_diffs) / mask_sum
    for i, datum in enumerate(data):
        adv = datum.loss_fn_inputs["advantages"].to_torch()
        n = len(adj_masks[i])
        kl_adv = kl_penalty_coef * adj_masks[i] * (avg_logp_diff - logprob_diffs[i])
        if len(adv) != n:
            adv = adv[:n]
            kl_adv = kl_adv[: min(len(kl_adv), n)]
        datum.loss_fn_inputs["advantages"] = TensorData.from_torch(adv + kl_adv)
    return {"kl_policy_base": float(avg_logp_diff)}


@chz.chz
class Config:
    # Matches APS360 Table 1 training settings
    base_url: str | None = None
    log_path: str = "./log_qwen3_1p7b_math_grpo"
    model_name: str = "Qwen/Qwen3-1.7B"  # backbone_hgf_id
    dataset_path: str = "TIGER-Lab/WebInstruct-verified"
    math_only: bool = True
    n_nodes: int = 1
    n_gpu: int = 4
    train_batch_size: int = 128  # alias used as batch_size
    batch_size: int = 128
    max_prompt_length: int = 384
    max_response_length: int = 128
    max_tokens: int = 128  # = max_response_length
    learning_rate: float = 5e-7
    ppo_micro_batch_size_per_gpu: int = 4
    ppo_mini_batch_size: int = 64
    clip_ratio_low: float = 0.3
    clip_ratio_high: float = 0.3
    temperature: float = 1.0
    rollout_n: int = 8
    group_size: int = 8  # = rollout_n
    kl_coeff: float = 0.01
    lora_rank: int = 32
    save_every: int = 20
    total_epochs: int = 1
    verifier_name: str = "TIGER-Lab/general-verifier"
    wandb_project: str | None = "APS360-Qwen3-GRPO"
    wandb_name: str | None = "qwen3-1.7b-math-grpo"


def main(config: Config):
    import os as _os

    wandb_key = _os.environ.get("WANDB_API_KEY", "")
    use_wandb = bool(wandb_key) and wandb_key != "REPLACE_ME"
    if not use_wandb:
        _os.environ.pop("WANDB_API_KEY", None)

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project if use_wandb else None,
        wandb_name=config.wandb_name if use_wandb else None,
        config=config,
        do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    verifier = GeneralVerifier(
        config.verifier_name, tensor_parallel_size=max(1, config.n_gpu)
    )

    logger.info(f"Loading dataset: {config.dataset_path}")
    dataset = datasets.load_dataset(config.dataset_path)
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]
    if config.math_only:
        pre = len(train_dataset)
        train_dataset = train_dataset.filter(is_math_example)
        logger.info(f"Math-only filter: {pre} -> {len(train_dataset)} examples")

    def _filter_overlong(row):
        message = [
            {
                "role": "user",
                "content": row["question"]
                + " Please reason step by step, and put your final answer within \\boxed{}.",
            }
        ]
        n_tokens = len(renderer.build_generation_prompt(message).to_ints())
        return n_tokens <= config.max_prompt_length

    pre = len(train_dataset)
    train_dataset = train_dataset.filter(_filter_overlong)
    logger.info(
        f"Filtered overlong prompts: {pre} -> {len(train_dataset)} "
        f"(max_prompt_length={config.max_prompt_length})"
    )

    batch_size = config.train_batch_size
    group_size = config.rollout_n
    max_tokens = config.max_response_length

    n_batches_per_epoch = max(1, len(train_dataset) // batch_size)
    n_train_batches = n_batches_per_epoch * config.total_epochs

    service_client = tinker.ServiceClient(base_url=config.base_url)
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    # Table aliases already resolved above as batch_size / group_size / max_tokens
    sampling_params = tinker.types.SamplingParams(
        max_tokens=max_tokens,
        temperature=config.temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # PPO clip thresholds from clip_ratio ε (thresholds = 1 ± ε)
    clip_low_threshold = 1.0 - config.clip_ratio_low
    clip_high_threshold = 1.0 + config.clip_ratio_high
    loss_fn_config = {
        "clip_low_threshold": clip_low_threshold,
        "clip_high_threshold": clip_high_threshold,
    }

    # Base-model sampler for KL-vs-reference advantage adjustment (kl_coeff)
    base_sampling_client = (
        service_client.create_sampling_client(base_model=config.model_name)
        if config.kl_coeff > 0
        else None
    )

    logger.info(
        f"Training {config.model_name} for {n_train_batches} steps "
        f"({config.total_epochs} epoch(s) x {n_batches_per_epoch} batches); "
        f"train_rows={len(train_dataset)}, batch_size={batch_size}, "
        f"rollout_n={group_size}, lr={config.learning_rate}, "
        f"max_prompt={config.max_prompt_length}, max_response={max_tokens}, "
        f"clip=[{clip_low_threshold:.2f},{clip_high_threshold:.2f}], "
        f"kl_coeff={config.kl_coeff}, ppo_mini_batch={config.ppo_mini_batch_size}, "
        f"n_gpu={config.n_gpu}"
    )

    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        step = batch_idx
        within_epoch = batch_idx % n_batches_per_epoch
        metrics: dict[str, float] = {
            "progress/batch": float(batch_idx),
            "progress/epoch": float(batch_idx // n_batches_per_epoch),
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        if step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        batch_start = within_epoch * batch_size
        batch_end = min((within_epoch + 1) * batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        sampling_path = (
            training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
        )
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        batch_futures: list[list[Future]] = []
        batch_prompts: list[list[int]] = []
        for question in batch_rows["question"]:
            message = [
                {
                    "role": "user",
                    "content": question
                    + " Please reason step by step, and put your final answer within \\boxed{}.",
                }
            ]
            model_input = renderer.build_generation_prompt(message)
            prompt_tokens = model_input.to_ints()
            futures = [
                sampling_client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                )
                for _ in range(group_size)
            ]
            batch_futures.append(futures)
            batch_prompts.append(prompt_tokens)

        all_questions: list[str] = []
        all_answers: list[str] = []
        all_responses: list[str] = []
        all_metadata: list[dict] = []

        for futures, prompt_tokens, question, answer in zip(
            batch_futures, batch_prompts, batch_rows["question"], batch_rows["answer"]
        ):
            group_tokens: list[list[int]] = []
            group_logprobs: list[list[float]] = []
            group_ob_lens: list[int] = []
            for future in futures:
                sample_result = future.result()
                sampled_tokens = sample_result.sequences[0].tokens
                sampled_logprobs = sample_result.sequences[0].logprobs
                assert sampled_logprobs is not None
                group_tokens.append(prompt_tokens + sampled_tokens)
                group_ob_lens.append(len(prompt_tokens) - 1)
                group_logprobs.append(sampled_logprobs)
                parsed_message, _ = renderer.parse_response(sampled_tokens)
                response_content = parsed_message["content"]
                all_questions.append(question)
                all_answers.append(answer)
                all_responses.append(response_content)
            all_metadata.append(
                {
                    "group_tokens": group_tokens,
                    "group_logprobs": group_logprobs,
                    "group_ob_lens": group_ob_lens,
                    "group_size": len(futures),
                }
            )

        all_rewards = verifier.verify_batch(all_questions, all_answers, all_responses)

        training_datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        reward_idx = 0
        for metadata in all_metadata:
            gsz = metadata["group_size"]
            group_rewards = all_rewards[reward_idx : reward_idx + gsz]
            reward_idx += gsz
            mean_r = sum(group_rewards) / len(group_rewards)
            advantages = [r - mean_r for r in group_rewards]
            batch_rewards.append(mean_r)
            if all(a == 0.0 for a in advantages):
                continue
            for tokens, logprob, advantage, ob_len in zip(
                metadata["group_tokens"],
                metadata["group_logprobs"],
                advantages,
                metadata["group_ob_lens"],
            ):
                input_tokens = [int(t) for t in tokens[:-1]]
                target_tokens = tokens[1:]
                all_logprobs = [0.0] * ob_len + logprob
                all_advantages = [0.0] * ob_len + [advantage] * (
                    len(input_tokens) - ob_len
                )
                all_mask = [0.0] * ob_len + [1.0] * (len(input_tokens) - ob_len)
                training_datums.append(
                    types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(
                                torch.tensor(target_tokens)
                            ),
                            "logprobs": TensorData.from_torch(
                                torch.tensor(all_logprobs)
                            ),
                            "advantages": TensorData.from_torch(
                                torch.tensor(all_advantages)
                            ),
                            "mask": TensorData.from_torch(torch.tensor(all_mask)),
                        },
                    )
                )

        if training_datums:
            if base_sampling_client is not None:
                kl_metrics = incorporate_kl_penalty_sync(
                    training_datums,
                    base_sampling_client,
                    config.kl_coeff,
                )
                metrics.update(kl_metrics)

            # PPO mini-batches (Table 1: ppo_mini_batch_size=64)
            mini = max(1, config.ppo_mini_batch_size)
            for i in range(0, len(training_datums), mini):
                chunk = training_datums[i : i + mini]
                fwd_bwd_future = training_client.forward_backward(
                    chunk,
                    loss_fn="ppo",
                    loss_fn_config=loss_fn_config,
                )
                optim_step_future = training_client.optim_step(adam_params)
                _ = fwd_bwd_future.result()
                _ = optim_step_future.result()

        metrics["time/total"] = time.time() - t_start
        metrics["reward/mean"] = (
            sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        )
        metrics["train/kl_coeff"] = config.kl_coeff
        metrics["train/clip_low"] = clip_low_threshold
        metrics["train/clip_high"] = clip_high_threshold
        metrics["train/ppo_mini_batch_size"] = float(config.ppo_mini_batch_size)
        metrics["train/n_datums"] = float(len(training_datums))
        ml_logger.log_metrics(metrics, step=batch_idx)

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
