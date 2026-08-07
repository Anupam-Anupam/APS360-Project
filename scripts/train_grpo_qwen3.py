"""
APS360 — GRPO (Zero-RL) training for Qwen3-1.7B-Base on mathematics-only data.

Uses the Tinker API for LoRA updates and a local vLLM verifier
(TIGER-Lab/general-verifier) for LLM-as-a-judge rewards.

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
    def __init__(self, model_name: str):
        self.llm = LLM(model=model_name, gpu_memory_utilization=0.7)
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


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "./log_qwen3_1p7b_math_grpo"
    model_name: str = "Qwen/Qwen3-1.7B-Base"
    dataset_path: str = "TIGER-Lab/WebInstruct-verified"
    math_only: bool = True
    batch_size: int = 128
    group_size: int = 8
    learning_rate: float = 4e-5
    max_prompt_length: int = 1024
    max_tokens: int = 4096
    temperature: float = 1.0
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

    verifier = GeneralVerifier(config.verifier_name)

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

    n_batches_per_epoch = max(1, len(train_dataset) // config.batch_size)
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

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(
        f"Training Qwen3-1.7B for {n_train_batches} steps "
        f"({config.total_epochs} epoch(s) x {n_batches_per_epoch} batches); "
        f"train_rows={len(train_dataset)}, batch_size={config.batch_size}"
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

        batch_start = within_epoch * config.batch_size
        batch_end = min((within_epoch + 1) * config.batch_size, len(train_dataset))
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
                for _ in range(config.group_size)
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
                        },
                    )
                )

        if training_datums:
            fwd_bwd_future = training_client.forward_backward(
                training_datums, loss_fn="importance_sampling"
            )
            optim_step_future = training_client.optim_step(adam_params)
            _ = fwd_bwd_future.result()
            _ = optim_step_future.result()

        metrics["time/total"] = time.time() - t_start
        metrics["reward/mean"] = (
            sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        )
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
