import json
import argparse
import pandas as pd
from . import common
from .gpqa_eval_qwen import GPQAEvalQwen
from .aime24_eval_qwen import AIME24EvalQwen
from .aime25_eval_qwen import AIME25EvalQwen
from .gsm8k_eval_qwen import Gsm8kEvalQwen
from .minerva_eval_qwen import MinervaEvalQwen
from .amc_eval_qwen import AmcEvalQwen
from .math_eval_qwen import MathEvalQwen
from .olympiad_eval_qwen import OlympiadEvalQwen
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)
from .sampler.qwen_chat_completion_sampler import QwenChatCompletionSampler


def main():
    parser = argparse.ArgumentParser(
        description="Run math / OOD evals against a served Qwen chat API."
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model", type=str, help="Model key from the registry")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="HF id or local path; overrides --model registry entry",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument(
        "--evals",
        type=str,
        default="gsm8k,math,aime24,amc",
        help="Comma-separated eval names",
    )
    args = parser.parse_args()

    models = {
        "Qwen3-1.7B-Base": QwenChatCompletionSampler(
            model="Qwen/Qwen3-1.7B-Base",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1,
            max_tokens=4096,
        ),
        "Qwen3-1.7B-Instruct": QwenChatCompletionSampler(
            model="Qwen/Qwen3-1.7B",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1,
            max_tokens=4096,
        ),
        "aps360-qwen3-1.7b-math-grpo": QwenChatCompletionSampler(
            model="aps360-qwen3-1.7b-math-grpo",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1,
            max_tokens=4096,
        ),
    }

    if args.model_path:
        models = {
            "custom": QwenChatCompletionSampler(
                model=args.model_path,
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                temperature=1,
                max_tokens=4096,
            )
        }

    if args.list_models:
        for name in models:
            print(f" - {name}")
        return

    if args.model and not args.model_path:
        if args.model not in models:
            print(f"Error: Model '{args.model}' not found.")
            return
        models = {args.model: models[args.model]}

    equality_checker = ChatCompletionSampler(model="gpt-4o")

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        match eval_name:
            case "math":
                return MathEvalQwen(
                    equality_checker=equality_checker,
                    num_examples=num_examples,
                    n_repeats=1,
                )
            case "aime24":
                return AIME24EvalQwen(
                    equality_checker=equality_checker,
                    n_repeats=1 if debug_mode else 8,
                    num_examples=num_examples,
                )
            case "aime25":
                return AIME25EvalQwen(
                    equality_checker=equality_checker,
                    n_repeats=1 if debug_mode else 8,
                    num_examples=num_examples,
                )
            case "olympiad":
                return OlympiadEvalQwen(
                    equality_checker=equality_checker,
                    n_repeats=1,
                    num_examples=num_examples,
                )
            case "gsm8k":
                return Gsm8kEvalQwen(
                    equality_checker=equality_checker,
                    n_repeats=1,
                    num_examples=num_examples,
                )
            case "minerva":
                return MinervaEvalQwen(
                    equality_checker=equality_checker,
                    n_repeats=1,
                    num_examples=num_examples,
                )
            case "amc":
                return AmcEvalQwen(
                    equality_checker=equality_checker,
                    n_repeats=1,
                    num_examples=num_examples,
                )
            case "gpqa":
                return GPQAEvalQwen(n_repeats=1, num_examples=num_examples)
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    eval_names = [e.strip() for e in args.evals.split(",") if e.strip()]
    evals = {name: get_evals(name, args.debug) for name in eval_names}
    debug_suffix = "_DEBUG" if args.debug else ""
    mergekey2resultpath = {}
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            file_stem = f"{eval_name}_{model_name}"
            report_filename = f"./{file_stem}{debug_suffix}.html"
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"./{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            mergekey2resultpath[file_stem] = result_filename

    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
