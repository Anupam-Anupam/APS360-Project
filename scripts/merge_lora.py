#!/usr/bin/env python3
"""Merge a Tinker LoRA adapter into Qwen/Qwen3-1.7B and save a HF folder for vLLM eval."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE = "Qwen/Qwen3-1.7B"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    ap.add_argument("--out", required=True, help="Output directory for merged model")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "config.json").exists():
        print(f"[merge] {out}/config.json exists; skipping.")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"[merge] base={args.base}")
    print(f"[merge] adapter={args.adapter}")
    print(f"[merge] out={args.out}")

    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(args.base)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(args.out)
    print("[merge] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
