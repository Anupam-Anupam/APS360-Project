#!/usr/bin/env python3
"""
Filter WebInstruct-verified down to mathematics examples and write parquet
files for offline inspection / alternate trainers.

Usage:
  python scripts/prepare_math_data.py --out-dir ./data/math_webinstruct
"""
from __future__ import annotations

import argparse
from pathlib import Path

import datasets

MATH_KEYWORDS = (
    "math",
    "mathematics",
    "algebra",
    "geometry",
    "calculus",
    "statistics",
    "probability",
    "number theory",
    "arithmetic",
)


def is_math(example: dict) -> bool:
    for key in ("category", "topic", "subject", "domain"):
        val = example.get(key)
        if not val:
            continue
        text = str(val).lower()
        if any(k in text for k in MATH_KEYWORDS):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="TIGER-Lab/WebInstruct-verified")
    ap.add_argument("--out-dir", default="./data/math_webinstruct")
    ap.add_argument("--test-size", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = datasets.load_dataset(args.dataset)
    train = ds["train"]
    test = ds["test"] if "test" in ds else None

    pre = len(train)
    train = train.filter(is_math)
    print(f"train math filter: {pre} -> {len(train)}")
    if test is not None:
        pre_t = len(test)
        test = test.filter(is_math)
        print(f"test math filter: {pre_t} -> {len(test)}")
        if len(test) > args.test_size:
            test = test.select(range(args.test_size))

    train.to_parquet(out / "train.parquet")
    if test is not None:
        test.to_parquet(out / "test.parquet")
    print(f"Wrote parquet under {out}")


if __name__ == "__main__":
    main()
