#!/usr/bin/env python3
"""
Pull the Colab-prepared math dataset and optionally write local parquet.

Default source: https://huggingface.co/datasets/anupamc/math-dataset
(already curated; no category filter needed).

Usage:
  python scripts/prepare_math_data.py --out-dir ./data/math_dataset
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
    ap.add_argument("--dataset", default="anupamc/math-dataset")
    ap.add_argument("--out-dir", default="./data/math_dataset")
    ap.add_argument(
        "--filter-math",
        action="store_true",
        help="Optional category filter (only needed for raw WebInstruct).",
    )
    ap.add_argument("--test-size", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = datasets.load_dataset(args.dataset)
    train = ds["train"]
    test = ds["test"] if "test" in ds else None

    if args.filter_math:
        pre = len(train)
        train = train.filter(is_math)
        print(f"train math filter: {pre} -> {len(train)}")
        if test is not None:
            pre_t = len(test)
            test = test.filter(is_math)
            print(f"test math filter: {pre_t} -> {len(test)}")

    if test is not None and args.test_size is not None and len(test) > args.test_size:
        test = test.select(range(args.test_size))

    print(f"train={len(train)}" + (f", test={len(test)}" if test is not None else ""))
    train.to_parquet(out / "train.parquet")
    if test is not None:
        test.to_parquet(out / "test.parquet")
    print(f"Wrote parquet under {out}")


if __name__ == "__main__":
    main()
