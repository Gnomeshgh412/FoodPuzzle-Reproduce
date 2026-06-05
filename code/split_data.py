#!/usr/bin/env python3
"""Create reconstructed train/dev/test splits for public FoodPuzzle tasks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


class SplitError(Exception):
    """Split generation 中可预期的失败。"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL task 文件。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise SplitError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise SplitError(f"JSONL row is not an object at {path}:{line_no}")
            if row.get("id") is None:
                raise SplitError(f"missing id at {path}:{line_no}")
            rows.append(row)
    return rows


def split_counts(total: int, train_ratio: float, dev_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """按最大余数法计算 split 数量，保证总数精确等于 total。"""
    ratios = [train_ratio, dev_ratio, test_ratio]
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(range(3), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in order[:remaining]:
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_ids(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['id']}\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def validate_disjoint(train: list[dict[str, Any]], dev: list[dict[str, Any]], test: list[dict[str, Any]], total: int) -> None:
    """校验 train/dev/test 互斥且覆盖所有样本。"""
    train_ids = {row["id"] for row in train}
    dev_ids = {row["id"] for row in dev}
    test_ids = {row["id"] for row in test}
    if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
        raise SplitError("split ids are not disjoint")
    if len(train_ids | dev_ids | test_ids) != total:
        raise SplitError("split ids do not cover all rows")


def create_split(args: argparse.Namespace) -> int:
    input_path = resolve_input_path(args)
    output_dir = resolve_output_dir(args)

    ratio_sum = args.train_ratio + args.dev_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise SplitError("train/dev/test ratios must sum to 1.0")

    output_dir.mkdir(parents=True, exist_ok=True)

    # MFP/MPC split 统一由本脚本管理；MPC 输入应来自 reconstruct_mpc_data.py。
    rows = read_jsonl(input_path)

    # 使用固定 seed 做行级别随机划分。
    rng = random.Random(args.seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    train_n, dev_n, test_n = split_counts(
        len(shuffled), args.train_ratio, args.dev_ratio, args.test_ratio
    )
    train = shuffled[:train_n]
    dev = shuffled[train_n : train_n + dev_n]
    test = shuffled[train_n + dev_n :]

    # 校验 split 互斥并覆盖全部样本。
    validate_disjoint(train, dev, test, len(rows))

    # 写出 split JSONL 和 id 文件。
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "dev.jsonl", dev)
    write_jsonl(output_dir / "test.jsonl", test)

    metadata = {
        "task": args.task,
        "source_file": str(input_path),
        "seed": args.seed,
        "split_ratio": {
            "train": args.train_ratio,
            "dev": args.dev_ratio,
            "test": args.test_ratio,
        },
        "split_level": "row",
        "stratified": False,
        "near_duplicate_decontamination": False,
        "is_official_split": False,
        "total": len(rows),
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "notes": [
            f"The public {args.task.upper()} task file has no official split field.",
            "This split is reconstructed to approximate the paper's train/dev/test evaluation logic.",
            "No near-duplicate decontamination is applied because the paper does not disclose such a procedure.",
            "No category stratification is applied.",
        ],
    }

    if args.task == "mfp":
        # 保留 MFP 既有行为：写出 id 文件和 split metadata。
        write_ids(output_dir / "train_ids.txt", train)
        write_ids(output_dir / "dev_ids.txt", dev)
        write_ids(output_dir / "test_ids.txt", test)
        write_json(output_dir / "split_metadata.json", metadata)

    print("SPLIT_STATUS: PASS")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input:
        return Path(args.input)
    if args.task == "mpc":
        return Path("data/processed/MPC_reconstructed_tasks.jsonl")
    raise SplitError("--input is required for --task mfp")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.task == "mpc":
        return Path("results/splits/mpc")
    raise SplitError("--output-dir is required for --task mfp")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create reconstructed FoodPuzzle splits")
    parser.add_argument("--task", choices=["mfp", "mpc"], required=True)
    parser.add_argument("--input")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        return create_split(args)
    except SplitError as exc:
        print("SPLIT_STATUS: FAIL")
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
