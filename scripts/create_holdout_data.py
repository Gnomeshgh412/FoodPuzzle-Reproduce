#!/usr/bin/env python3
"""Select an aligned MFP/MPC holdout without changing the existing splits."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


class HoldoutError(Exception):
    """Expected holdout selection failure."""


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise HoldoutError(f"required JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HoldoutError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict) or row.get("id") is None:
                raise HoldoutError(f"invalid row at {path}:{line_number}")
            rows.append(row)
    return rows


def food_name(task: str, row: dict[str, Any]) -> str:
    field = "actual_food" if task == "mfp" else "target_food"
    value = str(row.get(field) or "").strip()
    if not value:
        raise HoldoutError(f"{task} row {row.get('id')} has no {field}")
    return value


def molecule_profile(task: str, row: dict[str, Any]) -> frozenset[str]:
    if task == "mfp":
        values = row.get("molecules")
    else:
        partial = row.get("partial_molecules")
        missing = row.get("missing_molecules")
        values = (
            (partial if isinstance(partial, list) else [])
            + (missing if isinstance(missing, list) else [])
        )
    if not isinstance(values, list):
        raise HoldoutError(f"{task} row {row.get('id')} has no molecule list")
    profile = frozenset(normalize(value) for value in values if normalize(value))
    if not profile:
        raise HoldoutError(f"{task} row {row.get('id')} has an empty molecule profile")
    return profile


def index_unique_foods(
    task: str, rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = normalize(food_name(task, row))
        if key in indexed:
            raise HoldoutError(f"duplicate normalized {task} food name: {key}")
        indexed[key] = row
    return indexed


def profile_frequencies(
    task: str, rows: list[dict[str, Any]]
) -> Counter[frozenset[str]]:
    return Counter(molecule_profile(task, row) for row in rows)


def select_holdout(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    mfp_train = read_jsonl(Path(args.mfp_train))
    mpc_train = read_jsonl(Path(args.mpc_train))
    mfp_all = read_jsonl(Path(args.mfp_all))
    mpc_all = read_jsonl(Path(args.mpc_all))

    mfp_train_by_food = index_unique_foods("mfp", mfp_train)
    mpc_train_by_food = index_unique_foods("mpc", mpc_train)
    mfp_profile_counts = profile_frequencies("mfp", mfp_all)
    mpc_profile_counts = profile_frequencies("mpc", mpc_all)

    common_train_foods = sorted(set(mfp_train_by_food) & set(mpc_train_by_food))
    eligible_foods: list[str] = []
    excluded_mfp_duplicate = 0
    excluded_mpc_duplicate = 0
    for key in common_train_foods:
        mfp_unique = (
            mfp_profile_counts[molecule_profile("mfp", mfp_train_by_food[key])] == 1
        )
        mpc_unique = (
            mpc_profile_counts[molecule_profile("mpc", mpc_train_by_food[key])] == 1
        )
        excluded_mfp_duplicate += int(not mfp_unique)
        excluded_mpc_duplicate += int(not mpc_unique)
        if mfp_unique and mpc_unique:
            eligible_foods.append(key)

    if args.count <= 0:
        raise HoldoutError("--count must be positive")
    if args.count > len(eligible_foods):
        raise HoldoutError(
            f"requested {args.count} holdout foods, but only "
            f"{len(eligible_foods)} are eligible"
        )

    shuffled = list(eligible_foods)
    random.Random(args.seed).shuffle(shuffled)
    selected_foods = sorted(shuffled[: args.count])
    mfp_selected = [mfp_train_by_food[key] for key in selected_foods]
    mpc_selected = [mpc_train_by_food[key] for key in selected_foods]
    holdout_ids = [
        {
            "food": food_name("mfp", mfp_train_by_food[key]),
            "normalized_food": key,
            "mfp_id": str(mfp_train_by_food[key]["id"]),
            "mpc_id": str(mpc_train_by_food[key]["id"]),
        }
        for key in selected_foods
    ]

    mfp_sizes = [len(molecule_profile("mfp", row)) for row in mfp_selected]
    mpc_sizes = [len(molecule_profile("mpc", row)) for row in mpc_selected]
    mpc_n = [int(row.get("n") or 0) for row in mpc_selected]
    summary = {
        "seed": args.seed,
        "requested_count": args.count,
        "common_train_food_count": len(common_train_foods),
        "eligible_food_count": len(eligible_foods),
        "excluded_by_mfp_duplicate_profile": excluded_mfp_duplicate,
        "excluded_by_mpc_duplicate_profile": excluded_mpc_duplicate,
        "selection": {
            "aligned_foods": len(selected_foods),
            "mfp_rows": len(mfp_selected),
            "mpc_rows": len(mpc_selected),
        },
        "mfp_profile_size": {
            "min": min(mfp_sizes),
            "median": statistics.median(mfp_sizes),
            "max": max(mfp_sizes),
        },
        "mpc_profile_size": {
            "min": min(mpc_sizes),
            "median": statistics.median(mpc_sizes),
            "max": max(mpc_sizes),
        },
        "mpc_n": {
            "min": min(mpc_n),
            "median": statistics.median(mpc_n),
            "max": max(mpc_n),
        },
        "rules": [
            "Candidates must be present in both current train splits.",
            "MFP and MPC holdouts contain the same normalized food names.",
            "A candidate is excluded if its complete molecule profile is duplicated "
            "anywhere in the corresponding complete task dataset.",
            "Existing split files are never modified.",
        ],
    }
    return mfp_selected, mpc_selected, holdout_ids, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(
    output_dir: Path,
    mfp_rows: list[dict[str, Any]],
    mpc_rows: list[dict[str, Any]],
    holdout_ids: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    targets = [
        output_dir / "mfp_test.jsonl",
        output_dir / "mpc_test.jsonl",
        output_dir / "holdout_ids.json",
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise HoldoutError(
            "refusing to overwrite existing holdout files: " + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(targets[0], mfp_rows)
    write_jsonl(targets[1], mpc_rows)
    payload = {
        "seed": summary["seed"],
        "count": summary["requested_count"],
        "items": holdout_ids,
    }
    targets[2].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select an aligned MFP/MPC holdout from the existing train splits. "
            "The default mode is read-only; pass --write to create output files."
        )
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20250727)
    parser.add_argument("--mfp-train", default="results/splits/mfp/train.jsonl")
    parser.add_argument("--mpc-train", default="results/splits/mpc/train.jsonl")
    parser.add_argument("--mfp-all", default="data/processed/MFP_tasks.jsonl")
    parser.add_argument(
        "--mpc-all", default="data/processed/MPC_reconstructed_tasks.jsonl"
    )
    parser.add_argument("--output-dir", default="data/holdout")
    parser.add_argument(
        "--write",
        action="store_true",
        help="create the three formal holdout files; omitted means read-only check",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        mfp_rows, mpc_rows, holdout_ids, summary = select_holdout(args)
        if args.write:
            write_outputs(
                Path(args.output_dir),
                mfp_rows,
                mpc_rows,
                holdout_ids,
                summary,
            )
        print("HOLDOUT_STATUS: PASS")
        print(f"mode: {'write' if args.write else 'check-only'}")
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except HoldoutError as exc:
        print(f"HOLDOUT_STATUS: FAIL\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
