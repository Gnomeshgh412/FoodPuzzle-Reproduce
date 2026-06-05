#!/usr/bin/env python3
"""Audit the reproduced FoodPuzzle data files.

This script only reads local data files and prints structural summaries.
It does not run evaluation, model inference, or any external API calls.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DB = ROOT / "data" / "raw" / "flavordb.db"
MFP_JSONL = ROOT / "data" / "processed" / "MFP_tasks.jsonl"
MPC_JSONL = ROOT / "data" / "processed" / "MPC_tasks.jsonl"
MPC_RECONSTRUCTED_JSONL = ROOT / "data" / "processed" / "MPC_reconstructed_tasks.jsonl"
MPC_SPLIT_FILES = [
    ROOT / "results" / "splits" / "mpc" / "train.jsonl",
    ROOT / "results" / "splits" / "mpc" / "dev.jsonl",
    ROOT / "results" / "splits" / "mpc" / "test.jsonl",
]
REQUIRED_FILES = [RAW_DB, MFP_JSONL, MPC_JSONL, MPC_RECONSTRUCTED_JSONL, *MPC_SPLIT_FILES]

FIELDS_OF_INTEREST = {
    "split",
    "label",
    "answer",
    "target",
    "category",
    "food",
    "foods",
    "molecule",
    "molecules",
    "missing",
    "partial",
}

FIELD_NAME_FRAGMENTS = (
    "label",
    "answer",
    "target",
    "category",
    "food",
    "molecule",
    "missing",
    "partial",
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def summarize_value(value: Any) -> str:
    if isinstance(value, dict):
        keys = list(value.keys())[:5]
        return f"dict(len={len(value)}, keys={keys})"
    if isinstance(value, list):
        first_type = type(value[0]).__name__ if value else "empty"
        return f"list(len={len(value)}, first={first_type})"
    if isinstance(value, str):
        return f"str(len={len(value)})"
    if value is None:
        return "null"
    return type(value).__name__


def check_required_files() -> bool:
    print("== Required files ==")
    ok = True
    # 检查官方数据、MPC reconstructed data 和 MPC split 文件是否存在。
    # reconstructed data 由 reconstruct_mpc_data.py 生成；split 由 split_data.py 生成。
    for path in REQUIRED_FILES:
        if path.exists():
            print(f"PASS {rel(path)} size={path.stat().st_size}")
        else:
            print(f"FAIL {rel(path)} missing")
            ok = False
    return ok


def audit_jsonl(path: Path) -> bool:
    print(f"\n== JSONL audit: {rel(path)} ==")
    total = 0
    valid = 0
    top_key_counter: Counter[str] = Counter()
    interest_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []

    try:
        with path.open("r", encoding="utf-8") as f:
            # 逐行读取 JSONL，避免把任务文件整体载入后再审计。
            for line_no, line in enumerate(f, start=1):
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"FAIL invalid JSON at line {line_no}: {exc}")
                    return False
                if not isinstance(item, dict):
                    print(f"FAIL line {line_no} is {type(item).__name__}, expected object")
                    return False

                valid += 1
                if len(samples) < 3:
                    samples.append(item)

                # 汇总 top-level keys，并对关注字段统计出现频率。
                keys = set(item.keys())
                top_key_counter.update(keys)
                interest_keys = {
                    key
                    for key in keys
                    if key in FIELDS_OF_INTEREST
                    or any(fragment in key.lower() for fragment in FIELD_NAME_FRAGMENTS)
                }
                interest_counter.update(interest_keys)
                if "split" in item:
                    split_counter.update([str(item["split"])])
    except OSError as exc:
        print(f"FAIL cannot read {rel(path)}: {exc}")
        return False

    print(f"total_lines: {total}")
    print(f"valid_json_objects: {valid}")
    print("top_level_key_frequency:")
    for key, count in top_key_counter.most_common():
        print(f"  {key}: {count}")

    print("first_3_samples_key_type_summary:")
    for idx, sample in enumerate(samples, start=1):
        summary = {key: summarize_value(sample[key]) for key in sorted(sample.keys())}
        print(f"  sample_{idx}: {summary}")

    if split_counter:
        # 如果官方任务文件含 split 字段，统计 split 分布。
        print("split_distribution:")
        for split, count in split_counter.most_common():
            print(f"  {split}: {count}")
    else:
        print("split_distribution: no split field found")

    print("fields_of_interest_frequency:")
    if interest_counter:
        for key, count in interest_counter.most_common():
            print(f"  {key}: {count}")
    else:
        print("  none")

    return True


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def audit_sqlite(path: Path) -> bool:
    print(f"\n== SQLite audit: {rel(path)} ==")
    try:
        conn = sqlite3.connect(path)
    except sqlite3.Error as exc:
        print(f"FAIL cannot open SQLite database: {exc}")
        return False

    try:
        cur = conn.cursor()
        # 读取 SQLite schema 中所有普通 table 名称。
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [row[0] for row in cur.fetchall()]
        print(f"tables: {tables}")

        table_counts: dict[str, int] = {}
        food_candidates: list[tuple[str, int]] = []
        molecule_candidates: list[tuple[str, int]] = []

        for table in tables:
            q_table = quote_identifier(table)
            # 对每张表统计 row count，作为 raw 数据规模审计。
            cur.execute(f"SELECT COUNT(*) FROM {q_table}")
            row_count = int(cur.fetchone()[0])
            table_counts[table] = row_count

            cur.execute(f"PRAGMA table_info({q_table})")
            columns = [row[1] for row in cur.fetchall()]
            print(f"table {table}: rows={row_count}, columns={columns[:10]}")

            lowered_table = table.lower()
            lowered_columns = [col.lower() for col in columns]
            if "food" in lowered_table or any("food" in col for col in lowered_columns):
                food_candidates.append((table, row_count))
            if (
                "molecule" in lowered_table
                or "compound" in lowered_table
                or any("molecule" in col or "compound" in col for col in lowered_columns)
            ):
                molecule_candidates.append((table, row_count))

        # 对 FlavorDB 中名称明确的表单独统计 food 和 molecule 数量。
        if "food_entities" in table_counts:
            print(f"identified_food_count: food_entities rows={table_counts['food_entities']}")
        else:
            print("identified_food_count: not reliably identified")

        if "molecules" in table_counts:
            print(f"identified_molecule_count: molecules rows={table_counts['molecules']}")
        else:
            print("identified_molecule_count: not reliably identified")

        # 候选表只根据 table/column 名称识别，用于提示可能相关结构，不猜测官方 schema 语义。
        if food_candidates:
            print("food_related_table_candidates:")
            for table, row_count in food_candidates:
                print(f"  {table}: rows={row_count}")
        else:
            print("food_related_table_candidates: none reliably identified")

        if molecule_candidates:
            print("molecule_related_table_candidates:")
            for table, row_count in molecule_candidates:
                print(f"  {table}: rows={row_count}")
        else:
            print("molecule_related_table_candidates: none reliably identified")

    except sqlite3.Error as exc:
        print(f"FAIL SQLite read error: {exc}")
        return False
    finally:
        conn.close()

    return True


def main() -> int:
    ok = check_required_files()
    if ok:
        ok = audit_jsonl(MFP_JSONL) and ok
        ok = audit_jsonl(MPC_JSONL) and ok
        ok = audit_jsonl(MPC_RECONSTRUCTED_JSONL) and ok
        for split_path in MPC_SPLIT_FILES:
            ok = audit_jsonl(split_path) and ok
        ok = audit_sqlite(RAW_DB) and ok

    # 最后输出明确 PASS / FAIL 状态，方便人工和脚本检查。
    if ok:
        print("\nDATA_AUDIT_STATUS: PASS")
        return 0

    print("\nDATA_AUDIT_STATUS: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
