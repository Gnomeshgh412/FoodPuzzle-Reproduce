#!/usr/bin/env python3
"""Reconstruct FlavorDB-derived MPC task input.

本脚本只做本地 JSONL / SQLite 数据处理，不调用 LLM/API，不运行 prediction 或 evaluation。
输出的 MPC reconstructed task 属于 FlavorDB-derived reconstruction，不是 official exact reproduction。
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def normalize_text(value: Any) -> str:
    """保守文本归一：用于 food / molecule 名称匹配，不做语义改写。"""
    text = str(value or "").lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"JSONL row is not object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_schema(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "notnull": row[3],
            "default_value": row[4],
            "primary_key": row[5],
        }
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    ]


def pick_first(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def build_food_index(conn: sqlite3.Connection, food_schema: list[dict[str, Any]]) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """构造分级 food name -> entity 候选索引，避免 synonym/category 抢先造成歧义。"""
    columns = {row["name"] for row in food_schema}
    entity_id_col = pick_first(columns, ["entity_id", "id"])
    if not entity_id_col:
        raise RuntimeError("food_entities has no usable entity id column")

    name_fields = [
        field
        for field in [
            "entity_alias_readable",
            "entity_alias",
            "natural_source_name",
            "entity_alias_basket",
            "entity_alias_synonyms",
            "category_readable",
            "category",
        ]
        if field in columns
    ]
    if not name_fields:
        raise RuntimeError("food_entities has no usable food name fields")

    select_cols = [entity_id_col] + name_fields
    sql = "SELECT " + ", ".join(quote_ident(col) for col in select_cols) + " FROM food_entities"
    level_fields: list[tuple[str, str]] = []
    for field in ["entity_alias_readable", "entity_alias"]:
        if field in name_fields:
            level_fields.append((field, field))
    for field in name_fields:
        if field not in {"entity_alias_readable", "entity_alias"}:
            level_fields.append((field, field))

    index: dict[str, dict[str, list[dict[str, Any]]]] = {
        level: defaultdict(list) for level, _field in level_fields
    }
    for row in conn.execute(sql).fetchall():
        row_dict = dict(zip(select_cols, row))
        entity_id = row_dict[entity_id_col]
        for level, field in level_fields:
            value = row_dict.get(field)
            if value is None or not str(value).strip():
                continue
            variants = [str(value)]
            # synonyms / basket 可能是分隔字符串；只按简单分隔展开，不做语义扩展。
            if field in {"entity_alias_synonyms", "entity_alias_basket"}:
                variants.extend(v.strip() for v in re.split(r"[;,|]", str(value)) if v.strip())
            for variant in variants:
                key = normalize_text(variant)
                if key:
                    index[level][key].append(
                        {
                            "entity_id": entity_id,
                            "name": variant,
                            "field": field,
                        }
                    )

    used_fields = {
        "food_entity_id": entity_id_col,
        "food_name_fields": name_fields,
    }
    return index, used_fields


def match_food(food: str, food_index: dict[str, dict[str, list[dict[str, Any]]]]) -> tuple[str, Any | None, list[dict[str, Any]]]:
    """按字段优先级匹配 food；某一级命中后不再尝试更低优先级。"""
    key = normalize_text(food)
    for _level, level_index in food_index.items():
        candidates = level_index.get(key, [])
        if not candidates:
            continue
        dedup: dict[Any, dict[str, Any]] = {}
        for candidate in candidates:
            dedup.setdefault(candidate["entity_id"], candidate)
        unique = list(dedup.values())
        if len(unique) > 1:
            return "ambiguous", None, unique
        return "matched_unique", unique[0]["entity_id"], unique
    return "unmatched", None, []


def load_full_molecules(
    conn: sqlite3.Connection,
    entity_id: Any,
    molecule_id_col: str,
    molecule_name_col: str,
) -> tuple[list[str], dict[str, list[str]]]:
    """读取某个 food 的完整 molecule set，并按名称稳定排序。"""
    sql = f"""
        SELECT m.{quote_ident(molecule_name_col)}
        FROM entity_molecule_link eml
        JOIN molecules m ON eml.molecule_id = m.{quote_ident(molecule_id_col)}
        WHERE eml.entity_id = ?
        ORDER BY LOWER(m.{quote_ident(molecule_name_col)}), m.{quote_ident(molecule_name_col)}
    """
    full: list[str] = []
    seen_raw: set[str] = set()
    norm_to_raw: dict[str, list[str]] = defaultdict(list)
    for row in conn.execute(sql, (entity_id,)).fetchall():
        if row[0] is None:
            continue
        name = str(row[0]).strip()
        if not name or name in seen_raw:
            continue
        seen_raw.add(name)
        full.append(name)
        norm_to_raw[normalize_text(name)].append(name)
    return full, norm_to_raw


def stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(values), "max": max(values), "mean": mean(values)}


def run(args: argparse.Namespace) -> int:
    mpc_path = Path(args.mpc)
    db_path = Path(args.db)
    output_path = Path(args.output)

    tasks = read_jsonl(mpc_path)
    conn = sqlite3.connect(db_path)
    try:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        required_tables = {"food_entities", "molecules", "entity_molecule_link"}
        missing_tables = sorted(required_tables - set(tables))
        if missing_tables:
            raise RuntimeError(f"missing FlavorDB tables: {missing_tables}")

        food_schema = table_schema(conn, "food_entities")
        molecule_schema = table_schema(conn, "molecules")
        molecule_columns = {row["name"] for row in molecule_schema}
        molecule_id_col = pick_first(molecule_columns, ["id", "molecule_id"])
        molecule_name_col = pick_first(molecule_columns, ["common_name", "name", "iupac_name"])
        if not molecule_id_col or not molecule_name_col:
            raise RuntimeError("molecules table has no usable id/name columns")

        food_index, food_used_fields = build_food_index(conn, food_schema)
        reconstructed: list[dict[str, Any]] = []
        invalid_cases: list[dict[str, Any]] = []
        failure_reasons: Counter[str] = Counter()
        food_status: Counter[str] = Counter()
        full_counts: list[int] = []
        partial_counts: list[int] = []
        missing_counts: list[int] = []
        full_available = 0
        missing_subset_success = 0

        for task in tasks:
            row_id = task.get("id")
            food = task.get("food")
            missing = task.get("missing_molecules")
            if not isinstance(food, str) or not food.strip():
                failure_reasons["invalid_food"] += 1
                invalid_cases.append({"id": row_id, "reason": "invalid_food"})
                continue
            if not isinstance(missing, list):
                failure_reasons["invalid_missing_molecules"] += 1
                invalid_cases.append({"id": row_id, "reason": "invalid_missing_molecules"})
                continue

            status, entity_id, candidates = match_food(food, food_index)
            food_status[status] += 1
            if status != "matched_unique":
                reason = "food_unmatched" if status == "unmatched" else "food_ambiguous"
                failure_reasons[reason] += 1
                invalid_cases.append(
                    {
                        "id": row_id,
                        "food": food,
                        "reason": reason,
                        "candidate_entity_ids": [item["entity_id"] for item in candidates],
                    }
                )
                continue

            full_molecules, full_norm_map = load_full_molecules(
                conn, entity_id, molecule_id_col, molecule_name_col
            )
            if full_molecules:
                full_available += 1
            else:
                failure_reasons["empty_full_molecules"] += 1
                invalid_cases.append({"id": row_id, "food": food, "reason": "empty_full_molecules"})
                continue

            missing_norms: set[str] = set()
            missing_not_in_full: list[str] = []
            for molecule in missing:
                norm = normalize_text(molecule)
                if norm in full_norm_map:
                    missing_norms.add(norm)
                else:
                    missing_not_in_full.append(str(molecule))

            if missing_not_in_full:
                failure_reasons["missing_not_subset_of_full"] += 1
                invalid_cases.append(
                    {
                        "id": row_id,
                        "food": food,
                        "reason": "missing_not_subset_of_full",
                        "missing_not_in_full": missing_not_in_full,
                    }
                )
                continue
            missing_subset_success += 1

            partial_molecules = [
                molecule for molecule in full_molecules if normalize_text(molecule) not in missing_norms
            ]
            if not partial_molecules:
                failure_reasons["partial_empty"] += 1
                invalid_cases.append({"id": row_id, "food": food, "reason": "partial_empty"})
                continue

            # 正式任务文件只保留必要字段；内部匹配状态不写入正式 JSONL。
            reconstructed.append(
                {
                    "id": row_id,
                    "task": task.get("task", "MPC"),
                    "target_food": food,
                    "partial_molecules": partial_molecules,
                    "n": len(missing),
                    "missing_molecules": missing,
                }
            )
            full_counts.append(len(full_molecules))
            partial_counts.append(len(partial_molecules))
            missing_counts.append(len(missing))

        if invalid_cases:
            raise RuntimeError(f"MPC reconstruction failed: {len(invalid_cases)} invalid cases")

        write_jsonl(output_path, reconstructed)

        summary = {
            "task": "mpc",
            "reconstruction_type": "FlavorDB-derived MPC reconstruction",
            "is_official_exact_reproduction": False,
            "source_files": {
                "mpc_tasks": str(mpc_path),
                "flavordb": str(db_path),
            },
            "output_file": str(output_path),
            "total_samples": len(tasks),
            "food_matched_unique": food_status["matched_unique"],
            "food_unmatched": food_status["unmatched"],
            "food_ambiguous": food_status["ambiguous"],
            "full_molecules_available": full_available,
            "missing_subset_success": missing_subset_success,
            "reconstruction_success": len(reconstructed),
            "invalid_cases": len(invalid_cases),
            "failure_reasons": dict(sorted(failure_reasons.items())),
            "used_fields": {
                **food_used_fields,
                "molecule_id": molecule_id_col,
                "molecule_name": molecule_name_col,
            },
            "count_stats": {
                "missing_molecules": stats(missing_counts),
                "partial_molecules": stats(partial_counts),
                "full_molecules": stats(full_counts),
            },
            "notes": [
                "This is FlavorDB-derived MPC reconstruction, not official exact reproduction.",
                "Official partial_molecules and n are not available in the public MPC task file.",
                "The reconstructed task file keeps missing_molecules as gold labels; they must not be exposed to models during prediction.",
                "Train/dev/test split is handled by code/split_data.py, not reconstruct_mpc_data.py.",
            ],
        }
        print("MPC_RECONSTRUCTION_STATUS: PASS")
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct FlavorDB-derived MPC task input.")
    parser.add_argument("--mpc", default="data/processed/MPC_tasks.jsonl")
    parser.add_argument("--db", default="data/raw/flavordb.db")
    parser.add_argument("--output", default="data/processed/MPC_reconstructed_tasks.jsonl")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
