#!/usr/bin/env python3
"""Minimal zero-shot baseline prediction generator for FoodPuzzle.

本脚本只生成 prediction JSONL，不做 retrieval、ICL、BM25 或 Scientific Agent。
LLM provider / .env.local / API fallback 逻辑复用 code/evaluation.py。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


class ZeroShotError(Exception):
    """Zero-shot prediction generation 中可预期的失败。"""


def load_evaluation_module() -> Any:
    """从同目录加载 evaluation.py，避免重构已有 provider/API 调用逻辑。"""
    evaluation_path = Path(__file__).resolve().parent / "evaluation.py"
    spec = importlib.util.spec_from_file_location("foodpuzzle_evaluation", evaluation_path)
    if spec is None or spec.loader is None:
        raise ZeroShotError(f"cannot import evaluation module from {evaluation_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取完整 task JSONL；坏行作为脚本级失败处理。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ZeroShotError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ZeroShotError(f"JSONL row is not an object at {path}:{line_no}")
            rows.append(row)
    return rows


def parse_json_value(content: str) -> Any | None:
    """解析 LLM JSON 输出；支持裸 JSON 和 Markdown code fence 中的 JSON。"""
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # 有些模型会在 JSON 前后加少量文字；优先截取对象，其次截取数组。
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    return None


def parse_json_object(content: str) -> dict[str, Any] | None:
    """解析 JSON object；MFP 兼容旧逻辑。"""
    data = parse_json_value(content)
    return data if isinstance(data, dict) else None


def parse_text_molecule_list(content: str) -> list[str]:
    """解析编号列表或 bullet list；只处理模型输出文本，不接触 prompt 或 gold。"""
    molecules: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        match = re.match(r"^(?:\d+[\.)]|[-*•])\s+(.+?)\s*$", line)
        if not match:
            continue
        item = match.group(1).strip()
        item = re.sub(r"^[\"']|[\"']$", "", item)
        item = re.sub(r"[,;。；]+$", "", item).strip()
        if item:
            molecules.append(item)
    return molecules


def parse_mpc_molecules(content: str) -> list[str] | None:
    """MPC 输出解析：支持 JSON object、JSON array、code fence、编号列表和 bullet list。"""
    data = parse_json_value(content)
    if isinstance(data, dict):
        value = data.get("predicted_molecules")
        if not isinstance(value, list):
            return None
        return value
    if isinstance(data, list):
        return data

    text_list = parse_text_molecule_list(content)
    if text_list:
        return text_list
    return None


def normalize_molecule(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_mfp_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """MFP zero-shot prompt：只使用 molecules，不使用 gold actual_food。"""
    molecules = row.get("molecules")
    if not isinstance(molecules, list):
        molecules = []
    prompt = (
        "FoodPuzzle Molecular Food Prediction task.\n"
        "Given only a list of flavor molecules, predict the most likely food source.\n"
        "Return a concise free-text food source or food category, not a fixed label.\n"
        "Do not answer with odor/flavor descriptors such as roasted, nutty, meaty, sweet, fatty, or smoky.\n"
        "Do not answer with cooking-style flavor associations such as roasted meat or roasted nuts unless it is truly a food source.\n"
        "If uncertain, choose a plausible higher-level food source/category rather than a specific aroma association.\n"
        "Do not explain your answer. Do not use Markdown. Return JSON only.\n"
        'Required JSON format: {"predicted_food": "<food name or food category>"}\n'
        "The predicted_food value should be a short string.\n"
        f"Flavor molecules: {json.dumps(molecules, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": "You are a strict FoodPuzzle zero-shot baseline that returns only valid JSON."},
        {"role": "user", "content": prompt},
    ]


def build_mpc_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """MPC zero-shot prompt：只使用 target_food、partial_molecules、n，不读取 gold missing_molecules。"""
    # MPC reconstructed split 使用 target_food；保留 food fallback 只为兼容旧探索性输入。
    food = row.get("target_food") or row.get("food") or ""
    partial_molecules = row.get("partial_molecules")
    n = row.get("n")
    if not isinstance(food, str) or not food.strip():
        raise ZeroShotError("invalid_target_food")
    if not isinstance(partial_molecules, list):
        raise ZeroShotError("invalid_partial_molecules")
    if not isinstance(n, int) or n <= 0:
        raise ZeroShotError("invalid_n")

    known_molecules = [str(item).strip() for item in partial_molecules if str(item).strip()]
    known_text = ", ".join(known_molecules)
    prompt = (
        "Task:\n"
        "You are a flavor chemistry assistant.\n\n"
        "Given a food item, a list of known flavor molecules already associated with that food, "
        "and the number of molecules to infer, predict the most likely additional flavor molecules.\n\n"
        "Food:\n"
        f"{food.strip()}\n\n"
        "Known molecules:\n"
        f"{known_text}\n\n"
        "Number of molecules to predict:\n"
        f"{n}\n\n"
        "Instruction:\n"
        f"Predict exactly {n} flavor molecules if possible. If exact {n} is difficult, still return the best possible list.\n"
        "Return molecule common names when possible.\n"
        "Do not include molecules already listed in Known molecules.\n"
        "Do not output flavor descriptors such as nutty, roasted, sweet, fruity, meaty, or floral.\n"
        "Do not output chemical class/category words such as esters, aldehydes, ketones, sulfur compounds, or pyrazines.\n"
        "Do not output explanations.\n"
        "Do not output Markdown.\n"
        "Return only valid JSON.\n"
        "Use exactly this JSON shape:\n"
        '{\n'
        '  "predicted_molecules": ["molecule name 1", "molecule name 2"]\n'
        '}'
    )
    return [
        {"role": "system", "content": "You are a strict FoodPuzzle zero-shot baseline that returns only valid JSON."},
        {"role": "user", "content": prompt},
    ]


def validate_mfp_prediction(data: dict[str, Any]) -> str | None:
    value = data.get("predicted_food")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def validate_mpc_prediction(data: dict[str, Any], row: dict[str, Any]) -> list[str] | None:
    value = data.get("predicted_molecules")
    if not isinstance(value, list):
        return None
    n = row.get("n")
    if not isinstance(n, int) or n <= 0:
        return None
    known = {
        normalize_molecule(item)
        for item in row.get("partial_molecules", [])
        if normalize_molecule(item)
    }
    molecules: list[str] = []
    seen: set[str] = set()
    for item in value:
        molecule = str(item).strip()
        normalized = normalize_molecule(molecule)
        if not molecule or not normalized or normalized in known or normalized in seen:
            continue
        seen.add(normalized)
        molecules.append(molecule)
        if len(molecules) >= n:
            break
    return molecules


def normalize_mpc_prediction_list(value: list[Any], row: dict[str, Any]) -> list[str]:
    """清洗 MPC 预测列表：去重、过滤已知 partial molecules，并按 n 截断。"""
    n = row.get("n")
    if not isinstance(n, int) or n <= 0:
        return []
    known = {
        normalize_molecule(item)
        for item in row.get("partial_molecules", [])
        if normalize_molecule(item)
    }
    molecules: list[str] = []
    seen: set[str] = set()
    for item in value:
        molecule = str(item).strip()
        normalized = normalize_molecule(molecule)
        if not molecule or not normalized or normalized in known or normalized in seen:
            continue
        seen.add(normalized)
        molecules.append(molecule)
        if len(molecules) >= n:
            break
    return molecules


def has_successful_prediction(row: dict[str, Any], task: str) -> bool:
    if row.get("error"):
        return False
    if task == "mfp":
        return isinstance(row.get("predicted_food"), str) and bool(row["predicted_food"].strip())
    return isinstance(row.get("predicted_molecules"), list) and bool(row["predicted_molecules"])


def has_failed_prediction(row: dict[str, Any], task: str) -> bool:
    if row.get("error"):
        return True
    if row.get("id") is None:
        return True
    return not has_successful_prediction(row, task)


def read_existing_ids(path: Path, task: str, success_only: bool = False) -> set[str]:
    """resume 时读取已有 prediction id；retry-errors 时只把成功行视为完成。"""
    if not path.is_file():
        return set()
    existing: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict) or row.get("id") is None:
                continue
            if not success_only or has_successful_prediction(row, task):
                existing.add(str(row["id"]))
    return existing


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ZeroShotError(f"invalid prediction JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ZeroShotError(f"prediction row is not an object at {path}:{line_no}")
            rows.append(row)
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
    tmp_path.replace(path)


def build_prediction_for_row(row: dict[str, Any], task: str, llm_config: dict[str, str], evaluation: Any) -> tuple[dict[str, Any], bool]:
    row_id = row.get("id")
    try:
        if task == "mfp":
            content = evaluation.call_chat_completion(build_mfp_messages(row), llm_config)
            data = parse_json_object(content)
            predicted_food = validate_mfp_prediction(data or {})
            if predicted_food is None:
                return {"id": row_id, "predicted_food": "", "error": "parse_failed"}, False
            return {"id": row_id, "predicted_food": predicted_food}, True

        content = evaluation.call_chat_completion(build_mpc_messages(row), llm_config)
        # 正式 MPC prediction 保留任务必要字段；不默认保存 Stage 2B-fix 诊断字段。
        base_output = {
            "id": row_id,
            "task": row.get("task", "mpc"),
            "target_food": row.get("target_food") or row.get("food"),
            "partial_molecules": row.get("partial_molecules", []),
            "n": row.get("n"),
            "missing_molecules": row.get("missing_molecules", []),
        }
        parsed_molecules = parse_mpc_molecules(content)
        if parsed_molecules is None:
            return {**base_output, "predicted_molecules": [], "error": "parse_failed"}, False
        predicted_molecules = normalize_mpc_prediction_list(parsed_molecules, row)
        if not predicted_molecules:
            return {**base_output, "predicted_molecules": [], "error": "empty_prediction"}, False
        return {**base_output, "predicted_molecules": predicted_molecules}, True
    except Exception:
        if task == "mfp":
            return {"id": row_id, "predicted_food": "", "error": "parse_failed"}, False
        return {
            "id": row_id,
            "task": row.get("task", "mpc"),
            "target_food": row.get("target_food") or row.get("food"),
            "partial_molecules": row.get("partial_molecules", []),
            "n": row.get("n"),
            "missing_molecules": row.get("missing_molecules", []),
            "predicted_molecules": [],
            "error": "parse_failed",
        }, False


def retry_error_predictions(
    rows: list[dict[str, Any]],
    output_path: Path,
    task_rows_by_id: dict[str, dict[str, Any]],
    task: str,
    llm_config: dict[str, str],
    evaluation: Any,
) -> dict[str, Any]:
    failed_indexes = [
        index
        for index, row in enumerate(rows)
        if has_failed_prediction(row, task) and row.get("id") is not None
    ]
    success = 0
    failures = 0
    retried_ids: list[str] = []

    for index in failed_indexes:
        row_id = str(rows[index].get("id"))
        task_row = task_rows_by_id.get(row_id)
        if task_row is None:
            rows[index] = {"id": rows[index].get("id"), "predicted_molecules": [], "error": "missing_input_row"}
            failures += 1
            atomic_write_jsonl(output_path, rows)
            continue
        new_row, ok = build_prediction_for_row(task_row, task, llm_config, evaluation)
        rows[index] = new_row
        retried_ids.append(row_id)
        if ok:
            success += 1
        else:
            failures += 1
        atomic_write_jsonl(output_path, rows)

    return {
        "retry_targets": len(failed_indexes),
        "retried": len(retried_ids),
        "success": success,
        "failures": failures,
        "output_path": str(output_path),
    }


def generate_predictions(args: argparse.Namespace) -> int:
    evaluation = load_evaluation_module()
    evaluation.load_local_env_file()

    if not args.use_llm:
        raise ZeroShotError("--use-llm is required to allow real API calls")
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise ZeroShotError(f"input file not found: {input_path}")
    if not output_path.parent.is_dir():
        raise ZeroShotError(f"output parent directory does not exist: {output_path.parent}")
    if output_path.exists() and not args.resume:
        raise ZeroShotError(f"output file already exists; remove it or use --resume: {output_path}")
    if args.retry_errors and not args.resume:
        raise ZeroShotError("--retry-errors requires --resume")
    if args.retry_errors and not output_path.is_file():
        raise ZeroShotError("--retry-errors requires an existing output file")

    llm_config = evaluation.resolve_llm_config(args)
    evaluation.require_api_key(llm_config)

    rows = read_jsonl(input_path)
    if args.retry_errors:
        prediction_rows = read_prediction_rows(output_path)
        task_rows_by_id = {
            str(row["id"]): row for row in rows if row.get("id") is not None
        }
        result = retry_error_predictions(
            prediction_rows,
            output_path,
            task_rows_by_id,
            args.task,
            llm_config,
            evaluation,
        )
        print("ZERO_SHOT_RETRY_STATUS: PASS")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    existing_ids = read_existing_ids(output_path, args.task, success_only=False) if args.resume else set()
    total = len(rows)
    success = 0
    failures = 0
    skipped_existing = 0

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for row in rows:
            row_id = row.get("id")
            if str(row_id) in existing_ids:
                skipped_existing += 1
                continue
            out, ok = build_prediction_for_row(row, args.task, llm_config, evaluation)
            if ok:
                success += 1
            else:
                failures += 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

    status = "PASS" if total > 0 and output_path.is_file() else "FAIL"
    print(f"ZERO_SHOT_STATUS: {status}")
    print(
        json.dumps(
            {
                "task": args.task,
                "provider": llm_config["provider"],
                "model": llm_config["model"],
                "total": total,
                "existing_successful_predictions": len(existing_ids),
                "success": success,
                "failures": failures,
                "skipped_existing": skipped_existing,
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FoodPuzzle minimal zero-shot baseline")
    parser.add_argument("--task", choices=["mfp", "mpc"], required=True)
    parser.add_argument("--input", required=True, help="task JSONL input path")
    parser.add_argument("--output", required=True, help="prediction JSONL output path")
    parser.add_argument("--llm-provider", choices=["openai", "deepseek"], default="deepseek")
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--llm-base-url", help="override provider Chat Completions endpoint")
    parser.add_argument("--use-llm", action="store_true", help="allow real API calls")
    parser.add_argument("--resume", action="store_true", help="append only missing ids when output exists")
    parser.add_argument("--retry-errors", action="store_true", help="with --resume, retry only existing error or empty prediction rows")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return generate_predictions(args)
    except ZeroShotError as exc:
        print("ZERO_SHOT_STATUS: FAIL")
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
