#!/usr/bin/env python3
"""BM25 ICL baseline for FoodPuzzle MFP / MPC on reconstructed splits.

MPC 分支的正式 prompt 只在 test query 中使用 target_food、partial_molecules 和 n；
retrieved demonstrations 来自 labeled train split，可以包含 train missing_molecules。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BM25_K1 = 1.5
BM25_B = 0.75


class ICLError(Exception):
    """ICL baseline 中可预期的失败。"""


def load_evaluation_module() -> Any:
    """复用 evaluation.py 的 .env.local、provider、API 调用和 retry 逻辑。"""
    evaluation_path = Path(__file__).resolve().parent / "evaluation.py"
    spec = importlib.util.spec_from_file_location("foodpuzzle_evaluation", evaluation_path)
    if spec is None or spec.loader is None:
        raise ICLError(f"cannot import evaluation module from {evaluation_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ICLError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ICLError(f"JSONL row is not an object at {path}:{line_no}")
            if row.get("id") is None:
                raise ICLError(f"missing id at {path}:{line_no}")
            rows.append(row)
    return rows


def tokenize(text: str) -> list[str]:
    """BM25 tokenizer：小写，按非字母数字切分，保留数字片段。"""
    return [token for token in re.split(r"[^0-9A-Za-z]+", text.lower()) if token]


def normalize_molecule(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def molecule_text(row: dict[str, Any]) -> str:
    molecules = row.get("molecules")
    if not isinstance(molecules, list):
        return ""
    return " ".join(str(molecule) for molecule in molecules)


def mpc_retrieval_text(row: dict[str, Any]) -> str:
    """MPC BM25 query/index 文本：只使用推理时可见字段，不使用当前样本 gold。"""
    target_food = row.get("target_food") or row.get("food") or ""
    partial_molecules = row.get("partial_molecules")
    if not isinstance(partial_molecules, list):
        partial_molecules = []
    n = row.get("n")
    return " ".join(
        [
            str(target_food),
            str(n) if isinstance(n, int) else "",
            " ".join(str(molecule) for molecule in partial_molecules),
        ]
    )


def retrieval_text(row: dict[str, Any], task: str) -> str:
    if task == "mfp":
        return molecule_text(row)
    if task == "mpc":
        return mpc_retrieval_text(row)
    raise ICLError(f"unknown task: {task}")


class BM25Index:
    """轻量纯 Python BM25，实现 deterministic top-k retrieval。"""

    def __init__(self, rows: list[dict[str, Any]], task: str, k1: float = BM25_K1, b: float = BM25_B):
        self.rows = rows
        self.task = task
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(retrieval_text(row, task)) for row in rows]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        doc_freq: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                doc_freq[token] += 1
        total_docs = len(rows)
        self.idf = {
            token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        freqs = self.term_freqs[doc_idx]
        doc_len = self.doc_len[doc_idx]
        for token in query_tokens:
            tf = freqs.get(token, 0)
            if tf == 0:
                continue
            idf = self.idf.get(token, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def retrieve(self, query_row: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        query_tokens = tokenize(retrieval_text(query_row, self.task))
        scored = []
        for idx, row in enumerate(self.rows):
            score = self.score(query_tokens, idx)
            # 分数相同用 train 文件顺序稳定排序。
            scored.append((score, idx, row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        retrieved = []
        for rank, (score, _, row) in enumerate(scored[:top_k], 1):
            retrieved.append(
                {
                    "id": row["id"],
                    "rank": rank,
                    "score": score,
                    "actual_food": row.get("actual_food"),
                    "target_food": row.get("target_food", row.get("food")),
                    "row": row,
                }
            )
        return retrieved


def parse_json_object(content: str) -> dict[str, Any] | None:
    try:
        data = json.loads(content)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def validate_prediction(data: dict[str, Any]) -> str | None:
    value = data.get("predicted_food")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


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

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    return None


def parse_text_molecule_list(content: str) -> list[str]:
    """解析编号列表或 bullet list；只处理模型输出，不接触 gold。"""
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
        return value if isinstance(value, list) else None
    if isinstance(data, list):
        return data

    text_list = parse_text_molecule_list(content)
    if text_list:
        return text_list
    return None


def format_molecules(row: dict[str, Any]) -> str:
    molecules = row.get("molecules")
    if not isinstance(molecules, list):
        return ""
    return ", ".join(str(molecule) for molecule in molecules)


def format_partial_molecules(row: dict[str, Any]) -> str:
    molecules = row.get("partial_molecules")
    if not isinstance(molecules, list):
        return ""
    return ", ".join(str(molecule) for molecule in molecules)


def format_missing_molecules(row: dict[str, Any]) -> str:
    molecules = row.get("missing_molecules")
    if not isinstance(molecules, list):
        return ""
    return ", ".join(str(molecule) for molecule in molecules)


def build_prompt_messages(query_row: dict[str, Any], retrieved: list[dict[str, Any]]) -> list[dict[str, str]]:
    """构造 MFP BM25 ICL prompt；query 不包含 test actual_food。"""
    demo_blocks = []
    for item in retrieved:
        row = item["row"]
        demo_blocks.append(
            f"Food: {row.get('actual_food')}\n"
            f"Molecules: {format_molecules(row)}"
        )
    prompt = (
        "FoodPuzzle Molecular Food Prediction task.\n"
        "Given flavor molecules, predict the most likely food source.\n"
        "Below are retrieved examples from the training split. They are references and may not be identical to the query.\n\n"
        "Examples:\n"
        + "\n\n".join(demo_blocks)
        + "\n\nQuery:\n"
        f"Molecules: {format_molecules(query_row)}\n\n"
        "Return a concise free-text food source or food category.\n"
        "Do not explain your answer. Do not use Markdown. Return JSON only.\n"
        'Required JSON format: {"predicted_food": "<free-text food source or category>"}'
    )
    return [
        {"role": "system", "content": "You are a strict FoodPuzzle ICL baseline that returns only valid JSON."},
        {"role": "user", "content": prompt},
    ]


def build_mpc_prompt_messages(query_row: dict[str, Any], retrieved: list[dict[str, Any]]) -> list[dict[str, str]]:
    """构造 MPC BM25 ICL prompt；query 只包含 target_food、partial_molecules、n。"""
    target_food = query_row.get("target_food") or query_row.get("food") or ""
    n = query_row.get("n")
    if not isinstance(target_food, str) or not target_food.strip():
        raise ICLError("invalid_target_food")
    if not isinstance(query_row.get("partial_molecules"), list):
        raise ICLError("invalid_partial_molecules")
    if not isinstance(n, int) or n <= 0:
        raise ICLError("invalid_n")

    demo_blocks = []
    for item in retrieved:
        row = item["row"]
        # demonstrations 来自 train split，允许包含 train gold missing_molecules。
        demo_blocks.append(
            f"Food: {row.get('target_food') or row.get('food')}\n"
            f"Known molecules: {format_partial_molecules(row)}\n"
            f"Number of missing molecules: {row.get('n')}\n"
            f"Missing molecules: {format_missing_molecules(row)}"
        )

    prompt = (
        "FoodPuzzle Missing Molecule Prediction task.\n"
        "Given a food item, known molecules already associated with that food, and n, "
        "predict the missing flavor molecules.\n"
        "Below are retrieved labeled examples from the training split.\n\n"
        "Examples:\n"
        + "\n\n".join(demo_blocks)
        + "\n\nQuery:\n"
        f"Food: {target_food.strip()}\n"
        f"Known molecules: {format_partial_molecules(query_row)}\n"
        f"Number of missing molecules to predict: {n}\n\n"
        f"Predict exactly {n} molecules if possible. If exact {n} is difficult, return the best possible list.\n"
        "Do not include molecules already listed in Known molecules.\n"
        "Do not output explanations. Do not use Markdown. Return only valid JSON.\n"
        'Required JSON format: {"predicted_molecules": ["molecule name 1", "molecule name 2"]}'
    )
    return [
        {"role": "system", "content": "You are a strict FoodPuzzle BM25 ICL baseline that returns only valid JSON."},
        {"role": "user", "content": prompt},
    ]


def normalize_mpc_prediction_list(value: list[Any], row: dict[str, Any]) -> list[str]:
    """清洗 MPC 预测列表：去重、过滤 query partial molecules，并按 n 截断。"""
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


def read_existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    existing: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and row.get("id") is not None:
                existing.add(str(row["id"]))
    return existing


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def run_icl(args: argparse.Namespace) -> int:
    if not args.use_llm:
        raise ICLError("--use-llm is required to allow real API calls")
    if args.top_k <= 0:
        raise ICLError("--top-k must be positive")

    train_path = Path(args.train)
    test_path = Path(args.test)
    output_path = Path(args.output)
    metadata_path = Path(args.retrieval_metadata)
    if not train_path.is_file():
        raise ICLError(f"train file not found: {train_path}")
    if not test_path.is_file():
        raise ICLError(f"test file not found: {test_path}")
    if not output_path.parent.is_dir():
        raise ICLError(f"output parent directory does not exist: {output_path.parent}")
    if not metadata_path.parent.is_dir():
        raise ICLError(f"retrieval metadata parent directory does not exist: {metadata_path.parent}")
    if output_path.exists() and not args.resume:
        raise ICLError(f"output file already exists; remove it or use --resume: {output_path}")
    if metadata_path.exists() and not args.resume:
        raise ICLError(f"retrieval metadata already exists; remove it or use --resume: {metadata_path}")

    evaluation = load_evaluation_module()
    evaluation.load_local_env_file()
    llm_config = evaluation.resolve_llm_config(args)
    evaluation.require_api_key(llm_config)

    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(test_path)
    train_ids = {str(row["id"]) for row in train_rows}
    test_ids = {str(row["id"]) for row in test_rows}
    if train_ids & test_ids:
        raise ICLError("train and test split overlap")

    index = BM25Index(train_rows, args.task)
    existing_predictions = read_existing_ids(output_path) if args.resume else set()
    existing_metadata = read_existing_ids(metadata_path) if args.resume else set()

    total = len(test_rows)
    skipped = 0
    newly_generated = 0
    success = 0
    failures = 0

    for row in test_rows:
        row_id = str(row["id"])
        if row_id in existing_predictions:
            skipped += 1
            continue

        retrieved = index.retrieve(row, args.top_k)
        if row_id not in existing_metadata:
            # retrieval_metadata 是正式 ICL 可追溯输出，不是调试字段。
            append_jsonl(
                metadata_path,
                {
                    "id": row["id"],
                    "retrieved": [
                        {
                            "id": item["id"],
                            "rank": item["rank"],
                            "score": item["score"],
                            "actual_food": item["actual_food"],
                            "target_food": item["target_food"],
                        }
                        for item in retrieved
                    ],
                },
            )

        try:
            if args.task == "mfp":
                content = evaluation.call_chat_completion(build_prompt_messages(row, retrieved), llm_config)
                data = parse_json_object(content)
                predicted_food = validate_prediction(data or {})
                if predicted_food is None:
                    raise ICLError("parse_failed")
                out = {"id": row["id"], "predicted_food": predicted_food}
            else:
                content = evaluation.call_chat_completion(build_mpc_prompt_messages(row, retrieved), llm_config)
                parsed_molecules = parse_mpc_molecules(content)
                if parsed_molecules is None:
                    raise ICLError("parse_failed")
                predicted_molecules = normalize_mpc_prediction_list(parsed_molecules, row)
                if not predicted_molecules:
                    raise ICLError("empty_prediction")
                # MPC formal prediction 保留 evaluation 必要字段；不暴露 full molecule set。
                out = {
                    "id": row["id"],
                    "task": row.get("task", "MPC"),
                    "target_food": row.get("target_food") or row.get("food"),
                    "partial_molecules": row.get("partial_molecules", []),
                    "n": row.get("n"),
                    "missing_molecules": row.get("missing_molecules", []),
                    "predicted_molecules": predicted_molecules,
                }
            success += 1
        except Exception as exc:
            if args.task == "mfp":
                out = {"id": row["id"], "predicted_food": "", "error": "parse_failed"}
            else:
                out = {
                    "id": row["id"],
                    "task": row.get("task", "MPC"),
                    "target_food": row.get("target_food") or row.get("food"),
                    "partial_molecules": row.get("partial_molecules", []),
                    "n": row.get("n"),
                    "missing_molecules": row.get("missing_molecules", []),
                    "predicted_molecules": [],
                    "error": "empty_prediction" if str(exc) == "empty_prediction" else "parse_failed",
                }
            failures += 1

        append_jsonl(output_path, out)
        newly_generated += 1

    print("ICL_STATUS: PASS")
    print(
        json.dumps(
            {
                "task": args.task,
                "method": "bm25_icl",
                "provider": llm_config["provider"],
                "model": llm_config["model"],
                "top_k": args.top_k,
                "total": total,
                "existing_predictions": len(existing_predictions),
                "newly_generated": newly_generated,
                "skipped": skipped,
                "success": success,
                "failures": failures,
                "output_path": str(output_path),
                "retrieval_metadata_path": str(metadata_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FoodPuzzle BM25 ICL baseline")
    parser.add_argument("--task", choices=["mfp", "mpc"], required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retrieval-metadata", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--llm-provider", choices=["openai", "deepseek"], default="deepseek")
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--llm-base-url", help="override provider Chat Completions endpoint")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        return run_icl(args)
    except ICLError as exc:
        print("ICL_STATUS: FAIL")
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
