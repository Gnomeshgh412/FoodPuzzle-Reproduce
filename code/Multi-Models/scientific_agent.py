#!/usr/bin/env python3
"""Scientific Agent baseline for FoodPuzzle MFP / MPC on reconstructed splits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class AgentError(Exception):
    """Scientific Agent 中可预期的失败。"""


def load_evaluation_module() -> Any:
    """复用 evaluation.py 的 .env.local、provider、API 调用和 JSON fallback 逻辑。"""
    evaluation_path = Path(__file__).resolve().parent / "evaluation.py"
    spec = importlib.util.spec_from_file_location("foodpuzzle_evaluation", evaluation_path)
    if spec is None or spec.loader is None:
        raise AgentError(f"cannot import evaluation module from {evaluation_path}")
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
                raise AgentError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise AgentError(f"JSONL row is not object at {path}:{line_no}")
            if row.get("id") is None:
                raise AgentError(f"missing id at {path}:{line_no}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and row.get("id") is not None:
                ids.add(str(row["id"]))
    return ids


def read_success_prediction_ids(path: Path) -> set[str]:
    """resume 时只跳过已有成功 prediction；API error 空预测允许后续重试。"""
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if (
                isinstance(row, dict)
                and row.get("id") is not None
                and isinstance(row.get("predicted_food"), str)
                and row["predicted_food"].strip()
                and not row.get("error")
            ):
                ids.add(str(row["id"]))
    return ids


def read_success_agent_prediction_ids(path: Path, task: str) -> set[str]:
    """按任务读取成功 prediction id；MPC 成功字段为非空 predicted_molecules。"""
    if task == "mfp":
        return read_success_prediction_ids(path)
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if (
                isinstance(row, dict)
                and row.get("id") is not None
                and isinstance(row.get("predicted_molecules"), list)
                and row["predicted_molecules"]
                and not row.get("error")
            ):
                ids.add(str(row["id"]))
    return ids


def read_success_hypotheses_ids(path: Path) -> set[str]:
    """只把无 error 且有 reviewer_output 的 hypotheses metadata 视为完成。"""
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if (
                isinstance(row, dict)
                and row.get("id") is not None
                and not row.get("error")
                and isinstance(row.get("reviewer_output"), dict)
            ):
                ids.add(str(row["id"]))
    return ids


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_official_evidence(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """加载仓库内 official Task1 evidence pickle；缺失时停止，不切换到 local evidence-lite。"""
    if not path.is_file():
        raise AgentError(f"official evidence pickle not found: {path}")
    info = {
        "filename": path.name,
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise AgentError("official evidence pickle top-level object is not dict")
    evidence: dict[str, list[str]] = {}
    for key, value in obj.items():
        if isinstance(key, str) and isinstance(value, list):
            evidence[key.strip().lower()] = [str(item) for item in value]
    info.update(
        {
            "structure": "dict[str, list[str]]",
            "top_type": type(obj).__name__,
            "top_len": len(obj),
            "normalized_key_count": len(evidence),
        }
    )
    return evidence, info


def normalize_text(value: Any) -> str:
    """对 food / molecule 文本做保守归一，供 MPC evidence lookup 和去重使用。"""
    text = str(value or "").lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_mpc_evidence(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """加载 MPC food-centered evidence；期望 dict key 可归一到 target_food。"""
    if not path.is_file():
        raise AgentError(f"MPC evidence pickle not found: {path}")
    info = {
        "filename": path.name,
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise AgentError("MPC evidence pickle top-level object is not dict")
    evidence: dict[str, list[str]] = {}
    value_type_counts: Counter[str] = Counter()
    for key, value in obj.items():
        norm_key = normalize_text(key)
        if not norm_key:
            continue
        snippets: list[str] = []
        value_type_counts[type(value).__name__] += 1
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    snippets.append(json.dumps(item, ensure_ascii=False))
                else:
                    snippets.append(str(item))
        elif isinstance(value, dict):
            snippets.append(json.dumps(value, ensure_ascii=False))
        elif value is not None:
            snippets.append(str(value))
        evidence[norm_key] = snippets
    info.update(
        {
            "structure": "dict[food_key, evidence_payload]",
            "top_type": type(obj).__name__,
            "top_len": len(obj),
            "normalized_key_count": len(evidence),
            "value_type_counts": dict(value_type_counts),
            "first_keys": [str(key) for key in list(obj.keys())[:10]],
        }
    )
    return evidence, info


def format_molecules(row: dict[str, Any], max_items: int | None = None) -> str:
    molecules = row.get("molecules")
    if not isinstance(molecules, list):
        return ""
    items = [str(molecule) for molecule in molecules]
    if max_items is not None:
        items = items[:max_items]
    return ", ".join(items)


def format_partial_molecules(row: dict[str, Any], max_items: int | None = None) -> str:
    molecules = row.get("partial_molecules")
    if not isinstance(molecules, list):
        return ""
    items = [str(molecule) for molecule in molecules]
    if max_items is not None:
        items = items[:max_items]
    return ", ".join(items)


def format_missing_molecules(row: dict[str, Any], max_items: int | None = None) -> str:
    molecules = row.get("missing_molecules")
    if not isinstance(molecules, list):
        return ""
    items = [str(molecule) for molecule in molecules]
    if max_items is not None:
        items = items[:max_items]
    return ", ".join(items)


def mpc_retrieval_text(row: dict[str, Any]) -> str:
    """MPC BM25 retrieval 只使用推理时可见字段，不使用当前样本 gold。"""
    target_food = row.get("target_food") or row.get("food") or ""
    n = row.get("n")
    return " ".join(
        [
            str(target_food),
            str(n) if isinstance(n, int) else "",
            format_partial_molecules(row),
        ]
    )


class MPCBM25Index:
    """MPC Agent 用轻量 BM25，从 labeled train split 检索 demonstrations。"""

    def __init__(self, rows: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.rows = rows
        self.k1 = k1
        self.b = b
        self.doc_tokens = [self.tokenize(mpc_retrieval_text(row)) for row in rows]
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

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [token for token in re.split(r"[^0-9A-Za-z]+", text.lower()) if token]

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

    def retrieve(self, row: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        query_tokens = self.tokenize(mpc_retrieval_text(row))
        scored = []
        for idx, train_row in enumerate(self.rows):
            scored.append((self.score(query_tokens, idx), idx, train_row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "id": train_row["id"],
                "rank": rank,
                "score": score,
                "target_food": train_row.get("target_food", train_row.get("food")),
                "row": train_row,
            }
            for rank, (score, _, train_row) in enumerate(scored[:top_k], 1)
        ]


def build_train_idf(train_rows: list[dict[str, Any]]) -> dict[str, float]:
    """只从 reconstructed train split 统计 molecule DF/IDF，避免使用 test label。"""
    df: Counter[str] = Counter()
    for row in train_rows:
        molecules = row.get("molecules")
        if isinstance(molecules, list):
            df.update({str(molecule).strip().lower() for molecule in molecules})
    total = len(train_rows)
    return {molecule: math.log(1 + (total + 1) / (freq + 1)) for molecule, freq in df.items()}


def select_starting_molecules(
    row: dict[str, Any],
    idf: dict[str, float],
    evidence: dict[str, list[str]],
    count: int,
) -> list[str]:
    """优先选有 official evidence 的 query molecules，再按 train IDF 近似信息量排序。"""
    molecules = row.get("molecules")
    if not isinstance(molecules, list):
        return []
    scored: list[tuple[tuple[int, float, int], str]] = []
    for order, molecule in enumerate(molecules):
        text = str(molecule).strip()
        norm = text.lower()
        has_evidence = 1 if evidence.get(norm) else 0
        scored.append(((has_evidence, idf.get(norm, 0.0), -order), text))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, molecule in scored:
        norm = molecule.lower()
        if norm in seen:
            continue
        selected.append(molecule)
        seen.add(norm)
        if len(selected) >= count:
            break
    return selected


def name_pattern(name: str) -> re.Pattern[str]:
    """对当前答案做大小写不敏感的边界匹配，避免匹配到长词内部。"""
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", re.IGNORECASE)


def mask_answer(text: str, actual_food: str) -> tuple[str, int]:
    if not actual_food:
        return text, 0
    return name_pattern(actual_food).subn("[MASKED_FOOD]", text)


def count_answer_hits(text: str, actual_food: str) -> int:
    if not actual_food:
        return 0
    return len(name_pattern(actual_food).findall(text))


def build_evidence_blocks(
    row: dict[str, Any],
    selected_molecules: list[str],
    evidence: dict[str, list[str]],
    max_snippets_per_molecule: int,
) -> tuple[list[dict[str, Any]], str, int]:
    """构造 answer-masked official evidence；actual_food 仅用于 masking 和 metadata。"""
    actual_food = str(row.get("actual_food") or "")
    blocks: list[dict[str, Any]] = []
    prompt_blocks: list[str] = []
    hits_after_mask = 0
    for molecule in selected_molecules:
        raw_snippets = evidence.get(molecule.strip().lower(), [])
        used_raw = raw_snippets[:max_snippets_per_molecule]
        snippets: list[str] = []
        masked_occurrences = 0
        for snippet in used_raw:
            masked, count = mask_answer(str(snippet), actual_food)
            masked_occurrences += count
            hits_after_mask += count_answer_hits(masked, actual_food)
            snippets.append(masked)
        lines = "\n".join(f"- {snippet}" for snippet in snippets) if snippets else "- no_evidence"
        prompt_blocks.append(f"Molecule: {molecule}\n{lines}")
        blocks.append(
            {
                "molecule": molecule,
                "raw_snippet_count": len(raw_snippets),
                "used_snippet_count": len(snippets),
                "masked_occurrences": masked_occurrences,
                "snippets": snippets,
            }
        )
    return blocks, "\n\n".join(prompt_blocks), hits_after_mask


def load_retrieval_metadata(path: Path) -> dict[str, list[dict[str, Any]]]:
    metadata: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        retrieved = row.get("retrieved")
        if isinstance(retrieved, list):
            metadata[str(row.get("id"))] = [item for item in retrieved if isinstance(item, dict)]
    return metadata


def resolve_demos(
    row_id: str,
    retrieval_metadata: dict[str, list[dict[str, Any]]],
    train_by_id: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    demos: list[dict[str, Any]] = []
    for item in retrieval_metadata.get(row_id, [])[:top_k]:
        demo_id = str(item.get("id"))
        if demo_id not in train_by_id:
            continue
        merged = dict(item)
        merged["row"] = train_by_id[demo_id]
        demos.append(merged)
    return demos


def build_demo_prompt(demos: list[dict[str, Any]], max_molecules: int | None = None) -> str:
    blocks = []
    for idx, demo in enumerate(demos, 1):
        row = demo["row"]
        blocks.append(
            f"Example {idx}:\n"
            f"Food: {row.get('actual_food')}\n"
            f"Molecules: {format_molecules(row, max_items=max_molecules)}"
        )
    return "\n\n".join(blocks)


def build_scientist_messages(
    row: dict[str, Any],
    selected_molecules: list[str],
    evidence_text: str,
    demos: list[dict[str, Any]],
) -> list[dict[str, str]]:
    selected_lines = "\n".join(f"{idx}. {molecule}" for idx, molecule in enumerate(selected_molecules, 1))
    prompt = (
        "Task:\n"
        "You are a flavor scientist. Given a set of flavor molecules, infer the most likely food source.\n\n"
        "FoodPuzzle Data Input:\n"
        f"Molecules:\n{format_molecules(row)}\n\n"
        "Selected Starting Molecules:\n"
        f"{selected_lines}\n\n"
        "Retrieved Evidence:\n"
        f"{evidence_text}\n\n"
        "BM25 Demonstrations from Training Set:\n"
        f"{build_demo_prompt(demos)}\n\n"
        "Instruction:\n"
        "Generate three candidate hypotheses for the most likely food source.\n"
        "Each hypothesis should include:\n"
        "- predicted_food\n"
        "- short rationale grounded in evidence and demonstrations\n\n"
        "Do not output Markdown.\n"
        "Do not output extra text.\n"
        "Output JSON only:\n"
        "{\n"
        '  "hypotheses": [\n'
        '    {"predicted_food": "...", "rationale": "..."},\n'
        '    {"predicted_food": "...", "rationale": "..."},\n'
        '    {"predicted_food": "...", "rationale": "..."}\n'
        "  ]\n"
        "}"
    )
    return [
        {"role": "system", "content": "You are a FoodPuzzle Scientist agent that returns only valid JSON."},
        {"role": "user", "content": prompt},
    ]


def format_hypotheses(hypotheses: list[dict[str, str]]) -> str:
    if not hypotheses:
        return "No valid hypotheses were generated."
    blocks = []
    for idx, hypothesis in enumerate(hypotheses, 1):
        blocks.append(
            f"{idx}. predicted_food: {hypothesis.get('predicted_food', '')}\n"
            f"   rationale: {hypothesis.get('rationale', '')}"
        )
    return "\n".join(blocks)


def build_reviewer_messages(
    row: dict[str, Any],
    evidence_text: str,
    demos: list[dict[str, Any]],
    hypotheses: list[dict[str, str]],
) -> list[dict[str, str]]:
    prompt = (
        "Task:\n"
        "You are a scientific reviewer. Review the candidate hypotheses and select the best final prediction.\n\n"
        "FoodPuzzle Data Input:\n"
        f"Molecules:\n{format_molecules(row)}\n\n"
        "Retrieved Evidence Summary:\n"
        f"{evidence_text}\n\n"
        "BM25 Demonstrations from Training Set:\n"
        f"{build_demo_prompt(demos)}\n\n"
        "Scientist Hypotheses:\n"
        f"{format_hypotheses(hypotheses)}\n\n"
        "Instruction:\n"
        "Select the best hypothesis based on consistency with the evidence, molecules, and demonstrations.\n"
        "Do not invent unsupported evidence.\n"
        "Return a concise final food source or category.\n\n"
        "Do not output Markdown.\n"
        "Do not output extra text.\n"
        "Output JSON only:\n"
        "{\n"
        '  "predicted_food": "...",\n'
        '  "selected_hypothesis_index": 1,\n'
        '  "review_reason": "..."\n'
        "}"
    )
    return [
        {"role": "system", "content": "You are a FoodPuzzle Reviewer agent that returns only valid JSON."},
        {"role": "user", "content": prompt},
    ]


def parse_json_object(content: str) -> dict[str, Any] | None:
    data = parse_json_value(content)
    return data if isinstance(data, dict) else None


def parse_json_value(content: str) -> Any | None:
    """解析 JSON / code fence JSON；MPC Agent 输出经常会被模型包进 ```json。"""
    text = str(content or "").strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    object_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(1).strip())
    array_match = re.search(r"(\[.*\])", text, flags=re.DOTALL)
    if array_match:
        candidates.append(array_match.group(1).strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def normalize_mpc_molecule_list(value: Any) -> list[str]:
    """把模型输出中的 molecule list 规整成去重后的字符串列表。"""
    if not isinstance(value, list):
        return []
    molecules: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        norm = normalize_text(text)
        if norm in seen:
            continue
        seen.add(norm)
        molecules.append(text)
    return molecules


def normalize_mpc_agent_final_molecules(
    molecules: list[str],
    row: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """让 MPC Agent final output 符合任务协议。

    这里只使用推理时可见的 partial_molecules 和 n：
    - 过滤已知 partial molecules，避免把输入分子当作 missing prediction；
    - 超过 n 时截断到前 n 个；
    - 不使用 gold missing_molecules，也不改变后续 evaluation metric。
    """
    partial_norm = {
        normalize_text(molecule)
        for molecule in row.get("partial_molecules", [])
        if str(molecule).strip()
    }
    n = row.get("n")
    limit = n if isinstance(n, int) and n >= 0 else None
    normalized: list[str] = []
    seen: set[str] = set()
    removed_partial = 0
    removed_duplicate_or_empty = 0
    for molecule in molecules:
        text = str(molecule).strip()
        norm = normalize_text(text)
        if not text or not norm:
            removed_duplicate_or_empty += 1
            continue
        if norm in partial_norm:
            removed_partial += 1
            continue
        if norm in seen:
            removed_duplicate_or_empty += 1
            continue
        seen.add(norm)
        normalized.append(text)
    before_truncate = len(normalized)
    truncated = False
    if limit is not None and len(normalized) > limit:
        normalized = normalized[:limit]
        truncated = True
    metadata = {
        "final_predicted_count_before_normalization": len(molecules),
        "final_predicted_count_after_normalization": len(normalized),
        "removed_partial_molecule_count": removed_partial,
        "removed_duplicate_or_empty_count": removed_duplicate_or_empty,
        "truncated_to_n": truncated,
        "count_before_truncation": before_truncate,
    }
    return normalized, metadata


def parse_text_molecule_list(content: str) -> list[str]:
    """兜底解析编号列表 / bullet list，不从 prompt 或 gold label 中抽取内容。"""
    molecules: list[str] = []
    for raw_line in str(content or "").splitlines():
        line = raw_line.strip()
        match = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", line)
        if not match:
            continue
        item = match.group(1).strip().strip('",')
        if item and not item.lower().startswith(("predicted", "molecules")):
            molecules.append(item)
    return normalize_mpc_molecule_list(molecules)


def parse_mpc_molecules(content: str) -> list[str] | None:
    """解析 MPC predicted_molecules，支持 JSON object / array / code fence / list。"""
    data = parse_json_value(content)
    if isinstance(data, dict):
        for key in ["predicted_molecules", "molecules", "missing_molecules"]:
            parsed = normalize_mpc_molecule_list(data.get(key))
            if parsed:
                return parsed
        hypotheses = data.get("hypotheses")
        if isinstance(hypotheses, list):
            for item in hypotheses:
                if isinstance(item, dict):
                    parsed = normalize_mpc_molecule_list(item.get("predicted_molecules"))
                    if parsed:
                        return parsed
    if isinstance(data, list):
        parsed = normalize_mpc_molecule_list(data)
        if parsed:
            return parsed
    parsed_text = parse_text_molecule_list(content)
    return parsed_text if parsed_text else None


def parse_hypotheses(content: str) -> list[dict[str, str]] | None:
    data = parse_json_object(content)
    if data is None or not isinstance(data.get("hypotheses"), list):
        return None
    parsed: list[dict[str, str]] = []
    for item in data["hypotheses"]:
        if not isinstance(item, dict):
            continue
        predicted = item.get("predicted_food")
        rationale = item.get("rationale")
        if isinstance(predicted, str) and predicted.strip():
            parsed.append(
                {
                    "predicted_food": predicted.strip(),
                    "rationale": str(rationale or "").strip(),
                }
            )
    return parsed if parsed else None


def parse_reviewer_output(content: str) -> dict[str, Any] | None:
    data = parse_json_object(content)
    if data is None:
        return None
    predicted = data.get("predicted_food")
    if not isinstance(predicted, str) or not predicted.strip():
        return None
    selected = data.get("selected_hypothesis_index")
    return {
        "predicted_food": predicted.strip(),
        "selected_hypothesis_index": selected if isinstance(selected, int) else None,
        "review_reason": str(data.get("review_reason") or "").strip(),
    }


def build_mpc_demo_prompt(demos: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, demo in enumerate(demos, 1):
        row = demo["row"]
        blocks.append(
            f"Example {idx}:\n"
            f"Target food: {row.get('target_food')}\n"
            f"Known molecules: {format_partial_molecules(row)}\n"
            f"Number of missing molecules: {row.get('n')}\n"
            f"Gold missing molecules: {format_missing_molecules(row)}"
        )
    return "\n\n".join(blocks)


def build_mpc_evidence_blocks(
    row: dict[str, Any],
    evidence: dict[str, list[str]],
    max_snippets: int,
) -> tuple[dict[str, Any], str]:
    """MPC evidence 只按 target_food 查本地 food-centered evidence，不从 test label 构造。"""
    target_food = str(row.get("target_food") or row.get("food") or "")
    evidence_key = normalize_text(target_food)
    snippets = evidence.get(evidence_key, [])
    used = snippets[:max_snippets]
    evidence_text = "\n".join(f"- {snippet}" for snippet in used) if used else "- no_evidence"
    metadata = {
        "evidence_key": evidence_key,
        "target_food": target_food,
        "evidence_found": bool(snippets),
        "raw_evidence_count": len(snippets),
        "used_evidence_count": len(used),
    }
    return metadata, evidence_text


def build_mpc_scientist_messages(
    row: dict[str, Any],
    evidence_text: str,
    demos: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """MPC Scientist prompt 只暴露 target_food、partial_molecules、n、evidence 和 train demos。

    正式 MPC Agent 鼓励 exactly n when possible，但 never exceed n，
    以贴近 MPC 输入中 n 表示缺失分子数量的任务定义。
    """
    n = row.get("n")
    prompt = (
        "Task:\n"
        "You are a flavor chemistry scientist. Infer missing flavor molecules for a target food.\n\n"
        "MPC Test Input:\n"
        f"Target food: {row.get('target_food')}\n"
        f"Known molecules: {format_partial_molecules(row)}\n"
        f"Number of missing molecules to predict: {n}\n\n"
        "Retrieved Food-Centered Evidence:\n"
        f"{evidence_text}\n\n"
        "BM25 Demonstrations from Labeled Training Set:\n"
        f"{build_mpc_demo_prompt(demos)}\n\n"
        "Instruction:\n"
        "Generate exactly three candidate hypotheses. Each hypothesis should contain a list of likely "
        "missing flavor molecules for the target food and a short rationale.\n"
        f"Each hypothesis predicted_molecules list should contain exactly {n} molecule names when possible.\n"
        f"If uncertain, still provide the best {n} candidate missing molecules.\n"
        f"Never output more than {n} molecule names in any hypothesis.\n"
        "Do not include molecules already listed in Known molecules.\n"
        "Do not put explanations, evidence text, or long prose inside predicted_molecules.\n"
        "Each predicted_molecules item must be a molecule common name string.\n"
        "Do not output Markdown.\n"
        "Do not output extra text.\n"
        "Output JSON only:\n"
        "{\n"
        '  "hypotheses": [\n'
        '    {"predicted_molecules": ["..."], "rationale": "..."},\n'
        '    {"predicted_molecules": ["..."], "rationale": "..."},\n'
        '    {"predicted_molecules": ["..."], "rationale": "..."}\n'
        "  ]\n"
        "}"
    )
    return [
        {"role": "system", "content": "You are a FoodPuzzle MPC Scientist agent that returns only valid JSON."},
        {"role": "user", "content": prompt},
    ]


def parse_mpc_hypotheses(content: str) -> list[dict[str, Any]] | None:
    data = parse_json_object(content)
    if data is None or not isinstance(data.get("hypotheses"), list):
        return None
    hypotheses: list[dict[str, Any]] = []
    for item in data["hypotheses"]:
        if not isinstance(item, dict):
            continue
        molecules = normalize_mpc_molecule_list(item.get("predicted_molecules"))
        if not molecules:
            continue
        hypotheses.append(
            {
                "predicted_molecules": molecules,
                "rationale": str(item.get("rationale") or "").strip(),
            }
        )
    return hypotheses if hypotheses else None


def format_mpc_hypotheses(hypotheses: list[dict[str, Any]]) -> str:
    if not hypotheses:
        return "No valid hypotheses were generated."
    blocks = []
    for idx, hypothesis in enumerate(hypotheses, 1):
        blocks.append(
            f"{idx}. predicted_molecules: {', '.join(hypothesis.get('predicted_molecules', []))}\n"
            f"   rationale: {hypothesis.get('rationale', '')}"
        )
    return "\n".join(blocks)


def build_mpc_reviewer_messages(
    row: dict[str, Any],
    evidence_text: str,
    demos: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    reviewer_evidence_mode: str,
) -> list[dict[str, str]]:
    """构造 MPC Reviewer prompt。

    正式 MPC Agent 默认不把 raw evidence 直接交给 Reviewer：
    Scientist 使用 evidence 生成 hypotheses，Reviewer 主要基于 task input、train demos
    和 hypotheses 做最终选择。保留 full 模式只用于显式 ablation / 对照。
    """
    n = row.get("n")
    evidence_block = ""
    if reviewer_evidence_mode == "full":
        evidence_block = (
            "Retrieved Food-Centered Evidence:\n"
            f"{evidence_text}\n\n"
        )
    prompt = (
        "Task:\n"
        "You are a scientific reviewer. Select the best final missing-molecule prediction.\n\n"
        "MPC Test Input:\n"
        f"Target food: {row.get('target_food')}\n"
        f"Known molecules: {format_partial_molecules(row)}\n"
        f"Number of missing molecules to predict: {n}\n\n"
        f"{evidence_block}"
        "BM25 Demonstrations from Labeled Training Set:\n"
        f"{build_mpc_demo_prompt(demos)}\n\n"
        "Scientist Hypotheses:\n"
        f"{format_mpc_hypotheses(hypotheses)}\n\n"
        "Instruction:\n"
        "Select one of the three hypotheses or synthesize a final list from them.\n"
        f"The final predicted_molecules list should contain exactly {n} molecule names when possible.\n"
        f"If uncertain, still provide the best {n} candidate missing molecules.\n"
        f"Never output more than {n} molecule names.\n"
        "Do not include molecules already listed in Known molecules.\n"
        "Do not output evidence, demonstrations, reasoning, markdown, or any text outside the final JSON.\n"
        "Each predicted_molecules item must be a molecule common name string, not a sentence or paragraph.\n"
        "Do not output Markdown.\n"
        "Do not output extra text.\n"
        "Output JSON only:\n"
        "{\n"
        '  "predicted_molecules": ["..."]\n'
        "}"
    )
    return [
        {"role": "system", "content": "You are a FoodPuzzle MPC Reviewer agent that returns only valid JSON."},
        {"role": "user", "content": prompt},
    ]


def parse_mpc_reviewer_output(content: str) -> dict[str, Any] | None:
    data = parse_json_object(content)
    if data is None:
        molecules = parse_mpc_molecules(content)
        if not molecules:
            return None
        return {
            "predicted_molecules": molecules,
            "selected_hypothesis_index": None,
            "review_reason": "",
        }
    molecules = normalize_mpc_molecule_list(data.get("predicted_molecules"))
    if not molecules:
        return None
    selected = data.get("selected_hypothesis_index")
    return {
        "predicted_molecules": molecules,
        "selected_hypothesis_index": selected if isinstance(selected, int) else None,
        "review_reason": str(data.get("review_reason") or "").strip(),
    }


def validate_common_output_paths(args: argparse.Namespace) -> None:
    for path_value, label in [
        (args.output, "output"),
        (args.evidence_metadata, "evidence metadata"),
        (args.retrieval_metadata_output, "retrieval metadata output"),
        (args.hypotheses_metadata, "hypotheses metadata"),
    ]:
        parent = Path(path_value).parent
        if not parent.is_dir():
            raise AgentError(f"{label} parent directory does not exist: {parent}")
        if Path(path_value).exists() and not args.resume:
            raise AgentError(f"{label} file already exists; remove it or use --resume: {path_value}")


def run_mfp_agent(args: argparse.Namespace) -> int:
    if args.task != "mfp":
        raise AgentError("only --task mfp is supported")
    if args.evidence_route != "answer_masked_official_evidence":
        raise AgentError("only --evidence-route answer_masked_official_evidence is supported")
    if not args.use_llm:
        raise AgentError("--use-llm is required to allow real API calls")
    if not args.official_evidence_pkl:
        raise AgentError("--official-evidence-pkl is required for --task mfp")
    if not args.icl_retrieval_metadata:
        raise AgentError("--icl-retrieval-metadata is required for --task mfp")

    for path_value, label in [
        (args.train, "train"),
        (args.test, "test"),
        (args.official_evidence_pkl, "official evidence pickle"),
        (args.icl_retrieval_metadata, "ICL retrieval metadata"),
    ]:
        if not Path(path_value).is_file():
            raise AgentError(f"{label} file not found: {path_value}")
    validate_common_output_paths(args)

    evaluation = load_evaluation_module()
    evaluation.load_local_env_file()
    llm_config = evaluation.resolve_llm_config(args)
    evaluation.require_api_key(llm_config)

    train_rows = read_jsonl(Path(args.train))
    test_rows = read_jsonl(Path(args.test))
    train_by_id = {str(row["id"]): row for row in train_rows}
    train_ids = set(train_by_id)
    test_ids = {str(row["id"]) for row in test_rows}
    if train_ids & test_ids:
        raise AgentError("train and test split overlap")

    official_evidence, evidence_info = load_official_evidence(Path(args.official_evidence_pkl))
    retrieval_metadata = load_retrieval_metadata(Path(args.icl_retrieval_metadata))
    train_idf = build_train_idf(train_rows)

    existing_predictions = read_success_prediction_ids(Path(args.output)) if args.resume else set()
    existing_evidence = read_existing_ids(Path(args.evidence_metadata)) if args.resume else set()
    existing_retrieval = read_existing_ids(Path(args.retrieval_metadata_output)) if args.resume else set()
    existing_hypotheses = read_success_hypotheses_ids(Path(args.hypotheses_metadata)) if args.resume else set()

    total = len(test_rows)
    skipped = 0
    newly_generated = 0
    success = 0
    failures = 0
    scientist_parse_failures = 0
    reviewer_parse_failures = 0

    for row in test_rows:
        row_id = str(row["id"])
        if row_id in existing_predictions:
            skipped += 1
            continue

        selected = select_starting_molecules(
            row, train_idf, official_evidence, args.starting_point_count
        )
        evidence_blocks, evidence_text, hits_after_mask = build_evidence_blocks(
            row, selected, official_evidence, args.max_snippets_per_molecule
        )
        demos = resolve_demos(row_id, retrieval_metadata, train_by_id, args.bm25_top_k)

        scientist_parse_failed = False
        reviewer_parse_failed = False
        hypotheses: list[dict[str, str]] = []
        reviewer_output: dict[str, Any] | None = None
        error: str | None = None
        try:
            scientist_content = evaluation.call_chat_completion(
                build_scientist_messages(row, selected, evidence_text, demos),
                llm_config,
            )
            parsed_hypotheses = parse_hypotheses(scientist_content)
            if parsed_hypotheses is None:
                scientist_parse_failed = True
                scientist_parse_failures += 1
            else:
                hypotheses = parsed_hypotheses[:3]

            reviewer_content = evaluation.call_chat_completion(
                build_reviewer_messages(row, evidence_text, demos, hypotheses),
                llm_config,
            )
            reviewer_output = parse_reviewer_output(reviewer_content)
            if reviewer_output is None:
                reviewer_parse_failed = True
                reviewer_parse_failures += 1
                error = "reviewer_parse_failed"
                prediction = ""
                failures += 1
            else:
                prediction = reviewer_output["predicted_food"]
                success += 1
        except Exception as exc:
            # 余额不足是全局 API 状态，继续逐条重试只会生成一批无效空预测；立即停止，保留已成功行供 resume。
            message = str(exc)
            if "Insufficient Balance" in message or "HTTP error: 402" in message:
                raise AgentError(f"provider quota/balance error; stop for resume later: {message}") from exc
            error = f"llm_error: {exc}"
            prediction = ""
            failures += 1

        newly_generated += 1

        if row_id not in existing_evidence:
            append_jsonl(
                Path(args.evidence_metadata),
                {
                    "id": row_id,
                    "actual_food_for_audit": row.get("actual_food"),
                    "evidence_route": args.evidence_route,
                    "selected_molecules": selected,
                    "evidence_blocks": evidence_blocks,
                    "official_evidence_actual_food_hits_after_mask": hits_after_mask,
                    "starting_point_method": "train_idf_with_official_evidence_availability",
                    "uses_test_actual_food_for_starting_point": False,
                },
            )
        if row_id not in existing_retrieval:
            append_jsonl(
                Path(args.retrieval_metadata_output),
                {
                    "id": row_id,
                    "retrieved": [
                        {
                            "id": demo.get("id"),
                            "rank": demo.get("rank"),
                            "score": demo.get("score"),
                            "actual_food": demo.get("actual_food"),
                        }
                        for demo in demos
                    ],
                    "bm25_demo_masking": False,
                    "bm25_demo_replacement": False,
                    "bm25_demo_similar_food_risk": row_id in {"32", "25"},
                },
            )
        if row_id not in existing_hypotheses:
            append_jsonl(
                Path(args.hypotheses_metadata),
                {
                    "id": row_id,
                    "scientist_parse_failed": scientist_parse_failed,
                    "reviewer_parse_failed": reviewer_parse_failed,
                    "hypotheses": hypotheses,
                    "reviewer_output": reviewer_output,
                    "llm_calls": {
                        "scientist": 1,
                        "reviewer": 1,
                    },
                    "error": error,
                },
            )

        prediction_row = {"id": row_id, "predicted_food": prediction}
        if error:
            prediction_row["error"] = error
        append_jsonl(Path(args.output), prediction_row)

    print("AGENT_STATUS: PASS")
    print(f"total: {total}")
    print(f"existing_predictions: {len(existing_predictions)}")
    print(f"newly_generated: {newly_generated}")
    print(f"skipped: {skipped}")
    print(f"success: {success}")
    print(f"failures: {failures}")
    print(f"scientist_parse_failures: {scientist_parse_failures}")
    print(f"reviewer_parse_failures: {reviewer_parse_failures}")
    print(f"output_path: {args.output}")
    print(f"official_evidence_size: {evidence_info['size']}")
    print(f"official_evidence_sha256: {evidence_info['sha256']}")
    return 0


def run_mpc_agent(args: argparse.Namespace) -> int:
    if args.task != "mpc":
        raise AgentError("only --task mpc is supported")
    if not args.use_llm:
        raise AgentError("--use-llm is required to allow real API calls")
    evidence_arg = args.evidence or args.official_evidence_pkl
    if not evidence_arg:
        raise AgentError("--evidence is required for --task mpc")
    evidence_path = Path(evidence_arg)

    for path_value, label in [
        (args.train, "train"),
        (args.test, "test"),
        (str(evidence_path), "MPC evidence pickle"),
    ]:
        if not Path(path_value).is_file():
            raise AgentError(f"{label} file not found: {path_value}")
    validate_common_output_paths(args)

    evaluation = load_evaluation_module()
    evaluation.load_local_env_file()
    llm_config = evaluation.resolve_llm_config(args)
    evaluation.require_api_key(llm_config)

    train_rows = read_jsonl(Path(args.train))
    test_rows = read_jsonl(Path(args.test))
    train_ids = {str(row["id"]) for row in train_rows}
    test_ids = {str(row["id"]) for row in test_rows}
    if train_ids & test_ids:
        raise AgentError("train and test split overlap")

    # MPC 不使用 MFP 的 entropy starting point；BM25 demonstrations 只来自 labeled train split。
    evidence, evidence_info = load_mpc_evidence(evidence_path)
    bm25 = MPCBM25Index(train_rows)

    existing_predictions = read_success_agent_prediction_ids(Path(args.output), "mpc") if args.resume else set()
    existing_evidence = read_existing_ids(Path(args.evidence_metadata)) if args.resume else set()
    existing_retrieval = read_existing_ids(Path(args.retrieval_metadata_output)) if args.resume else set()
    existing_hypotheses = read_success_hypotheses_ids(Path(args.hypotheses_metadata)) if args.resume else set()

    total = len(test_rows)
    skipped = 0
    newly_generated = 0
    success = 0
    failures = 0
    scientist_parse_failures = 0
    reviewer_parse_failures = 0

    for row in test_rows:
        row_id = str(row["id"])
        if row_id in existing_predictions:
            skipped += 1
            continue

        demos = bm25.retrieve(row, args.bm25_top_k)
        evidence_meta, evidence_text = build_mpc_evidence_blocks(
            row, evidence, args.max_evidence_snippets
        )

        scientist_parse_failed = False
        reviewer_parse_failed = False
        hypotheses: list[dict[str, Any]] = []
        reviewer_output: dict[str, Any] | None = None
        predicted_molecules: list[str] = []
        normalization_metadata: dict[str, Any] = {}
        error: str | None = None
        try:
            scientist_content = evaluation.call_chat_completion(
                build_mpc_scientist_messages(row, evidence_text, demos),
                llm_config,
            )
            parsed_hypotheses = parse_mpc_hypotheses(scientist_content)
            if parsed_hypotheses is None:
                scientist_parse_failed = True
                scientist_parse_failures += 1
            else:
                hypotheses = parsed_hypotheses[:3]

            reviewer_content = evaluation.call_chat_completion(
                build_mpc_reviewer_messages(
                    row, evidence_text, demos, hypotheses, args.reviewer_evidence_mode
                ),
                llm_config,
            )
            reviewer_output = parse_mpc_reviewer_output(reviewer_content)
            if reviewer_output is None:
                reviewer_parse_failed = True
                reviewer_parse_failures += 1
                error = "reviewer_parse_failed"
                failures += 1
            else:
                predicted_molecules, normalization_metadata = normalize_mpc_agent_final_molecules(
                    reviewer_output["predicted_molecules"], row
                )
                reviewer_output["predicted_molecules"] = predicted_molecules
                success += 1
        except Exception as exc:
            message = str(exc)
            if "Insufficient Balance" in message or "HTTP error: 402" in message:
                raise AgentError(f"provider quota/balance error; stop for resume later: {message}") from exc
            error = f"llm_error: {exc}"
            failures += 1

        newly_generated += 1

        if row_id not in existing_retrieval:
            append_jsonl(
                Path(args.retrieval_metadata_output),
                {
                    "id": row_id,
                    "retrieved": [
                        {
                            "id": demo.get("id"),
                            "rank": demo.get("rank"),
                            "score": demo.get("score"),
                            "target_food": demo.get("target_food"),
                        }
                        for demo in demos
                    ],
                    "retrieval_corpus": str(args.train),
                    "top_k": args.bm25_top_k,
                },
            )
        if row_id not in existing_evidence:
            append_jsonl(
                Path(args.evidence_metadata),
                {
                    "id": row_id,
                    **evidence_meta,
                    "evidence_file": str(evidence_path),
                    "evidence_route": "food_centered_local_evidence",
                    "evidence_top_k": args.max_evidence_snippets,
                    "reviewer_evidence_mode": args.reviewer_evidence_mode,
                },
            )
        if row_id not in existing_hypotheses:
            append_jsonl(
                Path(args.hypotheses_metadata),
                {
                    "id": row_id,
                    "target_food": row.get("target_food"),
                    "hypothesis_count": len(hypotheses),
                    "scientist_parse_failed": scientist_parse_failed,
                    "reviewer_parse_failed": reviewer_parse_failed,
                    "hypotheses": hypotheses,
                    "reviewer_selected_hypothesis": (
                        reviewer_output or {}
                    ).get("selected_hypothesis_index"),
                    "reviewer_output": reviewer_output,
                    **normalization_metadata,
                    "evidence_top_k": args.max_evidence_snippets,
                    "reviewer_evidence_mode": args.reviewer_evidence_mode,
                    "llm_calls": {
                        "scientist": 1,
                        "reviewer": 1,
                    },
                    "error": error,
                },
            )

        prediction_row = {
            "id": row_id,
            "task": row.get("task", "mpc"),
            "target_food": row.get("target_food"),
            "partial_molecules": row.get("partial_molecules", []),
            "n": row.get("n"),
            "predicted_molecules": predicted_molecules,
        }
        if error:
            prediction_row["error"] = error
        append_jsonl(Path(args.output), prediction_row)

    print("AGENT_STATUS: PASS")
    print(f"task: {args.task}")
    print(f"total: {total}")
    print(f"existing_predictions: {len(existing_predictions)}")
    print(f"newly_generated: {newly_generated}")
    print(f"skipped: {skipped}")
    print(f"success: {success}")
    print(f"failures: {failures}")
    print(f"scientist_parse_failures: {scientist_parse_failures}")
    print(f"reviewer_parse_failures: {reviewer_parse_failures}")
    print(f"output_path: {args.output}")
    print(f"mpc_evidence_size: {evidence_info['size']}")
    print(f"mpc_evidence_sha256: {evidence_info['sha256']}")
    return 0


def run_agent(args: argparse.Namespace) -> int:
    if args.task == "mfp":
        return run_mfp_agent(args)
    if args.task == "mpc":
        return run_mpc_agent(args)
    raise AgentError(f"unsupported task: {args.task}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FoodPuzzle Scientific Agent baseline.")
    parser.add_argument("--task", required=True, choices=["mfp", "mpc"])
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--official-evidence-pkl", default=None)
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--icl-retrieval-metadata", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-metadata", required=True)
    parser.add_argument("--retrieval-metadata", "--retrieval-metadata-output", dest="retrieval_metadata_output", required=True)
    parser.add_argument("--hypotheses-metadata", required=True)
    parser.add_argument("--starting-point-count", type=int, default=5)
    parser.add_argument("--max-snippets-per-molecule", type=int, default=3)
    parser.add_argument(
        "--max-evidence-snippets",
        "--evidence-top-k",
        dest="max_evidence_snippets",
        type=int,
        default=10,
        help="MPC food-centered evidence snippets per test food; default 10 for formal paper-aligned MPC Agent.",
    )
    parser.add_argument(
        "--reviewer-evidence-mode",
        choices=["none", "full"],
        default="none",
        help="MPC Reviewer evidence input mode; default none for formal paper-aligned MPC Agent.",
    )
    parser.add_argument("--bm25-top-k", type=int, default=3)
    parser.add_argument("--evidence-route", default="answer_masked_official_evidence")
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run_agent(args)
    except AgentError as exc:
        print("AGENT_STATUS: FAIL")
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
