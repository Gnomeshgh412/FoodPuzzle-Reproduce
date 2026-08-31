#!/usr/bin/env python3
"""Paper-aligned evaluation reconstruction for FoodPuzzle.

本文件不是官方 `evaluation.py` 的逐行复现。官方仓库未公开
`DSP_functions.py`、`config.py`、`utils.py`、pickle FlavorDB 和 baseline
result pickle，因此这里直接基于公开 JSONL + SQLite 重建 evaluation 入口。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PROVIDER = "deepseek"
FORMAL_EVALUATION_PROVIDER = "deepseek"
DEFAULT_MPC_FUNCTIONAL_GROUP_CACHE = (
    Path("results")
    / "Only-Deepseek"
    / "shared_cache"
    / "deepseek-v4-flash_functional_group_cache_official_parser_v2.json"
)
DEFAULT_MFP_FOOD_CATEGORY_CACHE = (
    Path("results")
    / "Only-Deepseek"
    / "shared_cache"
    / "deepseek-v4-flash_food_category_cache_official_mapping_v1.json"
)
MFP_FOOD_CATEGORY_PROMPT_VERSION = "official-category-mapping-json-v1"
MFP_EVALUATION_PROTOCOL_VERSION = "official-gold-deepseek-shared-judge-v1"
MPC_FUNCTIONAL_GROUP_PROMPT_VERSION = "official-permissive-json-v2"
MPC_EVALUATION_PROTOCOL_VERSION = "official-literal-gold-deepseek-judge-v4"

# provider 配置集中放置，避免 endpoint / key 环境变量散落在代码各处。
LLM_PROVIDERS = {
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com/chat/completions",
    },
}

# MFP macro categories 按官方 evaluation.py 原始顺序保留。
# 其中 Vegetable 是官方列表里已存在的大小写重复；评分比较仍按
# 官方代码小写化，因而语义上是论文 Table 1 的 21 个 macro categories。
MFP_MACRO_CATEGORIES = [
    "cereal",
    "plant",
    "seed",
    "flower",
    "animalproduct",
    "additive",
    "fishseafood",
    "dairy",
    "fruit",
    "bakery",
    "dish",
    "nutseed",
    "vegetable",
    "meat",
    "cerealcrop",
    "herb",
    "essentialoil",
    "fungus",
    "spice",
    "Vegetable",
    "beverage",
    "plantderivative",
]

# MPC fixed functional group vocabulary。
# 说明：该列表来自官方公开 evaluation.py 中写死的 functional_groups。
# 为了 paper/code-aligned reproduction，formal MPC evaluation 使用该固定列表，
# 不从本地 FlavorDB 动态生成 vocabulary。若后续使用 FlavorDB-expanded vocabulary，
# 只能作为 ablation / engineering extension，不能作为 formal 主线。
MPC_FUNCTIONAL_GROUP_VOCABULARY = [
    "thiocarboxylic",
    "cation",
    "sulfone",
    "hydroxy",
    "sulfonic",
    "alcohol",
    "ketone",
    "hydroxyhetarene",
    "amine",
    "aryl",
    "trialkylamine",
    "carboxylic",
    "alkyne",
    "ketene",
    "anhydride",
    "acetal",
    "amide",
    "derivative",
    "carbonitrile",
    "heterocyclic",
    "(alkylamine)",
    "aliphatic/aromatic",
    "imide,",
    "enol",
    "halide",
    "phenol",
    "sulfoxide",
    "aldehyde",
    "thioether",
    "hydroperoxide",
    "ester",
    "isothiocyanate",
    "alpha-aminoacid",
    "dialkylamine",
    "thiol",
    "ammonium",
    "aliphatic",
    "arylthiol",
    "aromatic",
    "thioacetal",
    "alpha-hydroxyacid",
    "acid",
    "sulfanyl",
    "alkylthiol",
    "salt",
    "alkene",
    "ether",
    "sulfenic",
    "carbonyl",
    "nitrite",
    "halogen",
    "chloride",
    "oxo(het)arene",
]


class EvaluationError(Exception):
    """Evaluation 中可预期的失败。"""


class ChatCompletionHTTPError(EvaluationError):
    """Chat Completions API 返回的 HTTP 错误，用于参数兼容 fallback 判断。"""

    def __init__(self, provider: str, status: int, detail: str):
        self.provider = provider
        self.status = status
        self.detail = detail
        super().__init__(f"{provider} API HTTP error: {status}: {detail}")


def strip_env_value_quotes(value: str) -> str:
    """去掉 .env.local 中成对的单双引号，不处理 shell 展开语法。"""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_local_env_line(line: str) -> tuple[str, str] | None:
    """解析 .env.local 的简单 key=value 行，跳过空行、注释和无效行。"""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = strip_env_value_quotes(value.strip())
    if not key:
        return None
    return key, value


def find_local_env_file() -> Path | None:
    """优先查当前工作目录，其次按 evaluation.py 位置回到项目根目录。"""
    cwd_candidate = Path.cwd() / ".env.local"
    if cwd_candidate.is_file():
        return cwd_candidate

    repo_candidate = Path(__file__).resolve().parents[2] / ".env.local"
    if repo_candidate.is_file():
        return repo_candidate
    return None


def load_local_env_file() -> None:
    """启动早期读取本地 .env.local；系统环境变量优先，不会被本地文件覆盖。"""
    env_path = find_local_env_file()
    if env_path is None:
        return

    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_local_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            # 若系统环境变量已存在，保持系统环境变量优先，避免 .env.local 覆盖。
            if key not in os.environ:
                os.environ[key] = value


def normalize_text(value: Any) -> str:
    """对 food / molecule / category 文本做保守归一，避免大小写和空白差异。"""
    return " ".join(str(value).strip().lower().split())


def normalize_compact(value: Any) -> str:
    """归一并移除空格、连字符和下划线，用于 macro category 对齐。"""
    text = normalize_text(value)
    return text.replace(" ", "").replace("-", "").replace("_", "")


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """读取 MFP / MPC gold 数据或 prediction 文件；坏行计为 parse failure。"""
    rows: list[dict[str, Any]] = []
    failures = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
                rows.append(row)
            except Exception as exc:
                failures += 1
                print(f"PARSE_FAILURE: {path}:{line_no}: {exc}", file=sys.stderr)
    return rows, failures


def read_prediction_jsonl(path: Path, task: str) -> tuple[dict[str, Any], int]:
    """读取 prediction 文件，并兼容官方/重建格式中的 prediction 字段。"""
    rows, failures = read_jsonl(path)
    predictions: dict[str, Any] = {}
    for row in rows:
        row_id = row.get("id")
        if row_id is None:
            failures += 1
            continue
        key = str(row_id)
        if task == "mfp":
            value = row.get("predicted_food", row.get("prediction"))
            if not isinstance(value, str) or not value.strip():
                failures += 1
                continue
            predictions[key] = value
        elif task == "mpc":
            value = row.get("predicted_molecules", row.get("prediction"))
            molecules = coerce_molecule_list(value)
            if molecules is None:
                failures += 1
                continue
            predictions[key] = molecules
        else:
            raise EvaluationError(f"unknown task: {task}")
    return predictions, failures


def read_mfp_prediction_jsonl_with_failures(path: Path) -> tuple[dict[str, str], int, set[str]]:
    """读取 MFP prediction，并保留可定位到 id 的 parse failure，供 per-sample detail 使用。"""
    predictions: dict[str, str] = {}
    parse_failed_ids: set[str] = set()
    failures = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
            except Exception as exc:
                failures += 1
                print(f"PARSE_FAILURE: {path}:{line_no}: {exc}", file=sys.stderr)
                continue

            row_id = row.get("id")
            if row_id is None:
                failures += 1
                continue
            key = str(row_id)
            value = row.get("predicted_food", row.get("prediction"))
            if not isinstance(value, str) or not value.strip():
                failures += 1
                parse_failed_ids.add(key)
                continue
            # Frozen from the public official evaluator before category mapping.
            cleaned = value.replace("[", "").replace("]", "").replace("*", "").strip()
            if not cleaned:
                failures += 1
                parse_failed_ids.add(key)
                continue
            predictions[key] = cleaned
    return predictions, failures, parse_failed_ids


def read_mpc_prediction_jsonl_with_failures(path: Path) -> tuple[dict[str, list[str]], int, set[str]]:
    """读取 MPC prediction，并保留可定位到 id 的 parse failure，供 per-sample detail 使用。"""
    predictions: dict[str, list[str]] = {}
    parse_failed_ids: set[str] = set()
    failures = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
            except Exception as exc:
                failures += 1
                print(f"PARSE_FAILURE: {path}:{line_no}: {exc}", file=sys.stderr)
                continue

            row_id = row.get("id")
            if row_id is None:
                failures += 1
                continue
            key = str(row_id)
            molecules = coerce_molecule_list(row.get("predicted_molecules", row.get("prediction")))
            # 空预测是合法 prediction，后续按 functional group F1=0 处理；只有解析失败或显式 error 才计 failure。
            if row.get("error") or molecules is None:
                failures += 1
                parse_failed_ids.add(key)
                continue
            predictions[key] = molecules
    return predictions, failures, parse_failed_ids


def coerce_molecule_list(value: Any) -> list[str] | None:
    """解析 MPC prediction，支持 list 或字符串形式的 list。"""
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                items = parsed
            else:
                items = [part.strip() for part in text.strip("[]").split(",")]
        except Exception:
            items = [part.strip() for part in text.strip("[]").split(",")]
    else:
        return None

    molecules = [str(item).strip() for item in items if str(item).strip()]
    return molecules


def build_food_category_map(db_path: Path) -> dict[str, str]:
    """按官方 evaluator 构造 entity_alias_readable -> macro category 映射。"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    mapping: dict[str, str] = {}
    try:
        rows = cur.execute("SELECT category, entity_alias_readable FROM food_entities")
        for category, readable in rows:
            if not category or not readable:
                continue
            # 官方代码只使用 entity_alias_readable，并取 category 的第一个层级。
            mapping[str(readable)] = str(category).split("-")[0]
    finally:
        conn.close()
    return mapping


def parse_functional_groups(raw_value: Any) -> set[str]:
    """逐字兼容官方 evaluation.py 对 FlavorDB functional_groups 的解析。"""
    # 特意保留官方公开代码的控制流、大小写和空 token。官方的
    # `else` 绑定到外层 `if "@" in group`；两个内层 if 都没有 else。
    groups = str(raw_value or "").split(" ")
    parsed: list[str] = []
    for group in groups:
        if "@" in group:
            if "compound" in group:
                parsed.append(group.split("@")[1])
            if "primary" in group:
                parsed.append(group.split("@")[0])
        else:
            parsed.append(group)
    return set(parsed)


def build_molecule_group_map(db_path: Path) -> dict[str, set[str]]:
    """从 SQLite 构造 gold molecule common_name 到 functional group set 的映射。"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    mapping: dict[str, set[str]] = {}
    try:
        rows = cur.execute("SELECT common_name, functional_groups FROM molecules")
        for common_name, raw_groups in rows:
            if not common_name:
                continue
            groups = parse_functional_groups(raw_groups)
            mapping[normalize_text(common_name)] = groups
    finally:
        conn.close()

    if not mapping:
        raise EvaluationError("FAIL: cannot parse molecule functional groups from DB")
    return mapping


def set_metrics(predicted: set[str], gold: set[str]) -> dict[str, float]:
    """计算 set precision / recall / F1 / IoU。"""
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}
    overlap = len(predicted & gold)
    # 官方公开 evaluator 在 precision 分母上加 1e-09。
    precision = overlap / (len(predicted) + 1e-09)
    recall = overlap / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    union = len(predicted | gold)
    iou = overlap / union if union else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 per-sample evaluation detail；不包含任何 API key 或 raw API response。"""
    if not path.parent.is_dir():
        raise EvaluationError(f"details parent directory does not exist: {path.parent}")
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写出 summary JSON；用于 MPC/MFP 正式评估报告落盘。"""
    if not path.parent.is_dir():
        raise EvaluationError(f"summary parent directory does not exist: {path.parent}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_food_category_cache(path: Path) -> dict[str, str]:
    """Read normalized predicted-food -> one official macro category."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    allowed = {normalize_compact(category) for category in MFP_MACRO_CATEGORIES}
    cache: dict[str, str] = {}
    for food, category in data.items():
        if isinstance(food, str) and isinstance(category, str):
            normalized_category = normalize_compact(category)
            if normalized_category in allowed:
                cache[normalize_text(food)] = normalized_category
    return cache


def write_food_category_cache(path: Path, cache: dict[str, str]) -> None:
    """Atomically persist the shared MFP judge mapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(dict(sorted(cache.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def food_category_cache_metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".metadata.json")


def validate_or_create_food_category_cache_metadata(
    path: Path, llm_config: dict[str, str]
) -> None:
    """Bind the MFP shared cache to one judge, prompt and official vocabulary."""
    vocabulary_sha256 = hashlib.sha256(
        json.dumps(
            MFP_MACRO_CATEGORIES,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected = {
        "provider": llm_config["provider"],
        "model": llm_config["model"],
        "base_url": llm_config["base_url"],
        "prompt_version": MFP_FOOD_CATEGORY_PROMPT_VERSION,
        "vocabulary_sha256": vocabulary_sha256,
    }
    metadata_path = food_category_cache_metadata_path(path)
    if metadata_path.is_file():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise EvaluationError(f"invalid food category cache metadata: {metadata_path}") from exc
        if existing != expected:
            raise EvaluationError(
                "food category cache judge mismatch; use a different cache path or the "
                f"recorded provider/model: {metadata_path}"
            )
        return
    if path.is_file() and path.stat().st_size > 0:
        raise EvaluationError(
            "food category cache exists without judge metadata; choose a new cache path "
            f"or create matching metadata first: {path}"
        )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = metadata_path.with_name(metadata_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(metadata_path)


def read_functional_group_cache(path: Path) -> dict[str, set[str]]:
    """读取 predicted molecule -> functional groups cache；保留全部非空标签。"""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    cache: dict[str, set[str]] = {}
    for molecule, groups in data.items():
        if not isinstance(groups, list):
            continue
        parsed: set[str] = set()
        for group in groups:
            if isinstance(group, str) and group.strip():
                parsed.add(group.strip())
        cache[normalize_text(molecule)] = parsed
    return cache


def write_functional_group_cache(path: Path, cache: dict[str, set[str]]) -> None:
    """原子写出 cache；cache 只用于避免重复 LLM 调用，不改变 evaluation 公式。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: sorted(value) for key, value in sorted(cache.items())}
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def functional_group_cache_metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".metadata.json")


def validate_or_create_functional_group_cache_metadata(
    path: Path, llm_config: dict[str, str]
) -> None:
    """绑定共享 cache 的 judge 与词表，防止不同模型误用同一路径。"""
    vocabulary_sha256 = hashlib.sha256(
        json.dumps(
            MPC_FUNCTIONAL_GROUP_VOCABULARY,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected = {
        "provider": llm_config["provider"],
        "model": llm_config["model"],
        "base_url": llm_config["base_url"],
        "prompt_version": MPC_FUNCTIONAL_GROUP_PROMPT_VERSION,
        "vocabulary_sha256": vocabulary_sha256,
    }
    metadata_path = functional_group_cache_metadata_path(path)
    if metadata_path.is_file():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise EvaluationError(f"invalid functional group cache metadata: {metadata_path}") from exc
        if existing != expected:
            raise EvaluationError(
                "functional group cache judge mismatch; use a different cache path or the "
                f"recorded provider/model: {metadata_path}"
            )
        return

    if path.is_file() and path.stat().st_size > 0:
        raise EvaluationError(
            "functional group cache exists without judge metadata; choose a new cache path "
            f"or create matching metadata first: {path}"
        )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = metadata_path.with_name(metadata_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(metadata_path)


def resolve_llm_config(args: argparse.Namespace) -> dict[str, Any]:
    """解析 provider / model / endpoint，优先级为 CLI > 环境变量 > 默认值。"""
    provider = args.llm_provider or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER
    provider = normalize_text(provider)
    if provider not in LLM_PROVIDERS:
        raise EvaluationError(f"unsupported llm provider: {provider}")

    provider_config = LLM_PROVIDERS[provider]
    model = args.llm_model or provider_config["default_model"]
    base_url = args.llm_base_url or provider_config["base_url"]
    config: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "key_env": provider_config["key_env"],
    }
    max_tokens = getattr(args, "llm_max_tokens", None)
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise EvaluationError("llm_max_tokens must be a positive integer")
        config["max_tokens"] = max_tokens
    return config


def resolve_formal_evaluation_llm_config(args: argparse.Namespace) -> dict[str, str]:
    """正式 MFP/MPC 评测固定使用同一个 judge，避免方法间评测模型漂移。"""
    if args.llm_provider and normalize_text(args.llm_provider) != FORMAL_EVALUATION_PROVIDER:
        raise EvaluationError(
            "formal evaluation provider is fixed to "
            f"{FORMAL_EVALUATION_PROVIDER}; received {args.llm_provider}"
        )
    evaluation_args = argparse.Namespace(
        llm_provider=FORMAL_EVALUATION_PROVIDER,
        llm_model=None,
        llm_base_url=None,
    )
    return resolve_llm_config(evaluation_args)


def require_api_key(llm_config: dict[str, str]) -> str:
    """读取 API key 并做安全检查；OpenAI / DeepSeek 均只允许来自环境变量。"""
    key_env = llm_config["key_env"]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise EvaluationError(
            f'{key_env} is missing. Set it with: export {key_env}="你的 key"'
        )
    return api_key


def is_response_format_compat_error(exc: ChatCompletionHTTPError) -> bool:
    """只把明确的 JSON Output 参数兼容问题作为 fallback 条件。"""
    if exc.status != 400:
        return False
    detail = exc.detail.lower()
    return "response_format" in detail or "json_object" in detail


def is_thinking_compat_error(exc: ChatCompletionHTTPError) -> bool:
    """只把明确的 DeepSeek thinking 参数兼容问题作为 fallback 条件。"""
    if exc.status != 400:
        return False
    detail = exc.detail.lower()
    return "thinking" in detail


def is_temperature_compat_error(exc: ChatCompletionHTTPError) -> bool:
    """GPT-5 等模型可能拒绝 temperature；仅对明确参数错误执行 fallback。"""
    if exc.status != 400:
        return False
    detail = exc.detail.lower()
    return "temperature" in detail and (
        "unsupported" in detail or "not support" in detail or "invalid" in detail
    )


def build_chat_payload(
    messages: list[dict[str, str]],
    llm_config: dict[str, Any],
    use_response_format: bool,
    use_thinking: bool,
    use_temperature: bool,
) -> dict[str, Any]:
    """构造 Chat Completions 请求体，DeepSeek thinking 字段只给 DeepSeek 使用。"""
    payload = {
        "model": llm_config["model"],
        "stream": False,
        "messages": messages,
    }
    if use_temperature:
        payload["temperature"] = 0
    if llm_config.get("max_tokens") is not None:
        payload["max_tokens"] = llm_config["max_tokens"]
    # JSON Output 能提高 judge 输出稳定性；prompt 中仍明确要求只输出 JSON。
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}
    # DeepSeek V4 可能默认启用 thinking mode；evaluation 中关闭它以稳定 JSON 输出。
    if llm_config["provider"] == "deepseek" and use_thinking:
        payload["thinking"] = {"type": "disabled"}
    return payload


def post_chat_payload(payload: dict[str, Any], llm_config: dict[str, str]) -> dict[str, Any]:
    """发送一次 Chat Completions 请求；错误中不包含任何 API key 内容。"""
    api_key = require_api_key(llm_config)
    request = urllib.request.Request(
        llm_config["base_url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    # DeepSeek/OpenAI 长串行 evaluation 中可能遇到偶发连接重置；这里只重试网络层错误。
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ChatCompletionHTTPError(llm_config["provider"], exc.code, detail) from exc
        except Exception as exc:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise EvaluationError(f"{llm_config['provider']} API request failed: {exc}") from exc
    raise EvaluationError(f"{llm_config['provider']} API request failed after retries")


def call_chat_completion_response(
    messages: list[dict[str, str]], llm_config: dict[str, Any]
) -> dict[str, Any]:
    """调用 Chat Completions，并保留截断诊断所需的响应元数据。"""
    use_response_format = True
    use_thinking = llm_config["provider"] == "deepseek"
    use_temperature = True

    while True:
        payload = build_chat_payload(
            messages,
            llm_config,
            use_response_format=use_response_format,
            use_thinking=use_thinking,
            use_temperature=use_temperature,
        )
        try:
            body = post_chat_payload(payload, llm_config)
            break
        except ChatCompletionHTTPError as exc:
            # DeepSeek thinking 参数在不同 API 行为下可能不兼容；仅对明确 400 参数错误重试。
            if use_thinking and is_thinking_compat_error(exc):
                use_thinking = False
                continue
            # OpenAI / DeepSeek 某些模型可能不支持 response_format；仅对明确 400 参数错误重试。
            if use_response_format and is_response_format_compat_error(exc):
                use_response_format = False
                continue
            if use_temperature and is_temperature_compat_error(exc):
                use_temperature = False
                continue
            raise

    try:
        choice = body["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("message content is not a string")
        return {
            "content": content,
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage"),
        }
    except Exception as exc:
        raise EvaluationError(f"{llm_config['provider']} API response has unexpected shape: {body}") from exc


def call_chat_completion(messages: list[dict[str, str]], llm_config: dict[str, Any]) -> str:
    """兼容旧调用方：返回文本，同时由 response 版本保留完整停止原因。"""
    return call_chat_completion_response(messages, llm_config)["content"]


def parse_category_json(content: str, categories: list[str]) -> str | None:
    """解析 MFP LLM JSON 输出；非法 JSON 或越界类别计为 failure。"""
    try:
        data = json.loads(content)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    category = data.get("category")
    normalized = normalize_compact(category)
    allowed = {normalize_compact(item): item for item in categories}
    if normalized not in allowed:
        return None
    return allowed[normalized]


def parse_groups_json(content: str, vocabulary: list[str]) -> set[str] | None:
    """解析 MPC LLM JSON 输出，保留响应中的全部非空 group。

    官方 evaluation.py 不会因为某个返回标签不在候选表中而丢弃整条响应；
    它把模型返回的标签全部加入预测集合，未知标签随后自然无法与 gold 相交，
    但仍会进入 F1 的预测集合分母。这里保留 JSON 输出约束，同时复现该语义。
    ``vocabulary`` 仅用于 prompt，不用于过滤模型输出。
    """
    try:
        data = json.loads(content)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    groups = data.get("functional_groups")
    if not isinstance(groups, list):
        return None
    parsed: set[str] = set()
    for group in groups:
        if not isinstance(group, str):
            continue
        value = group.strip()
        if value:
            parsed.add(value)
    return parsed


def predict_food_category(
    predicted_food: str, categories: list[str], llm_config: dict[str, str]
) -> str | None:
    """MFP 中调用 LLM 做 category mapping，输出必须是固定 JSON。"""
    prompt = (
        "Map the predicted food name to exactly one FoodPuzzle macro category.\n"
        "Choose only from the provided category list.\n"
        "Return JSON only, with no explanation.\n"
        'Required JSON format: {"category": "<one_category_from_list>"}\n'
        f"Categories: {json.dumps(categories, ensure_ascii=False)}\n"
        f"Predicted food: {predicted_food}"
    )
    content = call_chat_completion(
        [
            {
                "role": "system",
                "content": "You are a strict evaluator that returns only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        llm_config=llm_config,
    )
    return parse_category_json(content, categories)


def predict_functional_groups(
    molecule: str, vocabulary: list[str], llm_config: dict[str, str]
) -> set[str] | None:
    """MPC 中调用 LLM 做 functional groups mapping，输出必须来自官方固定 vocabulary。"""
    prompt = (
        "Select the functional groups for the given molecule.\n"
        "Choose zero or more groups only from the provided candidate list.\n"
        "If you cannot determine the groups, return an empty list.\n"
        "Return JSON only, with no explanation.\n"
        'Required JSON format: {"functional_groups": ["group_a", "group_b"]}\n'
        f"Candidate functional groups: {json.dumps(vocabulary, ensure_ascii=False)}\n"
        f"Molecule: {molecule}"
    )
    content = call_chat_completion(
        [
            {
                "role": "system",
                "content": "You are a strict chemistry evaluator that returns only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        llm_config=llm_config,
    )
    return parse_groups_json(content, vocabulary)


def evaluate_mfp(args: argparse.Namespace) -> int:
    """MFP 主指标：LLM category mapping 后的 accuracy。"""
    if not args.use_llm:
        raise EvaluationError("MFP evaluation requires --use-llm because the main metric uses LLM category mapping")
    llm_config = resolve_formal_evaluation_llm_config(args)
    require_api_key(llm_config)

    # 读取 MFP gold 数据与 prediction 文件。
    gold_rows, gold_parse_failures = read_jsonl(Path(args.gold))
    predictions, pred_parse_failures, pred_parse_failed_ids = read_mfp_prediction_jsonl_with_failures(Path(args.pred))

    # 从 SQLite `flavordb.db` 构造 food category 映射。
    food_categories = build_food_category_map(Path(args.db))
    cache_path = resolve_mfp_food_category_cache_path(args)
    print(f"MFP food category cache path: {cache_path}", flush=True)
    validate_or_create_food_category_cache_metadata(cache_path, llm_config)
    food_category_cache = read_food_category_cache(cache_path)
    unique_predicted_foods = sorted(
        {
            normalize_text(food): food
            for food in predictions.values()
            if normalize_text(food)
        }.items()
    )
    cache_hit_count = sum(1 for key, _ in unique_predicted_foods if key in food_category_cache)
    pending_foods = [
        (key, food)
        for key, food in unique_predicted_foods
        if key not in food_category_cache
    ]
    print(f"MFP_EVAL_PROGRESS unique_predicted_foods={len(unique_predicted_foods)}", flush=True)
    print(f"MFP_EVAL_PROGRESS cache_hit_count={cache_hit_count}", flush=True)
    print(f"MFP_EVAL_PROGRESS llm_pending_count={len(pending_foods)}", flush=True)
    for index, (key, food) in enumerate(pending_foods, 1):
        if index == 1 or index == len(pending_foods) or index % 25 == 0:
            print(
                f"MFP_EVAL_PROGRESS llm_food_category_prediction {index}/{len(pending_foods)}",
                flush=True,
            )
        category = predict_food_category(food, MFP_MACRO_CATEGORIES, llm_config)
        # A transient parse/API failure is never cached as a category.
        if category is not None:
            food_category_cache[key] = normalize_compact(category)
            write_food_category_cache(cache_path, food_category_cache)
    stats = {
        "total_gold": len(gold_rows),
        "total_predictions": len(predictions),
        "matched_ids": 0,
        "missing_predictions": 0,
        "extra_predictions": 0,
        "parse_failures": gold_parse_failures + pred_parse_failures,
        "gold_category_lookup_failures": 0,
        "llm_mapping_failures": 0,
        "correct": 0,
        "accuracy": 0.0,
        "unique_predicted_food_count": len(unique_predicted_foods),
        "cache_hit_count": cache_hit_count,
        "llm_food_category_prediction_count": len(pending_foods),
        "food_category_cache_path": str(cache_path),
        "evaluation_protocol_version": MFP_EVALUATION_PROTOCOL_VERSION,
        "judge_provider": llm_config["provider"],
        "judge_model": llm_config["model"],
        "judge_prompt_version": MFP_FOOD_CATEGORY_PROMPT_VERSION,
    }

    gold_ids = {str(row.get("id")) for row in gold_rows if row.get("id") is not None}
    stats["extra_predictions"] = len(set(predictions) - gold_ids)
    details: list[dict[str, Any]] = []

    for row in gold_rows:
        row_id = str(row.get("id"))
        gold_food = row.get("actual_food")
        predicted_food = predictions.get(row_id)
        detail = {
            "id": row_id,
            "actual_food": gold_food,
            "predicted_food": predicted_food,
            "gold_category": None,
            "predicted_category": None,
            "correct": False,
            "parse_failed": False,
            "llm_mapping_failed": False,
            "gold_category_lookup_failed": False,
            "missing_prediction": False,
            "failure_reason": None,
        }
        if row_id not in predictions:
            if row_id in pred_parse_failed_ids:
                detail["parse_failed"] = True
                detail["failure_reason"] = "parse_failed"
            else:
                stats["missing_predictions"] += 1
                detail["missing_prediction"] = True
                detail["failure_reason"] = "missing_prediction"
            details.append(detail)
            continue
        stats["matched_ids"] += 1

        # 与官方 evaluator 一致：actual_food 对 entity_alias_readable 做精确查找。
        gold_category = food_categories.get(gold_food)
        if not gold_category:
            stats["gold_category_lookup_failures"] += 1
            detail["gold_category_lookup_failed"] = True
            detail["failure_reason"] = "gold_category_lookup_failed"
            details.append(detail)
            continue
        detail["gold_category"] = gold_category

        # 三种方法通过项目级共享 cache 使用同一个 free-text -> category 映射。
        predicted_category = food_category_cache.get(normalize_text(predictions[row_id]))
        if predicted_category is None:
            stats["llm_mapping_failures"] += 1
            detail["llm_mapping_failed"] = True
            detail["failure_reason"] = "llm_mapping_failed"
            details.append(detail)
            continue
        detail["predicted_category"] = predicted_category
        if normalize_compact(predicted_category) == normalize_compact(gold_category):
            stats["correct"] += 1
            detail["correct"] = True
        details.append(detail)

    stats["accuracy"] = stats["correct"] / stats["total_gold"] if stats["total_gold"] else 0.0
    if args.save_details:
        write_jsonl(Path(args.save_details), details)
    if args.save_summary_json:
        write_json(Path(args.save_summary_json), stats)
    print_json_status("MFP_EVALUATION_STATUS", "PASS", stats)
    return 0


def resolve_mfp_food_category_cache_path(args: argparse.Namespace) -> Path:
    """MFP defaults to one project-level cache shared by every formal method."""
    if args.food_category_cache:
        return Path(args.food_category_cache)
    return DEFAULT_MFP_FOOD_CATEGORY_CACHE


def evaluate_mpc(args: argparse.Namespace) -> int:
    """MPC 主指标：official-code-aligned functional group set F1。"""
    if args.mpc_eval_mode != "official_llm":
        raise EvaluationError("MPC formal evaluation only supports --mpc-eval-mode official_llm")
    if not args.use_llm:
        raise EvaluationError("official_llm MPC evaluation requires --use-llm")
    llm_config = resolve_formal_evaluation_llm_config(args)
    require_api_key(llm_config)

    # 读取 MPC gold 数据与 prediction 文件。
    gold_rows, gold_parse_failures = read_jsonl(Path(args.gold))
    predictions, pred_parse_failures, pred_parse_failed_ids = read_mpc_prediction_jsonl_with_failures(Path(args.pred))

    # gold missing_molecules 按官方思路从 FlavorDB 读取 functional_groups。
    molecule_groups = build_molecule_group_map(Path(args.db))
    # predicted_molecules 全部走 LLM extraction，并向 judge 提供官方固定 vocabulary。
    # 与官方 evaluator 一致，返回结果不再被该 vocabulary 强制过滤。
    group_candidates = MPC_FUNCTIONAL_GROUP_VOCABULARY

    cache_path = resolve_mpc_functional_group_cache_path(args)
    print(f"MPC functional group cache path: {cache_path}", flush=True)
    validate_or_create_functional_group_cache_metadata(cache_path, llm_config)
    functional_group_cache = read_functional_group_cache(cache_path)
    all_predicted_molecules = sorted(
        {
            normalize_text(molecule): molecule
            for molecules in predictions.values()
            for molecule in molecules
            if normalize_text(molecule)
        }.items()
    )
    cache_hit_count = sum(1 for key, _ in all_predicted_molecules if key in functional_group_cache)
    pending_molecules = [
        (key, molecule)
        for key, molecule in all_predicted_molecules
        if key not in functional_group_cache
    ]

    print(f"MPC_EVAL_PROGRESS unique_predicted_molecules={len(all_predicted_molecules)}", flush=True)
    print(f"MPC_EVAL_PROGRESS cache_hit_count={cache_hit_count}", flush=True)
    print(f"MPC_EVAL_PROGRESS llm_pending_count={len(pending_molecules)}", flush=True)
    for index, (key, molecule) in enumerate(pending_molecules, 1):
        # 正式进度输出只按批次刷新；cache 每条写出以支持中断恢复。
        if index == 1 or index == len(pending_molecules) or index % 25 == 0:
            print(
                f"MPC_EVAL_PROGRESS llm_functional_group_prediction {index}/{len(pending_molecules)}",
                flush=True,
            )
        groups = predict_functional_groups(molecule, group_candidates, llm_config)
        # 官方 evaluator 在提取异常时跳过当前 molecule。不要把解析失败永久缓存成
        # 空集合，否则后续 resume 会把暂时失败误当作一次成功的空预测。
        if groups is not None:
            functional_group_cache[key] = groups
            write_functional_group_cache(cache_path, functional_group_cache)
    stats = {
        "total_gold": len(gold_rows),
        "samples_evaluated": 0,
        "total_predictions": len(predictions),
        "matched_ids": 0,
        "missing_predictions": 0,
        "extra_predictions": 0,
        "mpc_eval_mode": args.mpc_eval_mode,
        "parse_failures": gold_parse_failures + pred_parse_failures,
        "gold_group_lookup_failures": 0,
        "unique_predicted_molecule_count": len(all_predicted_molecules),
        "cache_hit_count": cache_hit_count,
        "llm_functional_group_prediction_count": len(pending_molecules),
        "failed_functional_group_prediction_count": 0,
        "unmapped_predicted_molecule_count": 0,
        "unique_unmapped_predicted_molecule_count": 0,
        "zero_f1_count": 0,
        "predicted_count_not_equal_n": 0,
        "average_precision": 0.0,
        "average_recall": 0.0,
        "average_f1": 0.0,
        "average_iou": 0.0,
        "evaluation_protocol_version": MPC_EVALUATION_PROTOCOL_VERSION,
        "functional_group_cache_path": str(cache_path),
        "judge_provider": llm_config["provider"],
        "judge_model": llm_config["model"],
        "judge_prompt_version": MPC_FUNCTIONAL_GROUP_PROMPT_VERSION,
    }

    metrics: list[dict[str, float]] = []
    unique_unmapped_predicted_molecules: set[str] = set()
    gold_ids = {str(row.get("id")) for row in gold_rows if row.get("id") is not None}
    stats["extra_predictions"] = len(set(predictions) - gold_ids)
    details: list[dict[str, Any]] = []

    for row in gold_rows:
        row_id = str(row.get("id"))
        detail = {
            "id": row_id,
            "target_food": row.get("target_food", row.get("food")),
            "n": row.get("n"),
            "missing_molecules": row.get("missing_molecules", []),
            "predicted_molecules": predictions.get(row_id, []),
            "predicted_molecule_count": len(predictions.get(row_id, [])),
            "predicted_count_equals_n": (
                len(predictions.get(row_id, [])) == row.get("n")
                if isinstance(row.get("n"), int)
                else None
            ),
            "gold_missing_molecule_count": len(row.get("missing_molecules", []))
            if isinstance(row.get("missing_molecules"), list)
            else None,
            "gold_functional_groups": [],
            "predicted_functional_groups": [],
            "overlap_functional_groups": [],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "parse_failed": False,
            "missing_prediction": False,
            "gold_group_lookup_failures": 0,
            "failed_functional_group_predictions": [],
            "unmapped_predicted_molecules": [],
            "failure_reason": None,
        }
        if detail["predicted_count_equals_n"] is False:
            stats["predicted_count_not_equal_n"] += 1
        if row_id not in predictions:
            if row_id in pred_parse_failed_ids:
                detail["parse_failed"] = True
                detail["failure_reason"] = "parse_failed"
            else:
                stats["missing_predictions"] += 1
                detail["missing_prediction"] = True
                detail["failure_reason"] = "missing_prediction"
            # 缺失或解析失败的 prediction 计入总体平均，F1 保持 0。
            metrics.append({"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0})
            details.append(detail)
            continue
        stats["matched_ids"] += 1
        stats["samples_evaluated"] += 1

        gold_groups: set[str] = set()
        # gold missing_molecules 视为官方 evaluation 中的 actual_molecules。
        for molecule in row.get("missing_molecules", []):
            groups = molecule_groups.get(normalize_text(molecule))
            if groups is None:
                stats["gold_group_lookup_failures"] += 1
                detail["gold_group_lookup_failures"] += 1
                continue
            gold_groups.update(groups)

        predicted_groups: set[str] = set()
        for molecule in predictions[row_id]:
            key = normalize_text(molecule)
            groups = functional_group_cache.get(key)
            if groups is None:
                stats["failed_functional_group_prediction_count"] += 1
                stats["unmapped_predicted_molecule_count"] += 1
                unique_unmapped_predicted_molecules.add(key)
                detail["failed_functional_group_predictions"].append(molecule)
                detail["unmapped_predicted_molecules"].append(molecule)
                continue
            predicted_groups.update(groups)

        row_metrics = set_metrics(predicted_groups, gold_groups)
        metrics.append(row_metrics)
        if row_metrics["f1"] == 0.0:
            stats["zero_f1_count"] += 1
        detail.update(row_metrics)
        detail["gold_functional_groups"] = sorted(gold_groups)
        detail["predicted_functional_groups"] = sorted(predicted_groups)
        detail["overlap_functional_groups"] = sorted(predicted_groups & gold_groups)
        details.append(detail)

    if metrics:
        stats["average_precision"] = sum(item["precision"] for item in metrics) / len(metrics)
        stats["average_recall"] = sum(item["recall"] for item in metrics) / len(metrics)
        stats["average_f1"] = sum(item["f1"] for item in metrics) / len(metrics)
        stats["average_iou"] = sum(item["iou"] for item in metrics) / len(metrics)
    stats["unique_unmapped_predicted_molecule_count"] = len(unique_unmapped_predicted_molecules)
    if cache_path.is_file():
        stats["functional_group_cache_sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()

    if args.save_details:
        write_jsonl(Path(args.save_details), details)
    if args.save_summary_json:
        write_json(Path(args.save_summary_json), stats)
    print_json_status("MPC_EVALUATION_STATUS", "PASS", stats)
    return 0


def resolve_mpc_functional_group_cache_path(args: argparse.Namespace) -> Path:
    """MPC 默认使用项目级共享 cache，保证所有方法采用同一 judge 映射。"""
    if args.functional_group_cache:
        return Path(args.functional_group_cache)
    return DEFAULT_MPC_FUNCTIONAL_GROUP_CACHE


def print_json_status(label: str, status: str, payload: dict[str, Any] | None = None) -> None:
    """统一输出 PASS / FAIL 状态，便于后续脚本读取。"""
    print(f"{label}: {status}")
    if payload is not None:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def check_api_config(args: argparse.Namespace) -> int:
    """按 provider 检查 API key 是否存在，不调用 API，也不打印 key 内容。"""
    llm_config = resolve_llm_config(args)
    key_env = llm_config["key_env"]
    if os.environ.get(key_env):
        print("API_CONFIG_STATUS: FOUND")
        print(f'provider: {llm_config["provider"]}')
        print(f"key_env: {key_env}")
    else:
        print("API_CONFIG_STATUS: MISSING")
        print(f'provider: {llm_config["provider"]}')
        print(f"key_env: {key_env}")
        print(f'Set it before API tests with: export {key_env}="你的 key"')
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FoodPuzzle paper-aligned evaluation reconstruction")
    parser.add_argument("--check-api-config", action="store_true", help="check provider API key only")
    parser.add_argument("--task", choices=["mfp", "mpc"], help="evaluation task")
    parser.add_argument("--gold", help="gold JSONL path")
    parser.add_argument("--pred", help="prediction JSONL path")
    parser.add_argument("--db", default="data/raw/flavordb.db", help="FlavorDB SQLite path")
    parser.add_argument("--use-llm", action="store_true", help="allow real Chat Completions API calls")
    parser.add_argument(
        "--llm-provider",
        choices=sorted(LLM_PROVIDERS),
        help="LLM provider; priority: CLI > LLM_PROVIDER env > openai",
    )
    parser.add_argument("--llm-model", help="chat model; defaults depend on provider")
    parser.add_argument("--llm-base-url", help="override provider Chat Completions endpoint")
    parser.add_argument("--save-details", help="write per-sample evaluation details JSONL")
    parser.add_argument("--save-summary-json", help="write aggregate evaluation summary JSON")
    parser.add_argument(
        "--mpc-eval-mode",
        choices=["official_llm"],
        default="official_llm",
        help="MPC evaluation mode; official_llm is the formal paper/code-aligned path",
    )
    parser.add_argument(
        "--food-category-cache",
        default=None,
        help=(
            "shared cache for predicted food -> official macro category during MFP evaluation; "
            f"defaults to {DEFAULT_MFP_FOOD_CATEGORY_CACHE}"
        ),
    )
    parser.add_argument(
        "--functional-group-cache",
        default=None,
        help=(
            "shared cache for predicted molecule -> functional groups during MPC official_llm evaluation; "
            f"defaults to {DEFAULT_MPC_FUNCTIONAL_GROUP_CACHE}"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_local_env_file()
    args = build_arg_parser().parse_args(argv)
    try:
        if args.check_api_config:
            return check_api_config(args)

        if args.task in {"mfp", "mpc"}:
            if not args.gold or not args.pred:
                raise EvaluationError("--task evaluation requires --gold and --pred")
            if args.task == "mfp":
                return evaluate_mfp(args)
            return evaluate_mpc(args)

        raise EvaluationError("No action selected. Use --check-api-config or --task.")
    except EvaluationError as exc:
        print_json_status("EVALUATION_STATUS", "FAIL", {"error": str(exc)})
        message = str(exc)
        return 42 if "Insufficient Balance" in message or "HTTP error: 402" in message else 1


if __name__ == "__main__":
    raise SystemExit(main())
