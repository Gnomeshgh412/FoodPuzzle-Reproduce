#!/usr/bin/env python3
"""第五轮 MFP：准备开放具体食物 Scientist 的开发集候选验证，不调用 API。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
基础脚本 = 项目根目录 / "scripts/第一轮_MFP_UniMol独占审查器验证.py"
默认输出目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第五轮/MFP_开放具体食物科学家"


def 加载基础模块() -> Any:
    spec = importlib.util.spec_from_file_location("第五轮MFP基础", 基础脚本)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载基础脚本：{基础脚本}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 文件哈希(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 读取响应(path: Path) -> list[dict[str, Any]]:
    """读取一行一个 JSON 的开放 Scientist 响应；不容忍重复样本。"""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"响应文件第 {line_number} 行不是合法JSON：{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"响应文件第 {line_number} 行必须是JSON对象")
        sample_id = str(value.get("样本编号") or value.get("id") or "").strip()
        if not sample_id:
            raise ValueError(f"响应文件第 {line_number} 行缺少样本编号")
        if sample_id in seen:
            raise ValueError(f"响应文件出现重复样本编号：{sample_id}")
        seen.add(sample_id)
        rows.append(value)
    return rows


def 提取开放食物(response: dict[str, Any]) -> str:
    value = response.get("开放具体食物")
    if value is None and isinstance(response.get("响应"), dict):
        value = response["响应"].get("开放具体食物")
    return str(value or "").strip()


def 本地评测(
    response_path: Path,
    dev: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    categories: dict[str, str],
    normalize: Any,
) -> dict[str, Any]:
    responses = {str(row.get("样本编号") or row.get("id")): row for row in 读取响应(response_path)}
    baseline_by_id = {str(row["样本编号"]): row for row in baseline_rows}
    macro_labels = {normalize(value) for value in categories.values() if normalize(value)}
    details: list[dict[str, Any]] = []
    for row in dev:
        sample_id = str(row.get("id"))
        baseline = baseline_by_id[sample_id]
        response = responses.get(sample_id)
        generated = 提取开放食物(response) if response is not None else ""
        generated_key = normalize(generated)
        top2 = [str(x) for x in baseline["冻结BM25前二名"]]
        top2_keys = [normalize(x) for x in top2]
        invalid_reasons: list[str] = []
        if response is None:
            invalid_reasons.append("缺少响应")
        elif not generated_key:
            invalid_reasons.append("开放具体食物为空")
        if generated_key in macro_labels:
            invalid_reasons.append("输出为宏类别而非具体食物")
        if generated_key and generated_key in set(top2_keys):
            invalid_reasons.append("复制冻结BM25前二名")
        valid = not invalid_reasons
        old_third = str(baseline["原BM25第三名"])
        baseline_candidates = top2 + [old_third]
        # 无效或缺失响应安全回退到原第三名，避免把接口完整性误算成候选方法退化。
        new_candidates = top2 + ([generated] if valid else [old_third])
        gold_food = str(row.get("actual_food") or "")
        gold_key = normalize(gold_food)
        gold_category = categories.get(gold_key)
        baseline_keys = [normalize(x) for x in baseline_candidates]
        new_keys = [normalize(x) for x in new_candidates]
        baseline_categories = {categories.get(x) for x in baseline_keys if categories.get(x)}
        new_categories = {categories.get(x) for x in new_keys if categories.get(x)}
        generated_category = categories.get(generated_key) if valid else None
        details.append({
            "样本编号": sample_id,
            "真实食物": gold_food,
            "真实宏类别": gold_category,
            "冻结BM25前二名": top2,
            "原BM25第三名": old_third,
            "开放具体食物": generated,
            "开放食物映射宏类别": generated_category,
            "响应有效": valid,
            "无效原因": invalid_reasons,
            "基线Top3具体食物命中": gold_key in baseline_keys,
            "新Top3具体食物命中": gold_key in new_keys,
            "基线Top3宏类别覆盖": bool(gold_category and gold_category in baseline_categories),
            "新Top3宏类别覆盖": bool(gold_category and gold_category in new_categories),
            "开放食物Top1具体食物命中": bool(valid and generated_key == gold_key),
            "开放食物Top1宏类别命中": bool(valid and gold_category and generated_category == gold_category),
        })
    n = len(details)
    metric_names = (
        "基线Top3具体食物命中", "新Top3具体食物命中",
        "基线Top3宏类别覆盖", "新Top3宏类别覆盖",
        "开放食物Top1具体食物命中", "开放食物Top1宏类别命中",
    )
    summary = {name + "率": sum(int(x[name]) for x in details) / n for name in metric_names}
    summary.update({
        "开发集样本数": n,
        "收到响应数": len(responses),
        "有效响应数": sum(int(x["响应有效"]) for x in details),
        "开放食物可映射宏类别数": sum(int(bool(x["开放食物映射宏类别"])) for x in details),
        "开放食物未映射宏类别数": sum(int(x["响应有效"] and not x["开放食物映射宏类别"]) for x in details),
        "新减基线Top3具体食物召回": summary["新Top3具体食物命中率"] - summary["基线Top3具体食物命中率"],
        "新减基线Top3宏类别oracle覆盖": summary["新Top3宏类别覆盖率"] - summary["基线Top3宏类别覆盖率"],
        "说明": "未知开放食物保留为未映射，不使用开发集真实标签反推类别。",
    })
    return {"指标": summary, "逐样本": details}


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("必须使用 PYTHONHASHSEED=0 启动实验")
    parser = argparse.ArgumentParser()
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--评估响应", type=Path, default=None, help="可选：一行一个JSON的Scientist响应文件")
    args = parser.parse_args()
    args.输出目录.mkdir(parents=True, exist_ok=True)

    base = 加载基础模块()
    train_path = 项目根目录 / "results/splits/mfp/train.jsonl"
    dev_path = 项目根目录 / "results/splits/mfp/dev.jsonl"
    train = base.读取_jsonl(train_path)
    dev = base.读取_jsonl(dev_path)
    model = base.BM25候选模型(train)

    train_foods = {base.归一化(row.get("actual_food")) for row in train}
    dev_foods = {base.归一化(row.get("actual_food")) for row in dev}
    overlap = sorted(train_foods & dev_foods)
    payload_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for row in dev:
        ledger, _ = model.rank(row, 3)
        if len(ledger) < 3:
            raise RuntimeError(f"样本 {row.get('id')} 无法获得三个 BM25 候选")
        payload_rows.append(
            {
                "样本编号": str(row.get("id")),
                "输入分子": list(row.get("molecules") or []),
                "冻结BM25类比食品": [str(item.get("food")) for item in ledger[:2]],
                "开放生成要求": (
                    "生成一个不在两个冻结类比食品中的具体食物名称；"
                    "类比食品只提供分子谱参照，不是封闭答案集合；不得输出宏类别；"
                    "严格返回JSON对象：{开放具体食物, 支持分子, 冲突}。"
                ),
            }
        )
        baseline_rows.append(
            {
                "样本编号": str(row.get("id")),
                "冻结BM25前二名": [str(item.get("food")) for item in ledger[:2]],
                "原BM25第三名": str(ledger[2].get("food")),
            }
        )

    payload_path = args.输出目录 / "待发送数据清单.jsonl"
    payload_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in payload_rows),
        encoding="utf-8",
    )
    baseline_path = args.输出目录 / "本地配对基线.jsonl"
    baseline_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in baseline_rows),
        encoding="utf-8",
    )
    protocol = {
        "实验名称": "MFP开放具体食物科学家候选验证",
        "当前阶段": "仅准备，不调用API",
        "唯一变化": "保留BM25前二名，将原第三名替换为Scientist开放生成的一个具体食物",
        "冻结部分": "BM25模型、前二名候选、开发集、宏类别映射评测方式；Reviewer暂不参与",
        "发送字段": ["样本编号", "输入分子", "冻结BM25类比食品", "开放生成要求"],
        "明确不发送": ["原BM25第三名", "真实食物", "真实宏类别", "开发集评测结果", "正式测试集数据"],
        "计划API调用次数": len(payload_rows),
        "API调用次数": 0,
        "正式测试集是否读取": False,
        "开发集样本数": len(dev),
        "训练具体食物数": len(train_foods),
        "开发集具体食物数": len(dev_foods),
        "训练开发具体食物交集数": len(overlap),
        "候选阶段主要指标": ["Top3具体食物召回", "Top3宏类别oracle覆盖"],
        "响应格式": {"样本编号": "原样返回", "开放具体食物": "一个具体食物名称", "支持分子": ["输入分子"], "冲突": ["输入分子或简短说明"]},
        "响应合法性": ["开放具体食物非空", "不是宏类别", "不得复制冻结BM25前二名"],
        "无效响应处理": "安全回退到原BM25第三名，不让格式错误制造虚假负增益",
        "下一阶段条件": "只有开放候选提高候选召回后，才单独研究Reviewer如何选择",
        "输入文件哈希": {"训练集": 文件哈希(train_path), "开发集": 文件哈希(dev_path)},
        "待发送清单哈希": 文件哈希(payload_path),
        "本地配对基线哈希": 文件哈希(baseline_path),
    }
    if args.评估响应 is not None:
        if not args.评估响应.is_file():
            raise FileNotFoundError(f"响应文件不存在：{args.评估响应}")
        agent = base.加载模块("第五轮MFP类别映射", 项目根目录 / "code/Only-Deepseek/optimized_agent.py")
        categories = agent.load_food_categories(项目根目录 / "data/raw/flavordb.db")
        evaluation = 本地评测(args.评估响应, dev, baseline_rows, categories, base.归一化)
        evaluation["响应文件"] = str(args.评估响应)
        evaluation["响应文件哈希"] = 文件哈希(args.评估响应)
        (args.输出目录 / "开放候选本地评测.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        protocol["本地评测"] = evaluation["指标"]
    (args.输出目录 / "冻结实验方案.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
