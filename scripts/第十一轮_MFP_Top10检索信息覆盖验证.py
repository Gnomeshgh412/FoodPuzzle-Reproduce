#!/usr/bin/env python3
"""MFP 第十一轮：只把 Scientist 的 BM25 demonstrations 从 3 增加到 10。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any


根目录 = Path(__file__).resolve().parents[1]
默认输出 = 根目录 / "results/Only-Deepseek/优化实验/第十一轮/MFP_Top10检索信息覆盖"
基线目录 = 根目录 / "results/Only-Deepseek/优化实验/第十一轮/双任务瓶颈补证/MFP_完整科学家阶段分解_联网重试"
随机种子 = 20260809


def 加载(名称: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(名称, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 读_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def 写_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"禁止覆盖：{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def 写_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"禁止覆盖：{path}")
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def 规范(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def 准备(output: Path) -> None:
    bm25 = 加载("第十一轮MFP_BM25", 根目录 / "code/Only-Deepseek/bm25_icl.py")
    train = 读_jsonl(根目录 / "results/splits/mfp/train.jsonl")
    dev = 读_jsonl(根目录 / "results/splits/mfp/dev.jsonl")
    index = bm25.BM25Index(train, "mfp")
    metadata = []
    for row in dev:
        retrieved = index.retrieve(row, 10)
        metadata.append(
            {
                "id": row["id"],
                "retrieved": [
                    {
                        "id": item["id"], "rank": item["rank"], "score": item["score"],
                        "actual_food": item["actual_food"], "target_food": item["target_food"],
                    }
                    for item in retrieved
                ],
            }
        )
    if len(metadata) != 71 or any(len(row["retrieved"]) != 10 for row in metadata):
        raise RuntimeError("Top10 检索元数据不完整")
    protocol = {
        "任务输出": "具体食物名称",
        "主指标": "Reviewer 宏类别 accuracy",
        "唯一改动": "BM25 demonstrations K=3 改为固定 K=10",
        "冻结": ["Scientist Prompt", "Scientist 三候选", "Reviewer Prompt", "DeepSeek 模型", "官方证据", "起始分子数", "评测器"],
        "对照": "本轮瓶颈补证中已冻结的 K=3 完整 Agent dev 结果",
        "数据": "MFP train=567, dev=71；不读取正式 test",
        "准入": [
            "Scientist Top3 宏类别 oracle 增益的配对 bootstrap 95% 下界>0",
            "Reviewer 宏类别 accuracy 增益的配对 bootstrap 95% 下界>0",
            "Reviewer 增益在5个固定 ID 交错分块中至少4块非负",
            "71条全部解析成功，无直接宏类别输出",
        ],
        "辅助诊断": ["具体食物Top1", "具体食物Top3", "候选类别数", "Reviewer选择序号"],
        "停止条件": "任一主准入线失败即未通过并停止；不调K、Prompt或其他参数",
    }
    写_json(output / "冻结实验方案.json", protocol)
    写_jsonl(output / "开发集BM25_Top10检索元数据.jsonl", metadata)
    print(json.dumps({"状态": "准备通过", "样本数": len(metadata)}, ensure_ascii=False))


def 拆分(output: Path) -> None:
    dev = {str(row["id"]): row for row in 读_jsonl(根目录 / "results/splits/mfp/dev.jsonl")}
    metadata = 读_jsonl(output / "假设与审查元数据.jsonl")
    ranks = {1: [], 2: [], 3: []}
    reviewer_rows = []
    diagnostics = []
    category_words = {"additive", "animal product", "beverage", "cereal", "dairy", "dish", "essential oil", "fish seafood", "flower", "fruit", "fungus", "herb", "legume", "meat", "nutseed", "plant", "seed", "spice", "vegetable"}
    for record in metadata:
        row_id = str(record["id"])
        gold = 规范(dev[row_id].get("actual_food"))
        hypotheses = record.get("hypotheses") or []
        if len(hypotheses) != 3 or record.get("error"):
            raise RuntimeError(f"样本 {row_id} 不是三个有效候选")
        foods = [str(item.get("predicted_food") or "").strip() for item in hypotheses]
        reviewer = record.get("reviewer_output") or {}
        reviewer_food = str(reviewer.get("predicted_food") or "").strip()
        for rank, food in enumerate(foods, 1):
            ranks[rank].append({"id": row_id, "predicted_food": food})
        reviewer_rows.append({"id": row_id, "predicted_food": reviewer_food})
        diagnostics.append(
            {
                "id": row_id,
                "Scientist_Top1具体食物正确": 规范(foods[0]) == gold,
                "Scientist_Top3具体食物正确": gold in {规范(x) for x in foods},
                "Reviewer具体食物正确": 规范(reviewer_food) == gold,
                "Reviewer选择序号": reviewer.get("selected_hypothesis_index"),
                "Reviewer直接输出宏类别": 规范(reviewer_food) in category_words,
            }
        )
    for rank in (1, 2, 3):
        写_jsonl(output / f"Scientist第{rank}候选_官方评测输入.jsonl", ranks[rank])
    写_jsonl(output / "Reviewer最终预测_官方评测输入.jsonl", reviewer_rows)
    写_jsonl(output / "具体食物诊断.jsonl", diagnostics)
    print(json.dumps({"状态": "拆分通过", "样本数": len(diagnostics)}, ensure_ascii=False))


def bootstrap_lower(values: list[float], repeats: int = 10000) -> float:
    rng = random.Random(随机种子)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(repeats))
    return means[max(0, int(0.025 * repeats) - 1)]


def 汇总(output: Path) -> None:
    labels = ["Scientist第1候选", "Scientist第2候选", "Scientist第3候选", "Reviewer最终预测"]
    new_tables = {label: {str(row["id"]): row for row in 读_jsonl(output / f"{label}_官方评测逐样本.jsonl")} for label in labels}
    old_names = ["科学家第1候选", "科学家第2候选", "科学家第3候选", "审查器最终预测"]
    old_tables = {name: {str(row["id"]): row for row in 读_jsonl(基线目录 / f"{name}_官方类别逐样本.jsonl")} for name in old_names}
    ids = [str(row["id"]) for row in 读_jsonl(根目录 / "results/splits/mfp/dev.jsonl")]
    rows = []
    for index, row_id in enumerate(ids):
        old_candidates = [bool(old_tables[name][row_id]["correct"]) for name in old_names[:3]]
        new_candidates = [bool(new_tables[label][row_id]["correct"]) for label in labels[:3]]
        rows.append(
            {
                "id": row_id, "固定分块": index % 5,
                "K3_Scientist_oracle": any(old_candidates), "K10_Scientist_oracle": any(new_candidates),
                "K3_Reviewer正确": bool(old_tables[old_names[3]][row_id]["correct"]),
                "K10_Reviewer正确": bool(new_tables[labels[3]][row_id]["correct"]),
            }
        )
    scientist_gain = [float(row["K10_Scientist_oracle"]) - float(row["K3_Scientist_oracle"]) for row in rows]
    reviewer_gain = [float(row["K10_Reviewer正确"]) - float(row["K3_Reviewer正确"]) for row in rows]
    fold_gain = [sum(reviewer_gain[i] for i, row in enumerate(rows) if row["固定分块"] == fold) / sum(row["固定分块"] == fold for row in rows) for fold in range(5)]
    diagnostics = 读_jsonl(output / "具体食物诊断.jsonl")
    summary = {
        "样本数": len(rows),
        "K3_Scientist_Top3宏类别oracle": sum(row["K3_Scientist_oracle"] for row in rows) / len(rows),
        "K10_Scientist_Top3宏类别oracle": sum(row["K10_Scientist_oracle"] for row in rows) / len(rows),
        "Scientist_oracle平均增益": sum(scientist_gain) / len(rows),
        "Scientist_oracle增益bootstrap_95%下界": bootstrap_lower(scientist_gain),
        "K3_Reviewer宏类别accuracy": sum(row["K3_Reviewer正确"] for row in rows) / len(rows),
        "K10_Reviewer宏类别accuracy": sum(row["K10_Reviewer正确"] for row in rows) / len(rows),
        "Reviewer平均增益": sum(reviewer_gain) / len(rows),
        "Reviewer增益bootstrap_95%下界": bootstrap_lower(reviewer_gain),
        "Reviewer五个固定分块增益": fold_gain,
        "Scientist_Top1具体食物命中数": sum(row["Scientist_Top1具体食物正确"] for row in diagnostics),
        "Scientist_Top3具体食物命中数": sum(row["Scientist_Top3具体食物正确"] for row in diagnostics),
        "Reviewer具体食物命中数": sum(row["Reviewer具体食物正确"] for row in diagnostics),
        "直接宏类别输出数": sum(row["Reviewer直接输出宏类别"] for row in diagnostics),
    }
    summary["是否通过"] = bool(
        summary["Scientist_oracle增益bootstrap_95%下界"] > 0
        and summary["Reviewer增益bootstrap_95%下界"] > 0
        and sum(value >= 0 for value in fold_gain) >= 4
        and summary["直接宏类别输出数"] == 0
    )
    summary["结论"] = "通过并冻结" if summary["是否通过"] else "未通过并停止"
    写_jsonl(output / "配对阶段结果.jsonl", rows)
    写_json(output / "完整审查结果.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("动作", choices=["准备", "拆分", "汇总"])
    parser.add_argument("--输出", type=Path, default=默认输出)
    args = parser.parse_args()
    {"准备": 准备, "拆分": 拆分, "汇总": 汇总}[args.动作](args.输出)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
